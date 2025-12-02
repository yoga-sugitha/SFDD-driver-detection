"""
GradCAM implementation for explainability
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from typing import List
import wandb

class GradCAMHelper:
    """
    Encapsulated GradCAM functionality
    Handles hook registration, gradient capture, and visualization
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.features = None
        self.gradients = None
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer"""
        def forward_hook(module, input, output):
            self.features = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        fwd_hook = self.target_layer.register_forward_hook(forward_hook)
        bwd_hook = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks = [fwd_hook, bwd_hook]
    
    def generate_cam(self, class_idx: int) -> torch.Tensor:
        """
        Generate CAM for a specific class
        
        Args:
            class_idx: Index within the batch dimension
            
        Returns:
            CAM heatmap [H, W]
        """
        if self.gradients is None or self.features is None:
            raise RuntimeError("Forward and backward passes must be completed first")
        
        # Global average pooling of gradients
        weights = torch.mean(self.gradients[class_idx], dim=[1, 2], keepdim=True)
        
        # Weighted combination of feature maps
        cam = torch.relu((weights * self.features[class_idx]).sum(dim=0))
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def cleanup(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def generate_gradcam_visualizations(
    model: L.LightningModule,
    data_module: L.LightningDataModule,
    logger: WandbLogger,
    class_names: List[str],
    num_samples: int = 6
):
    """
    Generate GradCAM visualizations with proper memory management
    
    Args:
        model: Trained Lightning module
        data_module: Data module with test set
        logger: WandB logger for logging visualizations
        class_names: List of class names
        num_samples: Number of samples to visualize
    """
    if not isinstance(logger, WandbLogger):
        print("⚠ GradCAM requires WandB logger")
        return
    
    print("\n" + "="*60)
    print("Generating GradCAM Visualizations...")
    print("="*60)
    
    device = next(model.parameters()).device
    
    # Setup GradCAM helper
    if hasattr(model.model, 'layer4'):
        target_layer = model.model.layer4
    else:
        print("⚠ Model does not have 'layer4' for GradCAM")
        return
    
    gradcam_helper = GradCAMHelper(model.model, target_layer)
    
    # Get test data
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    
    # Get first batch
    x, y = next(iter(test_loader))
    x = x.to(device)
    y = y.to(device)
    
    # Keep model in eval mode
    model.model.eval()
    
    # Get predictions without gradients first
    with torch.no_grad():
        preds = model(x)
        predicted = preds.argmax(dim=1)
    
    # Find correct and incorrect predictions
    correct_mask = (predicted == y)
    incorrect_mask = ~correct_mask
    
    num_correct_available = correct_mask.sum().item()
    num_incorrect_available = incorrect_mask.sum().item()
    
    print(f"Available samples - Correct: {num_correct_available}, Incorrect: {num_incorrect_available}")
    
    # Strategy: Try to get equal split, but if one type is scarce, use what's available
    target_per_type = num_samples // 2
    
    if num_correct_available >= target_per_type and num_incorrect_available >= target_per_type:
        # Ideal case: enough of both types
        num_correct = target_per_type
        num_incorrect = target_per_type
    elif num_correct_available + num_incorrect_available < num_samples:
        # Not enough total samples
        num_correct = num_correct_available
        num_incorrect = num_incorrect_available
    else:
        # One type is scarce, take all of scarce type and fill the rest
        if num_correct_available < target_per_type:
            num_correct = num_correct_available
            num_incorrect = min(num_samples - num_correct, num_incorrect_available)
        else:
            num_incorrect = num_incorrect_available
            num_correct = min(num_samples - num_incorrect, num_correct_available)
    
    print(f"Selecting - Correct: {num_correct}, Incorrect: {num_incorrect}, Total: {num_correct + num_incorrect}")
    
    correct_indices = torch.where(correct_mask)[0][:num_correct] if num_correct > 0 else torch.tensor([], dtype=torch.long)
    incorrect_indices = torch.where(incorrect_mask)[0][:num_incorrect] if num_incorrect > 0 else torch.tensor([], dtype=torch.long)
    
    # Concatenate indices
    if len(correct_indices) > 0 and len(incorrect_indices) > 0:
        selected_indices = torch.cat([correct_indices, incorrect_indices])
    elif len(correct_indices) > 0:
        selected_indices = correct_indices
    elif len(incorrect_indices) > 0:
        selected_indices = incorrect_indices
    else:
        selected_indices = torch.tensor([], dtype=torch.long)
    
    if len(selected_indices) == 0:
        print("⚠ No samples found for GradCAM visualization")
        gradcam_helper.cleanup()
        return
    
    gradcam_images = []
    
    # Process samples with proper gradient handling
    for idx in selected_indices:
        idx_item = idx.item()
        
        # Clean up previous iteration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Enable gradients only for this sample
            with torch.set_grad_enabled(True):
                x_sample = x[idx_item:idx_item+1].clone().detach()
                x_sample.requires_grad_(True)
                
                # Zero gradients
                model.model.zero_grad()
                if x_sample.grad is not None:
                    x_sample.grad.zero_()
                
                # Forward pass
                preds_sample = model(x_sample)
                target_class = predicted[idx_item]
                score = preds_sample[0, target_class]
                
                # Backward pass
                score.backward()
            
            # Generate CAM
            cam = gradcam_helper.generate_cam(0)
            cam = cam.unsqueeze(0).unsqueeze(0)
            cam = nn.functional.interpolate(
                cam, size=x.shape[-2:], 
                mode='bilinear', align_corners=False
            ).squeeze()
            
            # Visualize
            img = x[idx_item].cpu().detach()
            
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Original image
            axes[0].imshow(img.permute(1, 2, 0).numpy())
            axes[0].set_title("Original", fontsize=12)
            axes[0].axis("off")
            
            # Heatmap
            axes[1].imshow(cam.cpu().numpy(), cmap='jet')
            axes[1].set_title("GradCAM Heatmap", fontsize=12)
            axes[1].axis("off")
            
            # Overlay
            axes[2].imshow(img.permute(1, 2, 0).numpy())
            axes[2].imshow(cam.cpu().numpy(), cmap='jet', alpha=0.5)
            axes[2].set_title("Overlay", fontsize=12)
            axes[2].axis("off")
            
            # Labels
            true_label = class_names[y[idx_item].item()]
            pred_label = class_names[predicted[idx_item].item()]
            is_correct = (y[idx_item] == predicted[idx_item]).item()
            color = "green" if is_correct else "red"
            
            fig.suptitle(
                f"True: {true_label} | Predicted: {pred_label}", 
                color=color, fontsize=14, fontweight='bold'
            )
            plt.tight_layout()
            
            # Convert to WandB image
            gradcam_images.append(wandb.Image(fig))
            
            # Close figure and clear memory
            plt.close(fig)
            plt.close('all')
            
            # Clear tensors
            del x_sample, preds_sample, score, cam, img
            
            print(f"✓ Generated GradCAM for sample {idx_item}")
            
        except Exception as e:
            print(f"✗ Failed to generate GradCAM for sample {idx_item}: {e}")
            plt.close('all')
            continue
    
    # Log to WandB with detailed statistics
    if gradcam_images:
        # Count actual correct/incorrect in generated images
        actual_correct = sum(1 for idx in selected_indices[:len(gradcam_images)] 
                           if (predicted[idx] == y[idx]).item())
        actual_incorrect = len(gradcam_images) - actual_correct
        
        logger.experiment.log({
            "test/gradcam_samples": gradcam_images,
            "test/gradcam_count": len(gradcam_images),
            "test/gradcam_correct": actual_correct,
            "test/gradcam_incorrect": actual_incorrect
        })
        print(f"\n✓ Logged {len(gradcam_images)} GradCAM visualizations to WandB")
        print(f"  - Correct predictions: {actual_correct}")
        print(f"  - Incorrect predictions: {actual_incorrect}")
    else:
        print("\n⚠ No GradCAM visualizations generated")
    
    # Cleanup
    gradcam_helper.cleanup()
    del gradcam_images
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Ensure model stays in eval mode
    model.model.eval() # pyright: ignore[reportAttributeAccessIssue]