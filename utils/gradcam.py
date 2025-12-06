"""
Efficient GradCAM that uses already-computed test predictions
No need to re-run inference - just use the stored results!
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from typing import List, Optional, Tuple
import wandb


def find_target_layer(model: nn.Module, verbose: bool = True) -> Optional[nn.Module]:
    """
    Automatically find the best layer for GradCAM in any model.
    
    Strategy:
    1. Look for common layer names (layer4, features, backbone)
    2. Find the last convolutional block before classifier
    3. Search recursively through model structure
    """
    if verbose:
        print("\n" + "="*60)
        print("Searching for GradCAM target layer...")
        print("="*60)
    
    # Strategy 1: Check common names
    common_names = ['layer4', 'layer3', 'features', 'backbone', 'blocks']
    
    for name in common_names:
        if hasattr(model, name):
            target = getattr(model, name)
            if verbose:
                print(f"✓ Found layer by name: '{name}'")
                print(f"  Type: {type(target)}")
            return target
    
    # Strategy 2: Find last convolutional sequence
    last_conv_module = None
    last_conv_name = None
    
    def find_last_conv(module, prefix=''):
        nonlocal last_conv_module, last_conv_name
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            has_conv = any(isinstance(m, nn.Conv2d) for m in child.modules())
            
            if has_conv:
                if 'fc' not in name.lower() and 'classifier' not in name.lower():
                    last_conv_module = child
                    last_conv_name = full_name
            
            find_last_conv(child, full_name)
    
    find_last_conv(model)
    
    if last_conv_module is not None:
        if verbose:
            print(f"✓ Found last convolutional block: '{last_conv_name}'")
        return last_conv_module
    
    # Strategy 3: Get all modules and find last Conv2d
    all_modules = list(model.modules())
    for module in reversed(all_modules):
        if isinstance(module, nn.Conv2d):
            if verbose:
                print(f"✓ Found last Conv2d layer")
            return module
    
    if verbose:
        print("✗ Could not find suitable layer for GradCAM")
    
    return None


class GradCAMHelper:
    """GradCAM helper with automatic layer detection"""
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        
        if target_layer is None:
            target_layer = find_target_layer(model)
            if target_layer is None:
                raise ValueError("Could not automatically find target layer")
        
        self.target_layer = target_layer
        self.features = None
        self.gradients = None
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.features = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        fwd_hook = self.target_layer.register_forward_hook(forward_hook)
        bwd_hook = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks = [fwd_hook, bwd_hook]
    
    def generate_cam(self, class_idx: int) -> torch.Tensor:
        if self.gradients is None or self.features is None:
            raise RuntimeError("Forward and backward passes must be completed first")
        
        weights = torch.mean(self.gradients[class_idx], dim=[1, 2], keepdim=True)
        cam = torch.relu((weights * self.features[class_idx]).sum(dim=0))
        
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def select_samples_from_test_results(
    all_preds: torch.Tensor,
    all_targets: torch.Tensor,
    num_correct: int = 2,
    num_incorrect: int = 2
) -> Tuple[List[int], List[int]]:
    """
    Select sample indices from already-computed test results.
    This is MUCH faster than re-running inference!
    
    Args:
        all_preds: Tensor of all predictions from test set [N]
        all_targets: Tensor of all ground truth labels [N]
        num_correct: Number of correct predictions needed
        num_incorrect: Number of incorrect predictions needed
        
    Returns:
        Tuple of (correct_indices, incorrect_indices)
        
    Raises:
        RuntimeError: If insufficient samples found
    """
    print("\n" + "="*60)
    print("Selecting samples from test results...")
    print(f"Target: {num_correct} correct + {num_incorrect} incorrect")
    print("="*60)
    
    # Move to CPU for easier manipulation
    all_preds = all_preds.cpu()
    all_targets = all_targets.cpu()
    
    # Find correct and incorrect predictions
    correct_mask = (all_preds == all_targets)
    incorrect_mask = ~correct_mask
    
    correct_indices = torch.where(correct_mask)[0].tolist()
    incorrect_indices = torch.where(incorrect_mask)[0].tolist()
    
    print(f"\nAvailable samples:")
    print(f"  Correct predictions: {len(correct_indices)}")
    print(f"  Incorrect predictions: {len(incorrect_indices)}")
    
    # Check if we have enough
    if len(correct_indices) < num_correct:
        raise RuntimeError(
            f"\n❌ INSUFFICIENT CORRECT PREDICTIONS!\n"
            f"   Requested: {num_correct}, Available: {len(correct_indices)}\n"
            f"   Model accuracy may be too low.\n"
            f"   Try reducing num_correct or training the model longer."
        )
    
    if len(incorrect_indices) < num_incorrect:
        raise RuntimeError(
            f"\n❌ INSUFFICIENT INCORRECT PREDICTIONS!\n"
            f"   Requested: {num_incorrect}, Available: {len(incorrect_indices)}\n"
            f"   Model is very accurate (good news!).\n"
            f"   Try reducing num_incorrect to {len(incorrect_indices)} or less."
        )
    
    # Select the requested number
    selected_correct = correct_indices[:num_correct]
    selected_incorrect = incorrect_indices[:num_incorrect]
    
    print(f"\n✅ Selected:")
    print(f"   Correct: {len(selected_correct)} indices")
    print(f"   Incorrect: {len(selected_incorrect)} indices")
    print("="*60)
    
    return selected_correct, selected_incorrect


def generate_gradcam_visualizations(
    model: L.LightningModule,
    data_module: L.LightningDataModule,
    logger: WandbLogger,
    class_names: List[str],
    num_correct: int = 2,
    num_incorrect: int = 2,
):
    """
    Generate GradCAM visualizations efficiently using already-computed test results.
    
    This version:
    - Uses predictions already stored in model.all_test_preds/all_test_targets
    - Only loads the specific images needed (no batch iteration)
    - Much faster than the original implementation
    
    Args:
        model: Trained Lightning module (must have all_test_preds/all_test_targets)
        data_module: Data module with test set
        logger: WandB logger for logging visualizations
        class_names: List of class names
        num_correct: Number of correct predictions (default: 2)
        num_incorrect: Number of incorrect predictions (default: 2)
        
    Raises:
        RuntimeError: If insufficient correct or incorrect samples found
        ValueError: If test results not available
    """
    if not isinstance(logger, WandbLogger):
        print("⚠ GradCAM requires WandB logger")
        return
    
    # Check if test results are available
    if not hasattr(model, 'all_test_preds') or not hasattr(model, 'all_test_targets'):
        raise ValueError(
            "Test results not available! Make sure to run trainer.test() first "
            "and that the model stores predictions in all_test_preds/all_test_targets."
        )
    
    print("\n" + "="*60)
    print("Generating GradCAM Visualizations (Efficient Method)")
    print(f"Using pre-computed test results: {len(model.all_test_preds)} samples")
    print("="*60)
    
    device = next(model.parameters()).device
    
    # Setup GradCAM helper
    try:
        gradcam_helper = GradCAMHelper(model.model, target_layer=None)
        print(f"✓ GradCAM helper initialized")
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        return
    
    # Select samples from test results (FAST - no inference needed!)
    try:
        correct_indices, incorrect_indices = select_samples_from_test_results(
            all_preds=model.all_test_preds,
            all_targets=model.all_test_targets,
            num_correct=num_correct,
            num_incorrect=num_incorrect
        )
    except RuntimeError as e:
        print(f"\n{e}")
        gradcam_helper.cleanup()
        return
    
    # Combine indices (correct first, then incorrect)
    all_indices = correct_indices + incorrect_indices
    
    print(f"\n✓ Processing {len(all_indices)} samples")
    print(f"   First {num_correct} are CORRECT")
    print(f"   Last {num_incorrect} are INCORRECT")
    print("="*60 + "\n")
    
    gradcam_images = []
    
    # Process each selected sample
    for sample_num, test_idx in enumerate(all_indices, 1):
        is_correct_section = (sample_num <= num_correct)
        
        try:
            # Load ONLY this specific image (no batch loading!)
            # This is MUCH faster than iterating through batches
            img_tensor, true_label = data_module.get_test_sample(test_idx)
            img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get prediction (stored from test)
            pred_label = model.all_test_preds[test_idx].item()
            is_actually_correct = (pred_label == true_label)
            
            # Generate GradCAM
            with torch.set_grad_enabled(True):
                img_tensor.requires_grad_(True)
                model.model.zero_grad()
                
                preds = model(img_tensor)
                score = preds[0, pred_label]
                score.backward()
            
            # Generate CAM
            cam = gradcam_helper.generate_cam(0)
            cam = cam.unsqueeze(0).unsqueeze(0)
            cam = nn.functional.interpolate(
                cam, size=img_tensor.shape[-2:], 
                mode='bilinear', align_corners=False
            ).squeeze()
            
            # Denormalize for visualization
            img = img_tensor[0].cpu().detach()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(img.permute(1, 2, 0).numpy())
            axes[0].set_title("Original", fontsize=12)
            axes[0].axis("off")
            
            axes[1].imshow(cam.cpu().numpy(), cmap='jet')
            axes[1].set_title("GradCAM Heatmap", fontsize=12)
            axes[1].axis("off")
            
            axes[2].imshow(img.permute(1, 2, 0).numpy())
            axes[2].imshow(cam.cpu().numpy(), cmap='jet', alpha=0.5)
            axes[2].set_title("Overlay", fontsize=12)
            axes[2].axis("off")
            
            # Add labels
            true_name = class_names[true_label]
            pred_name = class_names[pred_label]
            color = "green" if is_actually_correct else "red"
            status = "✓ CORRECT" if is_actually_correct else "✗ INCORRECT"
            
            fig.suptitle(
                f"{status} | True: {true_name} | Predicted: {pred_name}", 
                color=color, fontsize=14, fontweight='bold'
            )
            plt.tight_layout()
            
            gradcam_images.append(wandb.Image(fig))
            
            plt.close(fig)
            plt.close('all')
            
            # Verify sample is in correct section
            if is_correct_section and not is_actually_correct:
                print(f"  [{sample_num}/{len(all_indices)}] ⚠️  {status} (UNEXPECTED IN CORRECT SECTION)")
            elif not is_correct_section and is_actually_correct:
                print(f"  [{sample_num}/{len(all_indices)}] ⚠️  {status} (UNEXPECTED IN INCORRECT SECTION)")
            else:
                print(f"  [{sample_num}/{len(all_indices)}] ✓ {status} | True: {true_name}")
            
        except Exception as e:
            print(f"  [{sample_num}/{len(all_indices)}] ✗ Failed: {e}")
            plt.close('all')
            continue
    
    # Log results
    print("\n" + "="*60)
    print("✅ GradCAM Generation Complete!")
    print("="*60)
    print(f"  Generated: {len(gradcam_images)}/{len(all_indices)} visualizations")
    print("="*60 + "\n")
    
    if gradcam_images:
        logger.experiment.log({
            "test/gradcam_samples": gradcam_images,
            "test/gradcam_total": len(gradcam_images),
            "test/gradcam_requested_correct": num_correct,
            "test/gradcam_requested_incorrect": num_incorrect,
        })
    else:
        print("\n⚠ No GradCAM visualizations generated")
    
    # Cleanup
    gradcam_helper.cleanup()
    del gradcam_images
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()