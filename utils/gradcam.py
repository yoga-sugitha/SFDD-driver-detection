"""
GradCAM with GUARANTEED exact counts of correct + incorrect predictions
Fixed version that properly handles insufficient samples
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
    
    Args:
        model: PyTorch model
        verbose: Print search information
        
    Returns:
        Target layer for GradCAM or None if not found
    """
    if verbose:
        print("\n" + "="*60)
        print("Searching for GradCAM target layer...")
        print("="*60)
    
    # Strategy 1: Check common names (ResNet, VGG, etc.)
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
            
            # Check if this module or its children contain Conv2d
            has_conv = any(isinstance(m, nn.Conv2d) for m in child.modules())
            
            if has_conv:
                # Don't include classifier/fc layers
                if 'fc' not in name.lower() and 'classifier' not in name.lower():
                    last_conv_module = child
                    last_conv_name = full_name
            
            # Recurse
            find_last_conv(child, full_name)
    
    find_last_conv(model)
    
    if last_conv_module is not None:
        if verbose:
            print(f"✓ Found last convolutional block: '{last_conv_name}'")
            print(f"  Type: {type(last_conv_module)}")
        return last_conv_module
    
    # Strategy 3: Get all modules and find last Conv2d
    all_modules = list(model.modules())
    for module in reversed(all_modules):
        if isinstance(module, nn.Conv2d):
            if verbose:
                print(f"✓ Found last Conv2d layer")
                print(f"  Type: {type(module)}")
            return module
    
    if verbose:
        print("✗ Could not find suitable layer for GradCAM")
        print("\nModel structure:")
        print(model)
    
    return None


def inspect_model_architecture(model: nn.Module) -> dict:
    """
    Inspect model architecture to understand its structure.
    Useful for debugging and finding the right layer.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with architecture information
    """
    info = {
        "total_layers": 0,
        "conv_layers": [],
        "sequential_blocks": [],
        "named_modules": {}
    }
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE INSPECTION")
    print("="*60)
    
    # Count and list all layers
    for name, module in model.named_modules():
        if name:  # Skip root
            info["named_modules"][name] = type(module).__name__
            info["total_layers"] += 1
            
            if isinstance(module, nn.Conv2d):
                info["conv_layers"].append(name)
            elif isinstance(module, nn.Sequential):
                info["sequential_blocks"].append(name)
    
    # Print summary
    print(f"\nTotal modules: {info['total_layers']}")
    print(f"Conv2d layers: {len(info['conv_layers'])}")
    print(f"Sequential blocks: {len(info['sequential_blocks'])}")
    
    # Print first-level children (main structure)
    print("\n" + "-"*60)
    print("First-level structure:")
    print("-"*60)
    for name, child in model.named_children():
        has_conv = sum(1 for m in child.modules() if isinstance(m, nn.Conv2d))
        print(f"  {name:20s} | {type(child).__name__:25s} | Conv2d: {has_conv}")
    
    # Print last few conv layers (most relevant for GradCAM)
    if info["conv_layers"]:
        print("\n" + "-"*60)
        print("Last 5 Conv2d layers (best for GradCAM):")
        print("-"*60)
        for name in info["conv_layers"][-5:]:
            print(f"  {name}")
    
    # Suggest best layers
    print("\n" + "-"*60)
    print("Recommended layers for GradCAM:")
    print("-"*60)
    
    suggestions = []
    for name in ['layer4', 'layer3', 'blocks', 'features']:
        if name in info["named_modules"]:
            suggestions.append(name)
    
    if suggestions:
        for s in suggestions:
            print(f"  ✓ {s}")
    else:
        if info["sequential_blocks"]:
            print(f"  ⚠ Try: {info['sequential_blocks'][-1]}")
        elif info["conv_layers"]:
            print(f"  ⚠ Try: {info['conv_layers'][-1]}")
    
    print("="*60 + "\n")
    
    return info


class GradCAMHelper:
    """GradCAM helper with automatic layer detection"""
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        
        # Auto-detect target layer if not provided
        if target_layer is None:
            target_layer = find_target_layer(model)
            if target_layer is None:
                raise ValueError(
                    "Could not automatically find target layer. "
                    "Please specify target_layer manually or inspect model with "
                    "inspect_model_architecture(model)"
                )
        
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


def collect_samples_across_batches(
    model: L.LightningModule,
    dataloader,
    device,
    num_correct_needed: int = 3,
    num_incorrect_needed: int = 3,
    max_batches: int = 50  # Increased default
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect samples across multiple batches to GUARANTEE we have EXACTLY
    the requested number of correct and incorrect predictions.
    
    Args:
        model: Lightning module
        dataloader: Data loader
        device: Device to use
        num_correct_needed: Number of correct predictions needed
        num_incorrect_needed: Number of incorrect predictions needed
        max_batches: Maximum batches to search through
        
    Returns:
        Tuple of (images, labels, predictions, selected_indices)
        
    Raises:
        RuntimeError: If insufficient samples found after max_batches
    """
    print("\n" + "="*60)
    print("Collecting samples across batches...")
    print(f"Target: {num_correct_needed} correct + {num_incorrect_needed} incorrect")
    print("="*60)
    
    all_images = []
    all_labels = []
    all_predictions = []
    
    correct_samples = []
    incorrect_samples = []
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            x = x.to(device)
            y = y.to(device)
            
            preds = model(x).argmax(dim=1)
            
            # Store all data
            batch_start_idx = len(all_images)
            all_images.append(x)
            all_labels.append(y)
            all_predictions.append(preds)
            
            # Find correct and incorrect in this batch
            for i in range(len(x)):
                global_idx = batch_start_idx + i
                
                if preds[i] == y[i]:
                    if len(correct_samples) < num_correct_needed:
                        correct_samples.append(global_idx)
                else:
                    if len(incorrect_samples) < num_incorrect_needed:
                        incorrect_samples.append(global_idx)
            
            # Check if we have enough
            if len(correct_samples) >= num_correct_needed and len(incorrect_samples) >= num_incorrect_needed:
                print(f"\n✓ Found enough samples after {batch_idx + 1} batches")
                break
            
            print(f"  Batch {batch_idx + 1}: Correct={len(correct_samples)}/{num_correct_needed}, "
                  f"Incorrect={len(incorrect_samples)}/{num_incorrect_needed}")
    
    # Concatenate all collected data
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    
    # Final check
    actual_correct = len(correct_samples)
    actual_incorrect = len(incorrect_samples)
    
    print(f"\n" + "-"*60)
    print(f"Collection Summary:")
    print(f"  Total samples processed: {len(all_images)}")
    print(f"  Correct samples found: {actual_correct}/{num_correct_needed}")
    print(f"  Incorrect samples found: {actual_incorrect}/{num_incorrect_needed}")
    print("-"*60)
    
    # CRITICAL FIX: Ensure we have EXACTLY what was requested
    if actual_correct < num_correct_needed:
        error_msg = (
            f"\n❌ INSUFFICIENT CORRECT PREDICTIONS!\n"
            f"   Requested: {num_correct_needed}, Found: {actual_correct}\n"
            f"   Searched through {len(all_images)} samples in {batch_idx + 1} batches.\n\n"
            f"   Possible solutions:\n"
            f"   1. Increase max_batches (current: {max_batches})\n"
            f"   2. Reduce num_correct (model accuracy may be too low)\n"
            f"   3. Use a larger test dataset\n"
            f"   4. Check if model is trained properly"
        )
        raise RuntimeError(error_msg)
    
    if actual_incorrect < num_incorrect_needed:
        error_msg = (
            f"\n❌ INSUFFICIENT INCORRECT PREDICTIONS!\n"
            f"   Requested: {num_incorrect_needed}, Found: {actual_incorrect}\n"
            f"   Searched through {len(all_images)} samples in {batch_idx + 1} batches.\n\n"
            f"   This usually means your model is very accurate (good news!).\n\n"
            f"   Possible solutions:\n"
            f"   1. Increase max_batches (current: {max_batches})\n"
            f"   2. Reduce num_incorrect to {actual_incorrect} or less\n"
            f"   3. Use a larger test dataset\n"
            f"   4. Accept that your model is performing well!"
        )
        raise RuntimeError(error_msg)
    
    # Take EXACTLY the requested amounts
    correct_samples = correct_samples[:num_correct_needed]
    incorrect_samples = incorrect_samples[:num_incorrect_needed]
    
    print(f"\n✅ GUARANTEED SELECTION:")
    print(f"   Correct predictions: {len(correct_samples)} (requested: {num_correct_needed})")
    print(f"   Incorrect predictions: {len(incorrect_samples)} (requested: {num_incorrect_needed})")
    print(f"   Total: {len(correct_samples) + len(incorrect_samples)}")
    print("="*60)
    
    # Combine selected indices (correct first, then incorrect)
    selected_indices = torch.tensor(correct_samples + incorrect_samples, dtype=torch.long)
    
    return all_images, all_labels, all_predictions, selected_indices


def generate_gradcam_visualizations(
    model: L.LightningModule,
    data_module: L.LightningDataModule,
    logger: WandbLogger,
    class_names: List[str],
    num_correct: int = 3,
    num_incorrect: int = 3,
    max_batches: int = 50,
    inspect_architecture: bool = False
):
    """
    Generate GradCAM visualizations with GUARANTEED exact counts.
    Will raise an error if insufficient samples are found.
    
    Args:
        model: Trained Lightning module
        data_module: Data module with test set
        logger: WandB logger for logging visualizations
        class_names: List of class names
        num_correct: Number of correct predictions (default: 3)
        num_incorrect: Number of incorrect predictions (default: 3)
        max_batches: Maximum batches to search (default: 50)
        inspect_architecture: If True, print detailed model architecture
        
    Raises:
        RuntimeError: If insufficient correct or incorrect samples found
    """
    if not isinstance(logger, WandbLogger):
        print("⚠ GradCAM requires WandB logger")
        return
    
    print("\n" + "="*60)
    print("Generating GradCAM Visualizations")
    print(f"Guaranteed: {num_correct} correct + {num_incorrect} incorrect")
    print("="*60)
    
    device = next(model.parameters()).device
    
    # Optional: Inspect architecture
    if inspect_architecture:
        inspect_model_architecture(model.model)
    
    # Setup GradCAM helper (auto-detect target layer)
    try:
        gradcam_helper = GradCAMHelper(model.model, target_layer=None)
        print(f"✓ GradCAM helper initialized with target layer: {type(gradcam_helper.target_layer).__name__}")
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print("\nTry running with inspect_architecture=True:")
        print("  generate_gradcam_visualizations(..., inspect_architecture=True)")
        return
    
    # Setup data
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    
    # Collect samples - will raise error if insufficient samples
    try:
        all_images, all_labels, all_predictions, selected_indices = collect_samples_across_batches(
            model=model,
            dataloader=test_loader,
            device=device,
            num_correct_needed=num_correct,
            num_incorrect_needed=num_incorrect,
            max_batches=max_batches
        )
    except RuntimeError as e:
        print(f"\n{e}")
        gradcam_helper.cleanup()
        return
    
    print(f"\n✓ Processing {len(selected_indices)} samples (EXACTLY as requested)")
    print("="*60)
    
    gradcam_images = []
    successful_correct = 0
    successful_incorrect = 0
    
    # Process each selected sample
    for sample_num, idx in enumerate(selected_indices, 1):
        idx_item = idx.item()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Enable gradients for this sample
            with torch.set_grad_enabled(True):
                x_sample = all_images[idx_item:idx_item+1].clone().detach()
                x_sample.requires_grad_(True)
                
                model.model.zero_grad()
                if x_sample.grad is not None:
                    x_sample.grad.zero_()
                
                preds_sample = model(x_sample)
                target_class = all_predictions[idx_item]
                score = preds_sample[0, target_class]
                
                score.backward()
            
            # Generate CAM
            cam = gradcam_helper.generate_cam(0)
            cam = cam.unsqueeze(0).unsqueeze(0)
            cam = nn.functional.interpolate(
                cam, size=all_images.shape[-2:], 
                mode='bilinear', align_corners=False
            ).squeeze()
            
            # Visualize
            img = all_images[idx_item].cpu().detach()
            
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            
            # Create figure
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
            
            # Labels
            true_label = class_names[all_labels[idx_item].item()]
            pred_label = class_names[all_predictions[idx_item].item()]
            is_correct = (all_labels[idx_item] == all_predictions[idx_item]).item()
            color = "green" if is_correct else "red"
            status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
            
            if is_correct:
                successful_correct += 1
            else:
                successful_incorrect += 1
            
            fig.suptitle(
                f"{status} | True: {true_label} | Predicted: {pred_label}", 
                color=color, fontsize=14, fontweight='bold'
            )
            plt.tight_layout()
            
            gradcam_images.append(wandb.Image(fig))
            
            plt.close(fig)
            plt.close('all')
            
            del x_sample, preds_sample, score, cam, img
            
            print(f"  [{sample_num}/{len(selected_indices)}] ✓ {status:15s} | True: {true_label:20s} | Pred: {pred_label}")
            
        except Exception as e:
            print(f"  [{sample_num}/{len(selected_indices)}] ✗ Failed: {e}")
            plt.close('all')
            continue
    
    # Verify final counts match request
    print("\n" + "="*60)
    print("✅ GradCAM Generation Complete!")
    print("="*60)
    print(f"  Requested:  {num_correct} correct + {num_incorrect} incorrect = {num_correct + num_incorrect} total")
    print(f"  Generated:  {successful_correct} correct + {successful_incorrect} incorrect = {len(gradcam_images)} total")
    
    if successful_correct != num_correct or successful_incorrect != num_incorrect:
        print(f"\n  ⚠ WARNING: Some visualizations failed during generation!")
        print(f"     This is usually due to gradient computation errors.")
    
    print("="*60 + "\n")
    
    # Log to WandB
    if gradcam_images:
        logger.experiment.log({
            "test/gradcam_samples": gradcam_images,
            "test/gradcam_total": len(gradcam_images),
            "test/gradcam_correct": successful_correct,
            "test/gradcam_incorrect": successful_incorrect,
            "test/gradcam_requested_correct": num_correct,
            "test/gradcam_requested_incorrect": num_incorrect
        })
    else:
        print("\n⚠ No GradCAM visualizations generated")
    
    # Cleanup
    gradcam_helper.cleanup()
    del gradcam_images
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model.model.eval()