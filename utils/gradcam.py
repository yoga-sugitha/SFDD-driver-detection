# utils/gradcam_fast.py

import torch
import matplotlib.pyplot as plt
import wandb
import lightning as L
from typing import List
from models.gradcam import GradCAMHelper  # assuming your GradCAMHelper is in models/gradcam.py or utils/gradcam.py

def generate_gradcam_from_cached_predictions(
    model: L.LightningModule,
    data_module,
    logger,
    class_names: List[str],
    all_test_preds: torch.Tensor,
    all_test_targets: torch.Tensor,
    num_correct: int = 2,
    num_incorrect: int = 2,
    inspect_architecture: bool = False
):
    """
    Generate GradCAM on pre-identified correct/incorrect samples.
    Uses predictions from test loop — NO extra inference over full test set.
    """
    if not hasattr(logger, 'experiment'):
        print("⚠ Skipping GradCAM: logger not compatible")
        return

    device = next(model.parameters()).device
    model.eval()

    # Identify correct/incorrect indices
    correct_mask = (all_test_preds == all_test_targets)
    incorrect_mask = ~correct_mask

    correct_indices = torch.where(correct_mask)[0]
    incorrect_indices = torch.where(incorrect_mask)[0]

    if len(correct_indices) < num_correct:
        print(f"⚠ Only {len(correct_indices)} correct samples found. Using all available.")
        num_correct = len(correct_indices)
    if len(incorrect_indices) < num_incorrect:
        print(f"⚠ Only {len(incorrect_indices)} incorrect samples found. Using all available.")
        num_incorrect = len(incorrect_indices)

    if num_correct == 0 and num_incorrect == 0:
        print("⚠ No samples to visualize for GradCAM")
        return

    selected_indices = torch.cat([
        correct_indices[:num_correct],
        incorrect_indices[:num_incorrect]
    ]).cpu().tolist()

    # Optional: inspect model
    if inspect_architecture:
        from models.gradcam import inspect_model_architecture  # or wherever it lives
        inspect_model_architecture(model.model)

    # Initialize GradCAM
    try:
        gradcam = GradCAMHelper(model.model)
    except Exception as e:
        print(f"⚠ GradCAM init failed: {e}")
        return

    gradcam_images = []

    for i, idx in enumerate(selected_indices):
        try:
            # Load single image
            img, label = data_module.get_test_sample(idx)
            img = img.unsqueeze(0).to(device)  # Add batch dim
            label = torch.tensor(label).to(device)
            pred = all_test_preds[idx].to(device)

            # Enable gradients
            img.requires_grad_(True)
            model.model.zero_grad()

            # Forward + backward
            output = model(img)
            score = output[0, pred]
            score.backward()

            # Generate CAM
            cam = gradcam.generate_cam(0).cpu()

            # Denormalize for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_vis = img[0].cpu() * std + mean
            img_vis = torch.clamp(img_vis, 0, 1)

            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_vis.permute(1, 2, 0))
            axes[1].imshow(cam, cmap='jet')
            axes[2].imshow(img_vis.permute(1, 2, 0))
            axes[2].imshow(cam, cmap='jet', alpha=0.5)

            true_name = class_names[label.item()]
            pred_name = class_names[pred.item()]
            is_correct = (pred == label).item()
            status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
            color = "green" if is_correct else "red"

            fig.suptitle(f"{status} | True: {true_name} | Pred: {pred_name}", color=color, fontweight='bold')
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()

            gradcam_images.append(wandb.Image(fig))
            plt.close(fig)

        except Exception as e:
            print(f"⚠ Failed to process sample {idx}: {e}")
            plt.close('all')
            continue

    # Log to WandB
    if gradcam_images:
        logger.experiment.log({
            "test/gradcam_samples": gradcam_images,
            "test/gradcam_total": len(gradcam_images),
            "test/gradcam_correct": num_correct,
            "test/gradcam_incorrect": num_incorrect
        })
        print(f"✅ Logged {len(gradcam_images)} GradCAM visualizations to WandB")
    else:
        print("⚠ No GradCAM images generated")

    gradcam.cleanup()
    torch.cuda.empty_cache()