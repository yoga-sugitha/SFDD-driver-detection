"""
Visualization utilities
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb
from lightning.pytorch.loggers import WandbLogger
import torch
from typing import List

def plot_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: List[str],
    logger: WandbLogger = None,
    save_path: str = None
):
    """
    Plot and optionally log confusion matrix
    
    Args:
        predictions: Predicted class indices
        targets: True class indices
        class_names: List of class names
        logger: Optional WandB logger
        save_path: Optional path to save figure
    """
    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Generate confusion matrix
    cm = confusion_matrix(targets_np, preds_np)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save locally if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    # Log to WandB
    if logger and isinstance(logger, WandbLogger):
        logger.experiment.log({
            "test/confusion_matrix": wandb.Image(plt)
        })
        print("✓ Confusion matrix logged to WandB")
    
    plt.close()
