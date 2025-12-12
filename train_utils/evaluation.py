# train_utils/evaluation.py
from pathlib import Path
import os
import torch
import lightning as L
from omegaconf import DictConfig
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from modules.lightning_module import LightningModule
from utils.metrics import compute_model_complexity, measure_inference_latency
from utils.visualization import plot_confusion_matrix
from utils.gradcam import generate_gradcam_visualizations


def evaluate_model_complexity(model: LightningModule, cfg: DictConfig) -> dict:
    print(f"\n{'='*70}")
    print("Evaluating Model Complexity...")
    print(f"{'='*70}\n")
    
    original_device = next(model.parameters()).device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    complexity_metrics = compute_model_complexity(
        model.model.to(device),
        input_size=(3, cfg.data.hparams.img_size, cfg.data.hparams.img_size)
    )
    model.model.to(original_device)
    
    print(f"✓ Model Parameters: {complexity_metrics['num_params']:,}")
    if complexity_metrics['flops'] > 0:
        print(f"✓ FLOPs: {complexity_metrics['flops']:,}")
    
    return complexity_metrics


def evaluate_inference_latency(model: LightningModule, data_module: L.LightningDataModule) -> dict:
    """
    Evaluate inference latency
    
    Args:
        model: Lightning model
        data_module: Data module with test data
        
    Returns:
        Dictionary of latency metrics
    """
    print(f"\n{'='*70}")
    print("Measuring Inference Latency...")
    print(f"{'='*70}\n")
    
    # Ensure test data is ready
    if not hasattr(data_module, 'test_dataset') or data_module.test_dataset is None:
        data_module.setup(stage="test")
    
    test_loader = data_module.test_dataloader()
    
    # Measure latency
    latency_metrics = measure_inference_latency(
        model,
        test_loader,
        num_batches=min(50, len(test_loader))  # Use available batches
    )
    
    print(f"✓ Avg Latency: {latency_metrics['avg_inference_latency_ms']:.2f} ms")
    
    return latency_metrics

def cleanup_distributed():
    """Cleanup distributed training state"""
    if torch.distributed.is_initialized():
        print("\n✓ Destroying distributed process group from training...")
        torch.distributed.destroy_process_group()
        print("✓ Process group destroyed")

def generate_visualizations(
    model: LightningModule,
    data_module: L.LightningDataModule,
    cfg: DictConfig,
    logger,
    checkpoint_dir: str
):
    """
    Generate confusion matrix and GradCAM visualizations
    
    Args:
        model: Lightning model
        data_module: Data module
        cfg: Configuration object
        logger: Logger instance
        checkpoint_dir: Directory for saving visualizations
    """
    # Generate confusion matrix
    if hasattr(model, 'all_test_preds') and hasattr(model, 'all_test_targets'):
        cm_path = os.path.join(checkpoint_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            model.all_test_preds,
            model.all_test_targets,
            model.class_names,
            logger=logger if isinstance(logger, WandbLogger) else None,
            save_path=cm_path
        )
        print(f"\n✓ Confusion matrix saved to {cm_path}")
    
    # Generate GradCAM visualizations
    if cfg.xai.enable_gradcam and isinstance(logger, WandbLogger):
        print(f"\n{'='*70}")
        print("Generating GradCAM Visualizations...")
        print(f"{'='*70}\n")
        
        generate_gradcam_visualizations(
            model=model,
            data_module=data_module,
            logger=logger,
            class_names=model.class_names,
            num_correct=cfg.xai.num_correct_samples,
            num_incorrect=cfg.xai.num_incorrect_samples,
        )
        print("✓ GradCAM visualizations complete")


def run_test_evaluation(
    model: LightningModule,
    data_module: L.LightningDataModule,
    cfg: DictConfig,
    logger
) -> dict:
    """
    Run test evaluation with single GPU
    
    Args:
        model: Lightning model
        data_module: Data module
        cfg: Configuration object
        logger: Logger instance
        
    Returns:
        Dictionary of test metrics
    """
    print(f"\n{'='*70}")
    print("Creating single-GPU trainer for testing...")
    print(f"{'='*70}\n")
    
    test_trainer = L.Trainer(
        accelerator=cfg.training.trainer.accelerator,
        devices=1,  # Single GPU to avoid DDP issues
        logger=logger,
        enable_progress_bar=True,
    )
    
    print(f"\n{'='*70}")
    print("Running Test Evaluation (Single GPU)...")
    print(f"{'='*70}\n")
    
    test_results = test_trainer.test(model, ckpt_path=None, datamodule=data_module)
    return test_results[0] if test_results else {}

def log_final_metrics(logger, test_result, complexity_metrics, latency_metrics):
    extra_metrics = {
        "model/num_parameters": complexity_metrics["num_params"],
        "model/flops": complexity_metrics["flops"],
        **{f"model/{k}": v for k, v in latency_metrics.items()}
    }
    
    if isinstance(logger, WandbLogger):
        logger.log_metrics(extra_metrics)
        logger.experiment.finish()
        print("\n✓ Results logged to WandB")
    elif isinstance(logger, CSVLogger):
        logger.log_metrics({f"final_{k}": v for k, v in extra_metrics.items()})
        print("\n✓ Results logged to CSV")
