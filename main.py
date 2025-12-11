"""
Main training script with multi-GPU training but single-GPU testing
This avoids DDP index misalignment issues for GradCAM
+ Kaggle wandb online logging is removed,instead this uses offline-mode wandb logger that is provided by pytorch lightning
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar
)
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from data.datamodule import SFDDDataModule
from data.howdrive_datamodule import HowDriveDataModule
from modules.lightning_module import ResNetModule
from utils.metrics import compute_model_complexity, measure_inference_latency
from utils.visualization import plot_confusion_matrix
from utils.gradcam import generate_gradcam_visualizations

def setup_logger(cfg: DictConfig, experiment_name: str):
    """Setup appropriate logger based on config"""
    if cfg.logging.enable_wandb:
        try:
            logger = WandbLogger(
                offline=cfg.logging.wandb_offline,
                name=experiment_name,
                project=cfg.logging.wandb.wandb_project,
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=cfg.experiment.tags if cfg.experiment.tags else None,
            )
            print("✓ WandB logger initialized")
            return logger
        except Exception as e:
            print(f"⚠ WandB Logger failed: {e}. Falling back to CSV logger.")

    logger = CSVLogger(save_dir='logs', name=experiment_name)
    print("✓ CSV logger initialized")
    return logger

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    """
    Main training function

    Args:
        cfg: Hydra configuration object
    """
    # Print configuration
    print("\n" + "="*70)
    print("Configuration")
    print("="*70)
    print(OmegaConf.to_yaml(cfg))
    print("="*70 + "\n")

    # set seed
    L.seed_everything(cfg.experiment.seed)

    experiment_name = f"{cfg.model.name}_{cfg.experiment.dataset_name}_{cfg.data.hparams.task_type}_{cfg.optimizer.name}_{cfg.experiment.seed}_{cfg.training.trainer.max_epochs}_{cfg.experiment.id}"
    print(f"Starting Experiment: {experiment_name}\n")
    logger = setup_logger(cfg, experiment_name)


    # Setup Lightning Data Module
    DATASET_REGISTRY = {
        "howdrive-sample": HowDriveDataModule,
        "howdrive": HowDriveDataModule,
        "sfdd": SFDDDataModule,
    }
    data_cls = DATASET_REGISTRY.get(cfg.data.name)
    if data_cls is None:
        raise ValueError(f"Unknown dataset name: {cfg.data.name}")
    
    data_hparams = OmegaConf.to_container(cfg.data.hparams, resolve=True)
    data_module = data_cls(**data_hparams)
    
    # setup data to get actual num_classes and class_names
    data_module.setup(stage='fit')
    num_classes = data_module.num_classes
    class_names = data_module.class_names

    print(f"\n✓ Task Type: {cfg.data.hparams.task_type}")
    print(f"✓ Number of Classes: {num_classes}")
    print(f"✓ Class Names: {class_names}\n")

    # Prepare model hyperparameters (override num_classes from data)
    model_hparams = OmegaConf.to_container(cfg.model.hparams, resolve=True)
    model_hparams['num_classes'] = num_classes

    # Prepare optimizer hyperparameters
    optimizer_hparams = OmegaConf.to_container(cfg.optimizer.hparams, resolve=True)
    
    # Initialize model
    model = ResNetModule(
        model_name=cfg.model.name,
        model_hparams=model_hparams,
        optimizer_name=cfg.optimizer.name,
        optimizer_hparams=optimizer_hparams,
        class_names=class_names,
    )
    
    # Setup checkpoint directory
    checkpoint_dir = os.path.join(cfg.logging.checkpoint_dir, experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup training kwargs
    trainer_kwargs = OmegaConf.to_container(cfg.training.trainer, resolve=True)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            save_weights_only=True,
            mode="max",
            monitor="val_acc",
            dirpath=checkpoint_dir,
            filename="best-checkpoint",
            save_top_k=1,
        ),
        LearningRateMonitor("epoch"),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=cfg.training.callbacks.patience,
            verbose=True
        ),
    ]
    
    # Setup trainer
    trainer_kwargs.update({
        "default_root_dir": checkpoint_dir,
        "logger": logger,
        "callbacks": callbacks,
        
    })
    
    trainer = L.Trainer(**trainer_kwargs)

    
    # Train
    print(f"\n{'='*70}")
    print("Starting Training...")
    print(f"{'='*70}\n")
    trainer.fit(model, datamodule=data_module)
    
    # Load best checkpoint
    print("\n" + "="*70)
    print("Loading best checkpoint for evaluation...")
    print("="*70)
    
    # Load checkpoint to CPU first to avoid device conflicts
    best_model = ResNetModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        model_name=cfg.model.name,
        model_hparams=model_hparams,
        optimizer_name=cfg.optimizer.name,
        optimizer_hparams=optimizer_hparams,
        class_names=class_names,
        map_location='cpu'  # Load to CPU first
    )
    best_model.eval()
    
    # ============================================================
    # CRITICAL: Destroy distributed process group from training
    # ============================================================
    if torch.distributed.is_initialized():
        print("\n✓ Destroying distributed process group from training...")
        torch.distributed.destroy_process_group()
        print("✓ Process group destroyed")
    
    # ============================================================
    # TESTING TRAINER: Use SINGLE GPU to avoid DDP issues
    # ============================================================
    print(f"\n{'='*70}")
    print("Creating single-GPU trainer for testing...")
    print(f"{'='*70}\n")
    
    test_trainer = L.Trainer(
        accelerator=cfg.training.trainer.accelerator,
        devices=1, 
        logger=logger,
    )
    
    # Compute model complexity
    print(f"\n{'='*70}")
    print("Evaluating Model Complexity...")
    print(f"{'='*70}\n")
    
    # Move model to GPU for complexity computation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    complexity_metrics = compute_model_complexity(
        best_model.model.to(device),
        input_size=(3, cfg.data.hparams.img_size, cfg.data.hparams.img_size)
    )
    
    # Measure inference latency
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    latency_metrics = measure_inference_latency(
        best_model,
        test_loader,
        num_batches=50
    )
    
    # Test evaluation with single GPU
    print(f"\n{'='*70}")
    print("Running Test Evaluation (Single GPU)...")
    print(f"{'='*70}\n")
    test_results = test_trainer.test(best_model, ckpt_path=None, datamodule=data_module)
    test_result = test_results[0] if test_results else {}
    
    # Generate confusion matrix
    if hasattr(best_model, 'all_test_preds') and hasattr(best_model, 'all_test_targets'):
        cm_path = os.path.join(checkpoint_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            best_model.all_test_preds,
            best_model.all_test_targets,
            class_names,
            logger=logger if isinstance(logger, WandbLogger) else None,
            save_path=cm_path
        )
    
    # Generate GradCAM visualizations
    # Now the indices will be correctly aligned!
    if cfg.xai.enable_gradcam and isinstance(logger, WandbLogger):
        generate_gradcam_visualizations(
            model=best_model,
            data_module=data_module,
            logger=logger,
            class_names=class_names,
            num_correct=cfg.xai.num_correct_samples,
            num_incorrect=cfg.xai.num_incorrect_samples,
        )
    
    # Compile and log all metrics
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
    
    # Print summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Experiment:        {experiment_name}")
    print(f"Model:             {cfg.model.name}")
    print(f"Optimizer:         {cfg.optimizer.name}")
    print(f"Test Accuracy:     {test_result.get('test_acc', 0.0):.4f}")
    print(f"Test Precision:    {test_result.get('test_precision', 0.0):.4f}")
    print(f"Test Recall:       {test_result.get('test_recall', 0.0):.4f}")
    print(f"Test F1 (Macro):   {test_result.get('test_f1_macro', 0.0):.4f}")
    print(f"Test F1 (Weighted):{test_result.get('test_f1_weighted', 0.0):.4f}")
    print(f"Model Parameters:  {complexity_metrics['num_params']:,}")
    if complexity_metrics['flops'] > 0:
        print(f"FLOPs:             {complexity_metrics['flops']:,}")
    print(f"Avg Latency:       {latency_metrics['avg_inference_latency_ms']:.2f} ms")
    print(f"{'='*70}\n")
    
    return test_result.get('test_acc', 0.0)


if __name__ == "__main__":
    train()