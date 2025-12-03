"""
Main training script for SFDD Driver Detection
"""
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from data.datamodule import SFDDDataModule
from modules.lightning_module import ResNetModule
from utils.metrics import compute_model_complexity, measure_inference_latency
from utils.visualization import plot_confusion_matrix
from utils.gradcam import generate_gradcam_visualizations


def setup_logger(cfg: DictConfig, experiment_name: str):
    """Setup appropriate logger based on config"""
    if cfg.logging.enable_wandb:
        try:
            import wandb
            try:
                from kaggle_secrets import UserSecretsClient
                user_secrets = UserSecretsClient()
                wandb_key = user_secrets.get_secret("wandb_api")
                wandb.login(key=wandb_key)
            except:
                print("⚠ WandB key not found, using offline mode")
            
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

    experiment_name = f"{cfg.experiment.id}_{cfg.model.name}_{cfg.data.task_type}_{cfg.defaults.optimizer}"

    print(f"Starting Experiment: {experiment_name}\n")
    logger = setup_logger(cfg, experiment_name)

    # setup datamodule
    data_module = SFDDDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_split=cfg.data.val_split,
        img_size=cfg.data.img_size,
        seed=cfg.data.seed if hasattr(cfg.data, 'seed') else cfg.experiment.seed,
        task_type=cfg.data.task_type,
        binary_mapping=cfg.data.binary_mapping if hasattr(cfg.data, 'binary_mapping') else 'c0_vs_rest',
    )

    # setup data to get actual num_classes and class_names
    data_module.setup(stage='fit')
    num_classes = data_module.get_num_classes
    class_names = data_module.get_class_names


    print(f"\n✓ Task Type: {cfg.data.task_type}")
    print(f"✓ Number of Classes: {num_classes}")
    print(f"✓ Class Names: {class_names}\n")

    # Prepare model hyperparameters (override num_classes from data)
    model_hparams = OmegaConf.to_container(cfg.model, resolve=True)
    model_hparams['num_classes'] = num_classes

    # Prepare optimizer hyperparameters
    optimizer_hparams = {
        "lr": cfg.optimizer.lr,
        "weight_decay": cfg.optimizer.weight_decay,
    }
    
    # Add momentum for SGD
    if cfg.optimizer.name.lower() == "sgd" and hasattr(cfg.optimizer, 'momentum'):
        optimizer_hparams["momentum"] = cfg.optimizer.momentum
    
    # Initialize model
    model = ResNetModule(
        model_name=cfg.model.name,
        model_hparams=model_hparams,
        optimizer_name=cfg.optimizer.name,
        optimizer_hparams=optimizer_hparams,
        class_names=class_names,  # Use actual from data
    )
    
    # Setup checkpoint directory
    checkpoint_dir = os.path.join(cfg.logging.checkpoint_dir, experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
            patience=cfg.training.early_stopping_patience,
            verbose=True
        ),
    ]
    
    # Setup trainer
    trainer_kwargs = {
        "default_root_dir": checkpoint_dir,
        "accelerator": cfg.training.accelerator,
        "devices": cfg.training.devices,
        "max_epochs": cfg.training.max_epochs,
        "logger": logger,
        "callbacks": callbacks,
        "log_every_n_steps": cfg.training.log_every_n_steps,
    }
    
    # Add gradient clipping if specified
    if cfg.training.gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = cfg.training.gradient_clip_val
    
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
    best_model = ResNetModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        model_name=cfg.model.name,
        model_hparams=model_hparams,
        optimizer_name=cfg.optimizer.name,
        optimizer_hparams=optimizer_hparams,
        class_names=class_names,  # Use actual from data
    )
    best_model.eval()
    
    # Compute model complexity
    print(f"\n{'='*70}")
    print("Evaluating Model Complexity...")
    print(f"{'='*70}\n")
    complexity_metrics = compute_model_complexity(
        best_model.model,
        input_size=(3, cfg.data.img_size, cfg.data.img_size)
    )
    
    # Measure inference latency
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    latency_metrics = measure_inference_latency(
        best_model,
        test_loader,
        num_batches=50
    )
    
    # Test evaluation
    print(f"\n{'='*70}")
    print("Running Test Evaluation...")
    print(f"{'='*70}\n")
    test_results = trainer.test(best_model, ckpt_path=None, datamodule=data_module)
    test_result = test_results[0] if test_results else {}
    
    # Generate confusion matrix
    if hasattr(best_model, 'all_test_preds') and hasattr(best_model, 'all_test_targets'):
        cm_path = os.path.join(checkpoint_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            best_model.all_test_preds,
            best_model.all_test_targets,
            class_names,  # Use actual from data
            logger=logger if isinstance(logger, WandbLogger) else None,
            save_path=cm_path
        )
    
    # Generate GradCAM visualizations
    if cfg.xai.enable_gradcam and isinstance(logger, WandbLogger):
        generate_gradcam_visualizations(
            model=best_model,
            data_module=data_module,
            logger=logger,
            class_names=class_names,  # Use actual from data
            num_correct=cfg.xai.num_correct_samples,
            num_incorrect=cfg.xai.num_incorrect_samples,
            inspect_architecture=cfg.xai.inspect_architecture
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