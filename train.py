"""
Main training script for SFDD Driver Detection
"""
import os
import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from configs.base_config import Config
from configs.experiment_configs import get_experiment_config, get_all_experiments
from data.datamodule import SFDDDataModule
from modules.lightning_module import ResNetModule
from utils.metrics import compute_model_complexity, measure_inference_latency
from utils.visualization import plot_confusion_matrix
from utils.gradcam import generate_gradcam_visualizations


def setup_logger(config: Config, experiment_name: str):
    """Setup appropriate logger based on config"""
    if config.logging.enable_wandb:
        try:
            # Try to login to wandb if available
            import wandb
            from kaggle_secrets import UserSecretsClient
            try:
                user_secrets = UserSecretsClient()
                wandb_key = user_secrets.get_secret("wandb_api")
                wandb.login(key=wandb_key)
            except:
                print("⚠ WandB key not found in secrets, using offline mode")
            
            logger = WandbLogger(
                offline=config.logging.wandb_offline,
                name=experiment_name,
                project=config.logging.wandb_project,
                config=config.to_dict(),
            )
            print("✓ WandB logger initialized successfully")
            return logger
        except Exception as e:
            print(f"⚠ WandB Logger failed: {e}. Falling back to CSV logger.")
    
    logger = CSVLogger(save_dir="logs", name=experiment_name)
    print("✓ CSV logger initialized")
    return logger


def run_experiment(config: Config, exp_id: str):
    """
    Run a complete training and evaluation experiment
    
    Args:
        config: Configuration object
        exp_id: Experiment identifier
    """
    # Set seed
    L.seed_everything(config.training.seed)
    
    # Create experiment name
    experiment_name = (
        f"{exp_id}-{config.model.model_name}_"
        f"{config.optimizer.name}_seed{config.training.seed}"
    )
    
    print(f"\n{'='*70}")
    print(f"Starting Experiment: {experiment_name}")
    print(f"{'='*70}\n")
    
    # Setup logger
    logger = setup_logger(config, experiment_name)
    
    # Setup data module
    data_module = SFDDDataModule(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        val_split=config.data.val_split,
        img_size=config.data.img_size,
        seed=config.data.seed
    )
    
    # Prepare model hyperparameters
    model_hparams = {
        "num_classes": config.model.num_classes,
    }
    if config.model.model_name == "resnet_scratch":
        model_hparams.update({
            "c_hidden": config.model.c_hidden,
            "num_blocks": config.model.num_blocks,
            "act_fn_name": config.model.act_fn_name,
        })
    
    # Prepare optimizer hyperparameters
    optimizer_hparams = {
        "lr": config.optimizer.lr,
        "weight_decay": config.optimizer.weight_decay,
    }
    
    # Initialize model
    model = ResNetModule(
        model_name=config.model.model_name,
        model_hparams=model_hparams,
        optimizer_name=config.optimizer.name,
        optimizer_hparams=optimizer_hparams,
        class_names=config.class_names,
    )
    
    # Setup checkpoint directory
    checkpoint_dir = os.path.join(config.logging.checkpoint_dir, experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup trainer
    trainer = L.Trainer(
        default_root_dir=checkpoint_dir,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        max_epochs=config.training.max_epochs,
        logger=logger,
        callbacks=[
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
                patience=config.training.early_stopping_patience,
                verbose=True
            ),
        ],
        log_every_n_steps=config.training.log_every_n_steps,
    )
    
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
        model_name=config.model.model_name,
        model_hparams=model_hparams,
        optimizer_name=config.optimizer.name,
        optimizer_hparams=optimizer_hparams,
        class_names=config.class_names,
    )
    best_model.eval()
    
    # Compute model complexity
    print(f"\n{'='*70}")
    print("Evaluating Model Complexity...")
    print(f"{'='*70}\n")
    complexity_metrics = compute_model_complexity(
        best_model.model, 
        input_size=(3, config.data.img_size, config.data.img_size)
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
            config.class_names,
            logger=logger if isinstance(logger, WandbLogger) else None,
            save_path=cm_path
        )
    
    # Generate GradCAM visualizations
    if config.xai.enable_xai and isinstance(logger, WandbLogger):
        generate_gradcam_visualizations(
            model=best_model,
            data_module=data_module,
            logger=logger,
            class_names=config.class_names,
            num_samples=config.xai.num_gradcam_samples
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
    
    return {
        "test_metrics": test_result,
        "model_metrics": {**complexity_metrics, **latency_metrics}
    }


def main():
    parser = argparse.ArgumentParser(description="Train SFDD Driver Detection Model")
    parser.add_argument(
        "--exp_id", 
        type=str, 
        default="exp001_resnet18_pretrained",
        help="Experiment ID (e.g., exp001_resnet18_pretrained)"
    )
    parser.add_argument(
        "--run_all", 
        action="store_true",
        help="Run all experiments sequentially"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/kaggle/input/state-farm-distracted-driver-detection/imgs/train",
        help="Path to training data"
    )
    parser.add_argument(
        "--enable_wandb",
        action="store_true",
        help="Enable WandB logging"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=30,
        help="Maximum number of epochs"
    )
    
    args = parser.parse_args()
    
    if args.run_all:
        # Run all experiments
        exp_ids = get_all_experiments()
        print(f"\n{'='*70}")
        print(f"Running {len(exp_ids)} experiments")
        print(f"{'='*70}\n")
        
        results = {}
        for exp_id in exp_ids:
            config = get_experiment_config(exp_id)
            config.data.data_dir = args.data_dir
            config.logging.enable_wandb = args.enable_wandb
            config.training.max_epochs = args.max_epochs
            
            try:
                result = run_experiment(config, exp_id)
                results[exp_id] = result
            except Exception as e:
                print(f"\n✗ Experiment {exp_id} failed: {e}\n")
                continue
        
        # Print summary of all experiments
        print(f"\n{'='*70}")
        print("ALL EXPERIMENTS SUMMARY")
        print(f"{'='*70}")
        for exp_id, result in results.items():
            test_acc = result['test_metrics'].get('test_acc', 0.0)
            print(f"{exp_id}: Test Acc = {test_acc:.4f}")
        print(f"{'='*70}\n")
    else:
        # Run single experiment
        config = get_experiment_config(args.exp_id)
        config.data.data_dir = args.data_dir
        config.logging.enable_wandb = args.enable_wandb
        config.training.max_epochs = args.max_epochs
        
        run_experiment(config, args.exp_id)


if __name__ == "__main__":
    main()
