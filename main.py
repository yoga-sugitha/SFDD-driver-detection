# main.py
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from pathlib import Path

from data.datamodule import SFDDDataModule
from data.howdrive_datamodule import HowDriveDataModule
from modules.lightning_module import LightningModule
from train_utils.logger import setup_logger
from train_utils.callbacks import setup_callbacks
from train_utils.evaluation import (
    evaluate_inference_latency, 
    evaluate_model_complexity, 
    cleanup_distributed,
    run_test_evaluation, 
    generate_visualizations, 
    log_final_metrics
)
from train_utils.summary import print_summary
from train_utils.model_loading import load_best_model

DATASET_REGISTRY = {
    "howdrive-sample": HowDriveDataModule,
    "howdrive": HowDriveDataModule,
    "sfdd": SFDDDataModule,
}

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    print("\n" + "="*70)
    print("Configuration")
    print("="*70)
    print(OmegaConf.to_yaml(cfg))
    print("="*70 + "\n")

    # NOTE: Seed and experiment name
    L.seed_everything(cfg.experiment.seed, workers=True)
    experiment_name = (
        f"{cfg.model.name}_dataset--{cfg.data.name}_"
        f"task--{cfg.data.hparams.task_type}_{cfg.optimizer.name}_"
        f"seed--{cfg.experiment.seed}_epoch--{cfg.training.trainer.max_epochs}_"
        f"{cfg.experiment.id}"
    )
    print(f"Starting Experiment: {experiment_name}\n")


    # NOTE: Setup logger and checkpoint path
    logger = setup_logger(cfg, experiment_name)
    checkpoint_dir = Path(cfg.logging.checkpoint_dir) / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: Lightning DataModule
    data_cls = DATASET_REGISTRY.get(cfg.data.name)    
    data_module = data_cls(**OmegaConf.to_container(cfg.data.hparams, resolve=True))
    data_module.setup(stage='fit')


    # NOTE: Lightning Module
    model_hparams = OmegaConf.to_container(cfg.model.hparams, resolve=True)
    model_hparams['num_classes'] = data_module.num_classes
    model = LightningModule(
        model_name=cfg.model.name,
        model_hparams=model_hparams,
        optimizer_name=cfg.optimizer.name,
        optimizer_hparams=OmegaConf.to_container(cfg.optimizer.hparams, resolve=True),
        scheduler_name=cfg.scheduler.name,
        scheduler_hparams=OmegaConf.to_container(cfg.scheduler.hparams, resolve=True),
        class_names=data_module.class_names,
    )
    
    # NOTE: Setup training 
    callbacks = setup_callbacks(cfg, str(checkpoint_dir))
    trainer = L.Trainer(
        **OmegaConf.to_container(cfg.training.trainer, resolve=True),
        default_root_dir=str(checkpoint_dir),
        logger=logger,
        callbacks=callbacks
    )
    trainer.fit(model, datamodule=data_module)
    
    # NOTE: Load best model
    best_model = load_best_model(trainer, cfg, data_module.class_names, fallback_model=model)
    best_model.eval()
    
    # NOTE: cleanup distributed
    cleanup_distributed()
    
    # NOTE: Evaluate
    complexity_metrics = evaluate_model_complexity(best_model, cfg)
    latency_metrics = evaluate_inference_latency(best_model, data_module)
    test_result = run_test_evaluation(best_model, data_module, cfg, logger)

    # NOTE: Call Generate Visualization Function
    generate_visualizations(best_model, data_module, cfg, logger, str(checkpoint_dir))
    
    # NOTE: Log Final Metrics
    log_final_metrics(logger, test_result, complexity_metrics, latency_metrics)
    
    # NOTE: Print summary
    print_summary(experiment_name, cfg, test_result, complexity_metrics, latency_metrics)
    
    return test_result.get('test_acc', 0.0)

if __name__ == "__main__":
    train()