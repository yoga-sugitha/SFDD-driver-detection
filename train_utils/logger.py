# train_utils/logger.py
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger, CSVLogger
import os

def setup_logger(cfg: DictConfig, experiment_name: str):
    if cfg.logging.enable_wandb:
        try:
            # Resolve config to plain Python dict for WandB
            wandb_config = OmegaConf.to_container(cfg, resolve=True)
            logger = WandbLogger(
                offline=cfg.logging.wandb_offline,
                name=experiment_name,
                project=cfg.logging.wandb.wandb_project,
                config=wandb_config,
                tags=cfg.experiment.tags or None,
            )
            print("✓ WandB logger initialized")
            return logger
        except Exception as e:
            print(f"⚠ WandB failed: {e}. Falling back to CSV.")
    
    logger = CSVLogger(save_dir='logs', name=experiment_name)
    print("✓ CSV logger initialized")
    return logger