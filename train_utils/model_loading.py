# train_utils/model_loading.py
import os
from omegaconf import DictConfig, OmegaConf
from modules.lightning_module import LightningModule

def load_best_model(trainer, cfg: DictConfig, class_names: list, fallback_model=None):
    ckpt_path = trainer.checkpoint_callback.best_model_path
    if not ckpt_path or not os.path.exists(ckpt_path):
        if fallback_model is None:
            raise ValueError("No checkpoint and no fallback model provided.")
        print("⚠ No checkpoint found. Using current model weights.")
        return fallback_model

    print(f"✓ Loading checkpoint: {ckpt_path}")
    return LightningModule.load_from_checkpoint(
        ckpt_path,
        model_name=cfg.model.name,
        model_hparams=OmegaConf.to_container(cfg.model.hparams, resolve=True),
        optimizer_name=cfg.optimizer.name,
        optimizer_hparams=OmegaConf.to_container(cfg.optimizer.hparams, resolve=True),
        class_names=class_names,
        map_location='cpu'
    )