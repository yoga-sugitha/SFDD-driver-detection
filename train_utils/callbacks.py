# train_utils/setup_callbacks.py
from omegaconf import DictConfig
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar
)

def setup_callbacks(cfg: DictConfig, checkpoint_dir: str) -> list:
    """
    Setup training callbacks
    
    Args:
        cfg: Configuration object
        checkpoint_dir: Directory for saving checkpoints
        
    Returns:
        List of callbacks
    """
    callbacks = [
        ModelCheckpoint(
            save_weights_only=True,
            mode="max",
            monitor="val_acc",
            dirpath=checkpoint_dir,
            filename="best-checkpoint",
            save_top_k=1,
            verbose=True,
        ),
        LearningRateMonitor("epoch"),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=cfg.training.callbacks.patience,
            verbose=True
        ),
    ]
    
    return callbacks
