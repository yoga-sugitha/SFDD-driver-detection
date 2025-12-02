"""
Base configuration for SFDD Driver Detection
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class DataConfig:
    """Data-related configuration"""
    data_dir: str = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
    batch_size: int = 32
    num_workers: int = 0  # Set to 0 for Kaggle stability
    val_split: float = 0.2
    img_size: int = 224
    seed: int = 42

@dataclass
class TrainingConfig:
    """Training-related configuration"""
    max_epochs: int = 30
    seed: int = 42
    accelerator: str = "auto"
    devices: str = "auto"
    log_every_n_steps: int = 10
    early_stopping_patience: int = 15
    
@dataclass
class LoggingConfig:
    """Logging configuration"""
    enable_wandb: bool = False  # Set to False for Kaggle by default
    wandb_project: str = "ACRIG-SFDD-Driver_Detection"
    wandb_offline: bool = True
    checkpoint_dir: str = "checkpoints"
    
@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "resnet18_pretrained"
    num_classes: int = 10
    # ResNet scratch parameters
    c_hidden: List[int] = field(default_factory=lambda: [16, 32, 64])
    num_blocks: List[int] = field(default_factory=lambda: [3, 3, 3])
    act_fn_name: str = "relu"
    
@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    name: str = "Adam"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    
@dataclass
class XAIConfig:
    """Explainable AI configuration"""
    enable_xai: bool = False
    num_gradcam_samples: int = 8

@dataclass
class Config:
    """Main configuration object"""
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
    
    # Class names
    class_names: List[str] = field(default_factory=lambda: [
        "normal driver", "texting-right", "taking on the phone-right",
        "texting-left", "talking on the phone-left", "operating the radio",
        "drinking", "reaching behind", "hair and makeup", "talking to passanger"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        return {
            "data": vars(self.data),
            "training": vars(self.training),
            "logging": vars(self.logging),
            "model": vars(self.model),
            "optimizer": vars(self.optimizer),
            "xai": vars(self.xai),
            "class_names": self.class_names
        }
