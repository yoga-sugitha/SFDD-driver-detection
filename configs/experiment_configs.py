"""
Experiment configurations for different model architectures and settings
"""
from .base_config import Config, ModelConfig, OptimizerConfig, XAIConfig

def get_experiment_config(exp_id: str) -> Config:
    """
    Get configuration for a specific experiment
    
    Args:
        exp_id: Experiment identifier
        
    Returns:
        Config object for the experiment
    """
    configs = {
        "exp001_resnet18_pretrained": Config(
            model=ModelConfig(
                model_name="resnet18_pretrained",
                num_classes=10
            ),
            optimizer=OptimizerConfig(
                name="Adam",
                lr=1e-3,
                weight_decay=1e-4
            ),
            xai=XAIConfig(enable_xai=True)
        ),
        
        "exp002_resnet_scratch": Config(
            model=ModelConfig(
                model_name="resnet_scratch",
                num_classes=10,
                c_hidden=[16, 32, 64],
                num_blocks=[3, 3, 3],
                act_fn_name="relu"
            ),
            optimizer=OptimizerConfig(
                name="Adam",
                lr=1e-3,
                weight_decay=1e-4
            ),
            xai=XAIConfig(enable_xai=True)
        ),
        
        "exp003_resnet_scratch_deep": Config(
            model=ModelConfig(
                model_name="resnet_scratch",
                num_classes=10,
                c_hidden=[32, 64, 128],
                num_blocks=[4, 4, 4],
                act_fn_name="relu"
            ),
            optimizer=OptimizerConfig(
                name="AdamW",
                lr=5e-4,
                weight_decay=1e-4
            ),
            xai=XAIConfig(enable_xai=True)
        ),
        
        "exp004_sgd_momentum": Config(
            model=ModelConfig(
                model_name="resnet18_pretrained",
                num_classes=10
            ),
            optimizer=OptimizerConfig(
                name="SGD",
                lr=1e-2,
                weight_decay=1e-4
            ),
            xai=XAIConfig(enable_xai=False)
        ),
    }
    
    if exp_id not in configs:
        raise ValueError(f"Unknown experiment ID: {exp_id}. Available: {list(configs.keys())}")
    
    return configs[exp_id]

def get_all_experiments():
    """Get all available experiment configurations"""
    return [
        "exp001_resnet18_pretrained",
        "exp002_resnet_scratch",
        "exp003_resnet_scratch_deep",
        "exp004_sgd_momentum"
    ]
