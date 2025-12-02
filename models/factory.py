"""
Model factory for creating different architectures
"""
import torch.nn as nn
from typing import Dict, Any
from .resnet import ResNet
from .pretrained import (
    create_pretrained_resnet, 
    create_pretrained_resnet34, 
    create_pretrained_resnet50
)

def create_model(model_name: str, model_hparams: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create models based on configuration
    
    Args:
        model_name: Name/type of the model to create
        model_hparams: Dictionary of model hyperparameters
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_name is not recognized
    """
    model_registry = {
        "resnet_scratch": ResNet,
        "resnet18_pretrained": create_pretrained_resnet,
        "resnet34_pretrained": create_pretrained_resnet34,
        "resnet50_pretrained": create_pretrained_resnet50,
    }
    
    if model_name not in model_registry:
        available = ", ".join(model_registry.keys())
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {available}"
        )
    
    model_fn = model_registry[model_name]
    return model_fn(**model_hparams)
