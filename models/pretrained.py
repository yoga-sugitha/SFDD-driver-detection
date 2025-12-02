"""
Pretrained model loaders
"""
import torch.nn as nn
import torchvision.models as models

def create_pretrained_resnet(num_classes: int = 10, **kwargs):
    """
    Create pretrained ResNet18 with custom classifier
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Modified ResNet18 model
    """
    model = models.resnet18(weights='DEFAULT')
    
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


def create_pretrained_resnet34(num_classes: int = 10, **kwargs):
    """Create pretrained ResNet34 with custom classifier"""
    model = models.resnet34(weights='DEFAULT')
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model


def create_pretrained_resnet50(num_classes: int = 10, **kwargs):
    """Create pretrained ResNet50 with custom classifier"""
    model = models.resnet50(weights='DEFAULT')
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model
