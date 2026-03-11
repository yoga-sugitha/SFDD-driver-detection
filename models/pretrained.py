import torch.nn as nn
import torchvision.models as models

def create_pretrained_resnet(
    num_classes: int = 10,
    pretrained: bool = True,
    dropout: float = 0.3,
    **kwargs
):
    """
    Create ResNet18 with optional pretrained weights and configurable classifier.

    Args:
        num_classes: Number of output classes
        pretrained: If True, load ImageNet pretrained weights
        dropout: Dropout probability for the classifier head

    Returns:
        Modified ResNet18 model
    """
    weights = 'DEFAULT' if pretrained else None
    model = models.resnet18(weights=weights)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout + 0.2),        # first dropout slightly higher
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(dropout),              # second dropout from config
        nn.Linear(512, num_classes)
    )

    return model