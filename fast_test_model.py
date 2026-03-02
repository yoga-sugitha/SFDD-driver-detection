import torch
from models.core_resnet import ConvResNet
# from models.resnet import ResNet
# from models.pretrained import create_pretrained_resnet


model = ConvResNet(num_classes=4)
x = torch.randn(2,3,299,299)
# print(hasattr(model.backbone, "Mixed_7c"))
# Training mode
model.train()
out = model(x)
print(f"Train output shape: {out.shape}")  # Should be (2, 4)

# Eval mode
model.eval()
out = model(x)
print(f"Eval output shape: {out.shape}")  # Should be (2, 4)

print("✅ Model works correctly!")