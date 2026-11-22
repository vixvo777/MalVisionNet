import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from ..logger import logger

class SimpleResNet18(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super().__init__()

        # Use timm's ResNet18 implementation
        if pretrained:
            logger.info("Loading pretrained ResNet18")
            self.backbone = timm.create_model('resnet18', pretrained=True)
        else:
            logger.info("Loading non-pretrained ResNet18")
            self.backbone = timm.create_model('resnet18', pretrained=False)

        # Modify the final layer for binary classification
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    # Quick test
    model = SimpleResNet18(num_classes=1)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.flatten()}")