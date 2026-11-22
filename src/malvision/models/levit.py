import torch
import torch.nn as nn
import timm
from ..logger import logger

class LeViTModel(nn.Module):
    """
    LeViT model for efficient byteplot classification.
    """
    def __init__(self, num_classes=1, model_name='levit_128s', pretrained=False):
        super().__init__()

        logger.info(f"Loading {model_name}, pretrained={pretrained}")
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = LeViTModel(model_name='levit_128s')
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"LeViT output shape: {output.shape}")
