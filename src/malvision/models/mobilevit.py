import torch
import torch.nn as nn
import timm
from ..logger import logger

class MobileViTModel(nn.Module):
    """
    MobileViT model for byteplot classification.
    """
    def __init__(self, num_classes=1, model_name='mobilevit_xxs', pretrained=False):
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
    # Test pass
    model = MobileViTModel(model_name='mobilevit_xxs')
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"MobileViT output shape: {output.shape}")
