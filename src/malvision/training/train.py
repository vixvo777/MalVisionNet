"""
Training loop for MalVisionNet models.
Supports grayscale and RGB byteplot models with configurable backbones.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
import yaml
from pathlib import Path
import time
import psutil
import numpy as np

from ..logger import logger
from ..models import SimpleResNet18
from ..representations import GrayscaleConverter, RGBTrigramConverter


class ByteplotDataset(Dataset):
    """Dataset for loading byteplots from manifest files."""

    def __init__(self, manifest_path, representation="grayscale",
                 split="train", transform=None):
        self.manifest = pd.read_csv(manifest_path)
        self.split = split
        self.filter_split()

        if representation == "grayscale":
            self.converter = GrayscaleConverter()
        elif representation == "rgb":
            self.converter = RGBTrigramConverter()
        else:
            raise ValueError(f"Unknown representation: {representation}")

        self.transform = transform

    def filter_split(self):
        self.manifest = self.manifest[self.manifest['split'] == self.split]

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        # Load raw bytes (placeholder - needs actual implementation)
        # For now, use synthetic data for testing
        file_size = row['file_size']
        raw_bytes = bytes(np.random.randint(0, 256, size=file_size))

        # Convert to byteplot
        if self.split == "rgb":
            image, metadata = self.converter.convert(raw_bytes)
        else:
            image, metadata = self.converter.convert(raw_bytes)

        # Convert to tensor
        if len(image.shape) == 3:  # RGB
            image = image.permute(2, 0, 1)  # HWC to CHW

        # Create label (placeholder - needs actual implementation)
        label = torch.tensor([row['label'] == 'malicious'], dtype=torch.float32)

        return image, label


def train_model(config_path="configs/model.yaml"):
    """Main training function."""
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize model
    model = SimpleResNet18(num_classes=1)

    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    # Training loop
    epochs = config["epochs"]
    batch_size = config["batch_size"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Create datasets and dataloaders
        train_dataset = ByteplotDataset(
            config["train_manifest_path"],
            representation=config["representation"],
            split="train"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({
                "Loss": f"{total_loss/(total+1):.4f}",
                "Acc": f"{correct/total:.4f}"
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

        # Save checkpoint
        checkpoint_path = Path(config["checkpoint_dir"])
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'accuracy': accuracy,
        }, checkpoint_path / f"model_epoch_{epoch+1}.pth")

    logger.info("Training completed!")

    # Save final model
    final_path = Path(config["final_model_path"])
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path)

    return model


if __name__ == "__main__":
    train_model()