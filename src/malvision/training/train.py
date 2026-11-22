"""
End-to-end training pipeline for MalVisionNet.
Supports multiple backbones and representations.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
import pandas as pd

from ..models import SimpleResNet18, MobileViTModel, LeViTModel
from ..representations import GrayscaleConverter, RGBTrigramConverter
from ..data import ByteplotDataset
from ..logger import logger
from .metrics import calculate_metrics, print_metrics_report
from .losses import WeightedBCELoss

def train_model(config_path="configs/train.yaml"):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize representation converter
    if config['representation'] == 'grayscale':
        converter = GrayscaleConverter()
    elif config['representation'] == 'rgb':
        converter = RGBTrigramConverter()
    else:
        raise ValueError(f"Unknown representation: {config['representation']}")

    # Initialize model
    backbone = config['backbone']
    num_classes = 1  # Binary classification

    if backbone == 'resnet18':
        model = SimpleResNet18(num_classes=num_classes)
    elif backbone == 'mobilevit_xxs':
        model = MobileViTModel(num_classes=num_classes, model_name='mobilevit_xxs')
    elif backbone == 'levit_128s':
        model = LeViTModel(num_classes=num_classes, model_name='levit_128s')
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Setup training
    criterion = WeightedBCELoss(pos_weight=config.get('pos_weight', 1.0))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Prepare datasets
    train_dataset = ByteplotDataset(
        config['train_manifest_path'],
        representation=config['representation'],
        split='train'
    )
    val_dataset = ByteplotDataset(
        config['train_manifest_path'],
        representation=config['representation'],
        split='val'
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'] * 2
    )

    # Training loop
    best_val_f1 = 0.0
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0

        # Training batches
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images).squeeze()
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()

                val_preds.extend(predictions.cpu().numpy())
                val_probs.extend(probabilities.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        metrics = calculate_metrics(
            val_preds, val_labels, val_probs
        )

        # Save best model based on F1
        if metrics['f1_score'] > best_val_f1:
            best_val_f1 = metrics['f1_score']
            torch.save(
                model.state_dict(),
                Path(config['checkpoint_dir']) / "best_model.pth"
            )

        logger.info(f"Epoch {epoch+1}/{config['epochs']}")
        logger.info(f"  Loss: {epoch_loss/len(train_loader):.4f}")
        print_metrics_report(metrics)

    # Save final model
    final_model_path = Path(config['final_model_path']).with_suffix('.pth')
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    return model

if __name__ == "__main__":
    train_model()
