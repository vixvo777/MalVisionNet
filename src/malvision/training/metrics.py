"""
Evaluation metrics for binary classification.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

import pandas as pd


def calculate_metrics(predictions, labels, probabilities):
    """
    Calculate comprehensive metrics for binary classification.

    Args:
        predictions: Predicted binary labels (0 or 1)
        labels: True binary labels (0 or 1)
        probabilities: Predicted probabilities for class 1

    Returns:
        Dictionary containing all metrics
    """
    metrics = {}

    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['precision'] = precision_score(labels, predictions, zero_division=0)
    metrics['recall'] = recall_score(labels, predictions, zero_division=0)
    metrics['f1_score'] = f1_score(labels, predictions, zero_division=0)
    metrics['roc_auc'] = roc_auc_score(labels, probabilities)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    metrics['true_negative'] = tn
    metrics['false_positive'] = fp
    metrics['false_negative'] = fn
    metrics['true_positive'] = tp

    # Class-specific metrics
    if tp + fp > 0:
        metrics['precision_positive'] = tp / (tp + fp)
    else:
        metrics['precision_positive'] = 0

    if tn + fn > 0:
        metrics['precision_negative'] = tn / (tn + fn)
    else:
        metrics['precision_negative'] = 0

    if tp + fn > 0:
        metrics['recall_positive'] = tp / (tp + fn)
    else:
        metrics['recall_positive'] = 0

    if tn + fp > 0:
        metrics['recall_negative'] = tn / (tn + fp)
    else:
        metrics['recall_negative'] = 0

    # F1 scores per class
    if (tp + fp) > 0 and (tp + fn) > 0:
        metrics['f1_positive'] = 2 * tp / (2 * tp + fp + fn)
    else:
        metrics['f1_positive'] = 0

    if (tn + fn) > 0 and (tn + fp) > 0:
        metrics['f1_negative'] = 2 * tn / (2 * tn + fp + fn)
    else:
        metrics['f1_negative'] = 0

    # Advanced metrics
    metrics['balanced_accuracy'] = (metrics['recall_positive'] + metrics['recall_negative']) / 2

    # Threshold-based metrics (common in malware detection)
    # Sensitivity (same as recall_positive)
    metrics['tpr'] = metrics['recall_positive']  # True Positive Rate

    # False Positive Rate
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Positive Predictive Value (PPV)
    metrics['ppv'] = metrics['precision_positive']

    # Negative Predictive Value (NPV)
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0

    return metrics


def print_metrics_report(metrics):
    """Print formatted metrics report."""
    print("=" * 60)
    print("MALWARE DETECTION PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall (TPR):  {metrics['recall']:.4f}")
    print(f"F1 Score:      {metrics['f1_score']:.4f}")
    print(f"ROC AUC:       {metrics['roc_auc']:.4f}")
    print(f"Balanced Acc:  {metrics['balanced_accuracy']:.4f}")
    print()
    print("Confusion Matrix Components:")
    print(f"  True Negatives (TN):  {metrics['true_negative']}")
    print(f"  False Positives (FP): {metrics['false_positive']}")
    print(f"  False Negatives (FN): {metrics['false_negative']}")
    print(f"  True Positives (TP):  {metrics['true_positive']}")
    print()
    print("Malware Detection Specific:")
    print(f"  Sensitivity (TPR):    {metrics['tpr']:.4f}")
    print(f"  False Positive Rate:  {metrics['fpr']:.4f}")
    print(f"  Positive Predictive Value: {metrics['ppv']:.4f}")
    print(f"  Negative Predictive Value: {metrics['npv']:.4f}")
    print("=" * 60)


def save_metrics_to_csv(metrics, save_path):
    """Save metrics to CSV file for further analysis."""
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(save_path, index=False)
    print(f"Metrics saved to: {save_path}")


def load_predictions_from_file(filepath):
    """Load predictions and labels from CSV file."""
    df = pd.read_csv(filepath)
    return df['true_label'], df['predicted_label'], df['probability']


def predict_torch_model(model, data_loader, device):
    """Make predictions using a PyTorch model."""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze()
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)
