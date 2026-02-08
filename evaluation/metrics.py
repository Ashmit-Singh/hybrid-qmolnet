"""
Evaluation Metrics Module

Computes classification metrics for model evaluation:
- Accuracy
- ROC-AUC
- F1 Score
- Precision / Recall
- Confusion Matrix
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from typing import Dict, List, Tuple, Optional, Any


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels [N]
        y_pred: Predicted class labels [N]
        y_prob: Predicted probabilities for positive class [N] (optional)
    
    Returns:
        Dictionary of metric names to values
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
    }
    
    # Compute ROC-AUC if probabilities are available
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            # Can fail if only one class present
            metrics['roc_auc'] = 0.5
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Confusion matrix [2, 2] for binary classification
    """
    return confusion_matrix(y_true, y_pred)


def compute_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities for positive class
    
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    return roc_curve(y_true, y_prob)


def compute_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities for positive class
    
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    return precision_recall_curve(y_true, y_prob)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """
    Generate a text classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        target_names: Optional class names
    
    Returns:
        Formatted classification report string
    """
    if target_names is None:
        target_names = ['Class 0', 'Class 1']
    
    return classification_report(y_true, y_pred, target_names=target_names)


class MetricsTracker:
    """
    Tracks metrics across multiple evaluations.
    
    Useful for comparing models or tracking performance over time.
    """
    
    def __init__(self):
        self.records: List[Dict[str, Any]] = []
    
    def add(
        self,
        model_name: str,
        metrics: Dict[str, float],
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a metrics record.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
            additional_info: Optional extra information
        """
        record = {
            'model_name': model_name,
            **metrics,
        }
        if additional_info:
            record.update(additional_info)
        
        self.records.append(record)
    
    def get_best(self, metric: str = 'accuracy', mode: str = 'max') -> Dict[str, Any]:
        """
        Get the record with the best metric value.
        
        Args:
            metric: Metric to compare
            mode: 'max' or 'min'
        
        Returns:
            Best record dictionary
        """
        if not self.records:
            return {}
        
        if mode == 'max':
            return max(self.records, key=lambda x: x.get(metric, 0))
        return min(self.records, key=lambda x: x.get(metric, float('inf')))
    
    def to_dataframe(self):
        """Convert records to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.records)
    
    def summary(self) -> str:
        """Generate a summary of all tracked models."""
        if not self.records:
            return "No records tracked."
        
        lines = ["\n" + "="*60]
        lines.append("Model Comparison Summary")
        lines.append("="*60)
        
        for record in self.records:
            model_name = record.get('model_name', 'Unknown')
            lines.append(f"\n{model_name}:")
            for key, value in record.items():
                if key != 'model_name':
                    if isinstance(value, float):
                        lines.append(f"  {key}: {value:.4f}")
                    else:
                        lines.append(f"  {key}: {value}")
        
        lines.append("\n" + "="*60)
        return "\n".join(lines)


def print_metrics(
    metrics: Dict[str, float],
    model_name: str = "Model",
) -> None:
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Model name for header
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Metrics")
    print(f"{'='*50}")
    
    for name, value in metrics.items():
        print(f"  {name:15s}: {value:.4f}")
    
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Demo
    np.random.seed(42)
    
    # Generate dummy predictions
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.2, 0.6, 0.8, 0.9, 0.3, 0.7, 0.4, 0.2, 0.85, 0.15])
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, "Demo Model")
    
    # Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(get_classification_report(y_true, y_pred))
    
    # Test MetricsTracker
    tracker = MetricsTracker()
    tracker.add("Model A", {'accuracy': 0.85, 'f1': 0.82})
    tracker.add("Model B", {'accuracy': 0.88, 'f1': 0.85})
    print(tracker.summary())
