"""
Plotting Utilities Module

Generate publication-quality plots for:
- Training/validation curves
- ROC curves
- Confusion matrices
- Metric comparisons
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import auc


# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Optional path to save figure
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss plot
    ax1 = axes[0]
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2 = axes[1]
    if 'train_acc' in history:
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Curves', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: Optional[float] = None,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Plot ROC curve with AUC score.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: Area under the curve (computed if None)
        model_name: Model name for legend
        save_path: Optional path to save figure
        ax: Optional axes to plot on
        color: Line color
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if roc_auc is None:
        roc_auc = auc(fpr, tpr)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    ax.plot(
        fpr, tpr,
        color=color,
        lw=2,
        label=f'{model_name} (AUC = {roc_auc:.3f})'
    )
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    return fig


def plot_multiple_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> plt.Figure:
    """
    Plot multiple ROC curves for model comparison.
    
    Args:
        roc_data: Dict of model_name -> (fpr, tpr, auc)
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))
    
    for (model_name, (fpr, tpr, roc_auc)), color in zip(roc_data.items(), colors):
        ax.plot(
            fpr, tpr,
            color=color,
            lw=2,
            label=f'{model_name} (AUC = {roc_auc:.3f})'
        )
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC comparison saved to {save_path}")
    
    return fig


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        model_name: Model name for legend
        save_path: Optional path to save
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    pr_auc = auc(recall, precision)
    
    ax.plot(recall, precision, lw=2, label=f'{model_name} (AUC = {pr_auc:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix [n_classes, n_classes]
        class_names: Class labels
        save_path: Optional path to save
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
    
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = ['Inactive', 'Active']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={'size': 16},
        cbar_kws={'label': 'Count'},
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot bar chart comparing metrics across models.
    
    Args:
        metrics_dict: Dict of model_name -> {metric: value}
        metric_names: Metrics to include
        save_path: Optional path to save
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if metric_names is None:
        metric_names = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']
    
    model_names = list(metrics_dict.keys())
    n_models = len(model_names)
    n_metrics = len(metric_names)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    for i, (model_name, metrics) in enumerate(metrics_dict.items()):
        values = [metrics.get(m, 0) for m in metric_names]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i])
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.2f}',
                ha='center',
                va='bottom',
                fontsize=8,
            )
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names])
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics comparison saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Demo
    import numpy as np
    
    # Demo training curves
    n_epochs = 50
    history = {
        'train_loss': [1.0 * np.exp(-0.05 * i) + 0.1 + np.random.normal(0, 0.02) for i in range(n_epochs)],
        'val_loss': [1.1 * np.exp(-0.05 * i) + 0.15 + np.random.normal(0, 0.03) for i in range(n_epochs)],
        'train_acc': [0.5 + 0.4 * (1 - np.exp(-0.08 * i)) + np.random.normal(0, 0.02) for i in range(n_epochs)],
        'val_acc': [0.5 + 0.35 * (1 - np.exp(-0.08 * i)) + np.random.normal(0, 0.03) for i in range(n_epochs)],
    }
    
    fig = plot_training_curves(history, title="Demo Training")
    plt.show()
    
    # Demo ROC curve
    fpr = np.array([0, 0.1, 0.2, 0.4, 1.0])
    tpr = np.array([0, 0.5, 0.7, 0.9, 1.0])
    fig = plot_roc_curve(fpr, tpr, model_name="Demo Model")
    plt.show()
    
    # Demo confusion matrix
    cm = np.array([[40, 10], [5, 45]])
    fig = plot_confusion_matrix(cm)
    plt.show()
    
    # Demo metrics comparison
    metrics = {
        'Hybrid QMolNet': {'accuracy': 0.85, 'roc_auc': 0.91, 'f1': 0.83, 'precision': 0.80, 'recall': 0.86},
        'GNN Baseline': {'accuracy': 0.82, 'roc_auc': 0.88, 'f1': 0.80, 'precision': 0.78, 'recall': 0.82},
        'MLP Baseline': {'accuracy': 0.75, 'roc_auc': 0.80, 'f1': 0.72, 'precision': 0.70, 'recall': 0.74},
    }
    fig = plot_metrics_comparison(metrics)
    plt.show()
