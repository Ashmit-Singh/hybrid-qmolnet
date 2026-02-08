"""
Model Evaluator Module

Complete evaluation pipeline for trained models including:
- Test set evaluation
- Metric computation
- Result visualization
- Model comparison
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Dict, List, Tuple, Optional, Any
import json

from .metrics import (
    compute_metrics,
    compute_confusion_matrix,
    compute_roc_curve,
    compute_pr_curve,
    get_classification_report,
    print_metrics,
    MetricsTracker,
)


class ModelEvaluator:
    """
    Comprehensive model evaluator for molecular property prediction.
    
    Handles:
    - Running inference on test set
    - Computing all relevant metrics
    - Generating predictions and probabilities
    - Creating evaluation reports
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        model_name: str = "Model",
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained PyTorch model
            device: Device for inference
            model_name: Name for reporting
        """
        self.model = model
        self.model_name = model_name
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
        
        # Store evaluation results
        self.predictions: Optional[np.ndarray] = None
        self.probabilities: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.metrics: Optional[Dict[str, float]] = None
    
    @torch.no_grad()
    def predict(self, test_loader: PyGDataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions on test set.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Tuple of (predictions, probabilities, true_labels)
        """
        all_preds = []
        all_probs = []
        all_labels = []
        
        for batch in test_loader:
            batch = batch.to(self.device)
            
            logits = self.model.forward_batch(batch)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            all_labels.extend(batch.y.cpu().numpy())
        
        self.predictions = np.array(all_preds)
        self.probabilities = np.array(all_probs)
        self.labels = np.array(all_labels)
        
        return self.predictions, self.probabilities, self.labels
    
    def evaluate(self, test_loader: PyGDataLoader) -> Dict[str, float]:
        """
        Run full evaluation on test set.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        if self.predictions is None:
            self.predict(test_loader)
        
        # Compute metrics
        self.metrics = compute_metrics(
            self.labels,
            self.predictions,
            self.probabilities,
        )
        
        return self.metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix from last evaluation."""
        if self.predictions is None or self.labels is None:
            raise ValueError("Run evaluate() first")
        return compute_confusion_matrix(self.labels, self.predictions)
    
    def get_roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ROC curve data from last evaluation."""
        if self.probabilities is None or self.labels is None:
            raise ValueError("Run evaluate() first")
        return compute_roc_curve(self.labels, self.probabilities)
    
    def get_pr_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get precision-recall curve data."""
        if self.probabilities is None or self.labels is None:
            raise ValueError("Run evaluate() first")
        return compute_pr_curve(self.labels, self.probabilities)
    
    def get_classification_report(self) -> str:
        """Get formatted classification report."""
        if self.predictions is None or self.labels is None:
            raise ValueError("Run evaluate() first")
        return get_classification_report(
            self.labels,
            self.predictions,
            target_names=['Inactive', 'Active'],
        )
    
    def print_results(self) -> None:
        """Print evaluation results."""
        if self.metrics is None:
            raise ValueError("Run evaluate() first")
        
        print_metrics(self.metrics, self.model_name)
        print(self.get_classification_report())
        print("\nConfusion Matrix:")
        print(self.get_confusion_matrix())
    
    def save_results(self, output_dir: str) -> None:
        """
        Save evaluation results to files.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        if self.metrics is not None:
            metrics_path = os.path.join(output_dir, f"{self.model_name}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        
        # Save predictions
        if self.predictions is not None:
            preds_path = os.path.join(output_dir, f"{self.model_name}_predictions.npz")
            np.savez(
                preds_path,
                predictions=self.predictions,
                probabilities=self.probabilities,
                labels=self.labels,
            )
        
        print(f"Results saved to {output_dir}")
    
    @torch.no_grad()
    def get_embeddings(
        self,
        test_loader: PyGDataLoader,
        layer: str = 'gnn',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings for visualization.
        
        Args:
            test_loader: Test data loader
            layer: Which layer's embeddings to extract
        
        Returns:
            Tuple of (embeddings, labels)
        """
        all_embeddings = []
        all_labels = []
        
        for batch in test_loader:
            batch = batch.to(self.device)
            
            if hasattr(self.model, 'get_embeddings_batch'):
                emb = self.model.get_embeddings_batch(batch, layer=layer)
            else:
                emb = self.model.get_embeddings_batch(batch)
            
            all_embeddings.append(emb.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        labels = np.array(all_labels)
        
        return embeddings, labels


def compare_models(
    models: Dict[str, nn.Module],
    test_loader: PyGDataLoader,
    device: Optional[torch.device] = None,
    output_dir: Optional[str] = None,
) -> MetricsTracker:
    """
    Compare multiple models on the same test set.
    
    Args:
        models: Dictionary of model_name -> model
        test_loader: Test data loader
        device: Device for inference
        output_dir: Optional directory to save results
    
    Returns:
        MetricsTracker with all results
    """
    tracker = MetricsTracker()
    
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        evaluator = ModelEvaluator(
            model=model,
            device=device,
            model_name=model_name,
        )
        
        metrics = evaluator.evaluate(test_loader)
        tracker.add(model_name, metrics)
        
        if output_dir:
            evaluator.save_results(output_dir)
    
    print(tracker.summary())
    
    # Print best model
    best = tracker.get_best('accuracy', 'max')
    print(f"\nBest model by accuracy: {best.get('model_name')} ({best.get('accuracy', 0):.4f})")
    
    return tracker


if __name__ == "__main__":
    # Demo
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        
        def forward_batch(self, data):
            x = data.x.mean(dim=0, keepdim=True).expand(data.num_graphs, -1)
            return self.fc(x[:, :10])
        
        def get_embeddings_batch(self, data, layer='gnn'):
            x = data.x.mean(dim=0, keepdim=True).expand(data.num_graphs, -1)
            return x[:, :10]
    
    model = DummyModel()
    
    # Create dummy test data
    data_list = []
    for i in range(50):
        x = torch.randn(5, 10)
        edge_index = torch.randint(0, 5, (2, 10))
        y = torch.tensor([i % 2])
        data_list.append(Data(x=x, edge_index=edge_index, y=y))
    
    test_loader = PyGDataLoader(data_list, batch_size=16)
    
    # Evaluate
    evaluator = ModelEvaluator(model, model_name="DummyModel")
    metrics = evaluator.evaluate(test_loader)
    evaluator.print_results()
    
    # Get embeddings
    embeddings, labels = evaluator.get_embeddings(test_loader)
    print(f"\nEmbeddings shape: {embeddings.shape}")
