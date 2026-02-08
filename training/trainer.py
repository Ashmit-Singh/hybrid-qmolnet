"""
Trainer Module

Reusable training loop for both hybrid and classical models.
Supports metric logging, callbacks, and history tracking.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Optional, List, Dict, Any, Callable, Tuple
from tqdm import tqdm
import numpy as np

from .callbacks import Callback, EarlyStoppingCallback, CheckpointCallback


class TrainingHistory:
    """
    Records training history for visualization and analysis.
    """
    
    def __init__(self):
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.train_acc: List[float] = []
        self.val_acc: List[float] = []
        self.learning_rates: List[float] = []
        self.epoch_times: List[float] = []
        self.metrics: Dict[str, List[float]] = {}
    
    def add(self, metrics: Dict[str, float], lr: float, epoch_time: float) -> None:
        """
        Add metrics for an epoch.
        
        Args:
            metrics: Dictionary of metric values
            lr: Current learning rate
            epoch_time: Time taken for epoch
        """
        self.train_loss.append(metrics.get('train_loss', 0))
        self.val_loss.append(metrics.get('val_loss', 0))
        self.train_acc.append(metrics.get('train_acc', 0))
        self.val_acc.append(metrics.get('val_acc', 0))
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
        
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        """Get epoch with best metric value."""
        if metric not in self.metrics or len(self.metrics[metric]) == 0:
            return 0
        
        values = self.metrics[metric]
        if mode == 'min':
            return int(np.argmin(values))
        return int(np.argmax(values))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert history to dictionary."""
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_acc': self.train_acc,
            'val_acc': self.val_acc,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'metrics': self.metrics,
        }


class Trainer:
    """
    Unified trainer for hybrid and classical models.
    
    Features:
    - Flexible training loop for graph and non-graph models
    - Metric logging and history tracking
    - Callback system for early stopping and checkpointing
    - Support for learning rate scheduling
    - GPU/CPU device handling
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Callback]] = None,
        model_name: str = "Model",
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer (default: AdamW)
            scheduler: Learning rate scheduler
            criterion: Loss function (default: CrossEntropyLoss)
            device: Device to train on
            callbacks: List of callbacks
            model_name: Name for logging
        """
        self.model = model
        self.model_name = model_name
        self.history = TrainingHistory()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Note: model may have quantum layer that must run on CPU
        # Model handles device management internally
        self.model.to(self.device)
        
        # Set optimizer
        if optimizer is None:
            self.optimizer = AdamW(
                model.parameters(),
                lr=1e-3,
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer
        
        # Set scheduler
        self.scheduler = scheduler
        
        # Set criterion
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        # Set callbacks
        self.callbacks = callbacks or []
    
    def train_epoch(self, train_loader: PyGDataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model.forward_batch(batch)
            loss = self.criterion(logits, batch.y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * batch.num_graphs
            preds = logits.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_graphs
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, val_loader: PyGDataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in val_loader:
            batch = batch.to(self.device)
            
            logits = self.model.forward_batch(batch)
            loss = self.criterion(logits, batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            preds = logits.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_graphs
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def fit(
        self,
        train_loader: PyGDataLoader,
        val_loader: PyGDataLoader,
        num_epochs: int = 100,
        verbose: bool = True,
    ) -> TrainingHistory:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            verbose: Whether to print progress
        
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print()
        
        # Notify callbacks
        for callback in self.callbacks:
            callback.on_train_start(self)
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Notify callbacks
            for callback in self.callbacks:
                callback.on_epoch_start(epoch, self)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record metrics
            epoch_time = time.time() - epoch_start
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }
            self.history.add(metrics, current_lr, epoch_time)
            
            # Print progress
            if verbose:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"Time: {epoch_time:.1f}s")
            
            # Notify callbacks and check for early stopping
            should_stop = False
            for callback in self.callbacks:
                if callback.on_epoch_end(epoch, metrics, self):
                    should_stop = True
            
            if should_stop:
                break
        
        # Notify callbacks
        for callback in self.callbacks:
            callback.on_train_end(self)
        
        # Print summary
        best_epoch = self.history.get_best_epoch('val_loss', 'min')
        print(f"\nTraining complete!")
        print(f"Best validation loss: {self.history.val_loss[best_epoch]:.4f} at epoch {best_epoch+1}")
        print(f"Best validation accuracy: {self.history.val_acc[best_epoch]:.4f}")
        
        return self.history
    
    def load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {path}")


class DescriptorTrainer:
    """
    Specialized trainer for descriptor-based MLP models.
    
    Works with standard PyTorch DataLoaders instead of PyG DataLoaders.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        model_name: str = "DescriptorMLP",
    ):
        self.model = model
        self.model_name = model_name
        self.history = TrainingHistory()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        self.optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
    
    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> TrainingHistory:
        """
        Train the descriptor model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            num_epochs: Number of epochs
            batch_size: Batch size
            verbose: Whether to print progress
        
        Returns:
            Training history
        """
        print(f"\nTraining {self.model_name}...")
        
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        
        # Create DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        for epoch in range(num_epochs):
            # Train
            self.model.train()
            total_loss = 0
            correct = 0
            
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * len(X_batch)
                correct += (logits.argmax(1) == y_batch).sum().item()
            
            train_loss = total_loss / len(X_train)
            train_acc = correct / len(X_train)
            
            # Validate
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val)
                val_loss = self.criterion(val_logits, y_val).item()
                val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
            
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }
            self.history.add(metrics, self.optimizer.param_groups[0]['lr'], 0)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f}")
        
        print(f"\nTraining complete!")
        best_idx = np.argmin(self.history.val_loss)
        print(f"Best val loss: {self.history.val_loss[best_idx]:.4f} at epoch {best_idx+1}")
        
        return self.history


if __name__ == "__main__":
    # Demo usage
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
    
    model = DummyModel()
    
    # Create dummy data
    data_list = []
    for _ in range(100):
        x = torch.randn(5, 10)
        edge_index = torch.randint(0, 5, (2, 10))
        y = torch.randint(0, 2, (1,))
        data_list.append(Data(x=x, edge_index=edge_index, y=y))
    
    train_loader = PyGDataLoader(data_list[:80], batch_size=16, shuffle=True)
    val_loader = PyGDataLoader(data_list[80:], batch_size=16)
    
    # Train
    trainer = Trainer(
        model=model,
        callbacks=[
            EarlyStoppingCallback(patience=5, verbose=True),
        ],
        model_name="DummyModel",
    )
    
    history = trainer.fit(train_loader, val_loader, num_epochs=20)
    print(f"\nFinal history: {len(history.train_loss)} epochs")
