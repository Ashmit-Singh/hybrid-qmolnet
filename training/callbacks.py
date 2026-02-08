"""
Training Callbacks Module

Implements callback classes for training control including:
- Early stopping based on validation metrics
- Model checkpointing
- Progress bar tracking
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from tqdm import tqdm


class Callback:
    """Base callback class."""
    
    def on_epoch_start(self, epoch: int, trainer: 'Trainer') -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: 'Trainer') -> bool:
        """
        Called at the end of each epoch.
        
        Returns:
            True if training should stop
        """
        return False
    
    def on_train_start(self, trainer: 'Trainer') -> None:
        """Called at the start of training."""
        pass
    
    def on_train_end(self, trainer: 'Trainer') -> None:
        """Called at the end of training."""
        pass
    
    def on_batch_end(self, batch_idx: int, loss: float, trainer: 'Trainer') -> None:
        """Called at the end of each batch."""
        pass


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback to prevent overfitting.
    
    Monitors a metric and stops training if no improvement
    is seen for a specified number of epochs.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min',
        monitor: str = 'val_loss',
        verbose: bool = True,
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            monitor: Metric name to monitor
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
    
    def on_train_start(self, trainer: 'Trainer') -> None:
        """Reset state at training start."""
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: 'Trainer') -> bool:
        """Check if training should stop."""
        if self.monitor not in metrics:
            return False
        
        score = metrics[self.monitor]
        
        if self.mode == 'min':
            improved = self.best_score is None or score < self.best_score - self.min_delta
        else:
            improved = self.best_score is None or score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"  [EarlyStopping] New best {self.monitor}: {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"  [EarlyStopping] Stopping at epoch {epoch}. Best: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True
        
        return False


class CheckpointCallback(Callback):
    """
    Model checkpointing callback.
    
    Saves model checkpoints based on validation metrics.
    """
    
    def __init__(
        self,
        save_dir: str = 'outputs/checkpoints',
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize checkpointing.
        
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' for loss, 'max' for accuracy
            save_best_only: Only save when metric improves
            verbose: Whether to print messages
        """
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        self.best_score = None
        
        os.makedirs(save_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: 'Trainer') -> bool:
        """Save checkpoint if appropriate."""
        if self.monitor not in metrics:
            return False
        
        score = metrics[self.monitor]
        
        should_save = False
        if self.save_best_only:
            if self.mode == 'min':
                should_save = self.best_score is None or score < self.best_score
            else:
                should_save = self.best_score is None or score > self.best_score
            
            if should_save:
                self.best_score = score
        else:
            should_save = True
        
        if should_save:
            checkpoint_path = os.path.join(
                self.save_dir,
                f"checkpoint_epoch{epoch}_{self.monitor}_{score:.4f}.pt"
            )
            
            self._save_checkpoint(trainer, epoch, metrics, checkpoint_path)
            
            # Also save as 'best.pt'
            if self.save_best_only:
                best_path = os.path.join(self.save_dir, "best.pt")
                self._save_checkpoint(trainer, epoch, metrics, best_path)
            
            if self.verbose:
                print(f"  [Checkpoint] Saved to {checkpoint_path}")
        
        return False
    
    def _save_checkpoint(
        self,
        trainer: 'Trainer',
        epoch: int,
        metrics: Dict[str, float],
        path: str,
    ) -> None:
        """Save checkpoint to file."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'metrics': metrics,
        }
        torch.save(checkpoint, path)


class ProgressCallback(Callback):
    """
    Progress bar callback using tqdm.
    """
    
    def __init__(self, num_epochs: int):
        """
        Initialize progress tracking.
        
        Args:
            num_epochs: Total number of epochs
        """
        self.num_epochs = num_epochs
        self.pbar = None
    
    def on_train_start(self, trainer: 'Trainer') -> None:
        """Create progress bar."""
        self.pbar = tqdm(total=self.num_epochs, desc="Training", unit="epoch")
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: 'Trainer') -> bool:
        """Update progress bar."""
        if self.pbar is not None:
            # Format metrics for display
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.pbar.set_postfix_str(metrics_str)
            self.pbar.update(1)
        return False
    
    def on_train_end(self, trainer: 'Trainer') -> None:
        """Close progress bar."""
        if self.pbar is not None:
            self.pbar.close()
