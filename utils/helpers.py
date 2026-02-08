"""
Helper Utilities Module

Common utility functions for reproducibility, device management,
and model inspection used throughout the project.
"""

import os
import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDA deterministic operations
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic operations (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed} for reproducibility")


def get_device(prefer_cuda: bool = True, device_id: int = 0) -> torch.device:
    """
    Get the optimal available device for computation.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        device_id: CUDA device ID to use
    
    Returns:
        torch.device object
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        print(f"Using CUDA device: {torch.cuda.get_device_name(device_id)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS device")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
    
    Returns:
        Total parameter count
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model: torch.nn.Module, model_name: str = "Model") -> None:
    """
    Print a summary of the model architecture and parameters.
    
    Args:
        model: PyTorch model
        model_name: Name to display in the summary
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print(f"\n{'='*60}")
    print(f"{model_name} Summary")
    print(f"{'='*60}")
    print(model)
    print(f"\n{'-'*60}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print(f"{'='*60}\n")


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.
    
    Monitors a metric and stops training if no improvement
    is seen for a specified number of epochs.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for metrics
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            epoch: Current epoch number
        
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = self.best_score is None or score < self.best_score - self.min_delta
        else:
            improved = self.best_score is None or score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"  Early stopping: New best score = {score:.4f}")
        else:
            self.counter += 1
            if self.verbose and self.counter > 0:
                print(f"  Early stopping: No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  Early stopping triggered at epoch {epoch}")
        
        return self.early_stop
    
    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking training metrics across batches.
    """
    
    def __init__(self, name: str = 'Metric'):
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """
        Update the meter with a new value.
        
        Args:
            val: Value to add
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    additional_info: Optional[dict] = None
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        path: Path to save checkpoint
        additional_info: Additional information to store
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> dict:
    """
    Load a training checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
    
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {path} (epoch {checkpoint['epoch']})")
    return checkpoint


if __name__ == "__main__":
    # Demo
    set_seed(42)
    device = get_device()
    
    # Test AverageMeter
    meter = AverageMeter("Loss")
    for i in range(10):
        meter.update(np.random.random())
    print(f"Average: {meter.avg:.4f}")
    
    # Test EarlyStopping
    stopper = EarlyStopping(patience=3, mode='min')
    losses = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.88]
    for epoch, loss in enumerate(losses):
        if stopper(loss, epoch):
            print(f"Stopped at epoch {epoch}")
            break
