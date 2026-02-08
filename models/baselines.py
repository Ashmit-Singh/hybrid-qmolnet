"""
Baseline Models Module

Classical baseline models for comparison with the hybrid quantum approach.
Includes GNN-only and descriptor-based MLP classifiers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import Optional

from .gnn_encoder import GNNEncoder


class GNNClassifier(nn.Module):
    """
    Pure classical GNN-based classifier.
    
    Uses the same GNN encoder as the hybrid model but replaces
    the quantum layer with a classical MLP head. Serves as the
    primary baseline for comparing quantum vs classical processing.
    
    Architecture:
    GNN Encoder → MLP Classifier
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        gnn_hidden_dim: int = 64,
        gnn_embedding_dim: int = 32,
        gnn_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize the GNN classifier.
        
        Args:
            node_feature_dim: Input node feature dimension
            gnn_hidden_dim: GNN hidden layer dimension
            gnn_embedding_dim: GNN output embedding dimension
            gnn_layers: Number of GNN layers
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.gnn_embedding_dim = gnn_embedding_dim
        self.num_classes = num_classes
        
        # GNN Encoder (same as hybrid model)
        self.gnn_encoder = GNNEncoder(
            input_dim=node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            embedding_dim=gnn_embedding_dim,
            num_layers=gnn_layers,
            conv_type='gcn',
            dropout=dropout,
            pooling='mean',
        )
        
        # MLP Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(gnn_embedding_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim // 2, num_classes),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
        
        Returns:
            Class logits [batch_size, num_classes]
        """
        # GNN encoding
        embedding = self.gnn_encoder(x, edge_index, batch)
        
        # Classification
        logits = self.classifier(embedding)
        
        return logits
    
    def forward_batch(self, data: Batch) -> torch.Tensor:
        """Convenience method for PyG Batch objects."""
        return self.forward(data.x, data.edge_index, data.batch)
    
    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract GNN embeddings."""
        return self.gnn_encoder(x, edge_index, batch)
    
    def get_embeddings_batch(self, data: Batch) -> torch.Tensor:
        """Extract embeddings from batch."""
        return self.get_embeddings(data.x, data.edge_index, data.batch)
    
    def __repr__(self) -> str:
        return (
            f"GNNClassifier(\n"
            f"  node_features={self.node_feature_dim},\n"
            f"  embedding_dim={self.gnn_embedding_dim},\n"
            f"  num_classes={self.num_classes}\n"
            f")"
        )


class DescriptorMLP(nn.Module):
    """
    Molecular descriptor-based MLP classifier.
    
    Uses pre-computed molecular descriptors (from RDKit) instead of
    graph neural networks. Serves as a simple baseline that doesn't
    require graph processing.
    
    Architecture:
    Molecular Descriptors → MLP → Classification
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: tuple = (64, 32, 16),
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        """
        Initialize the descriptor MLP.
        
        Args:
            input_dim: Number of molecular descriptors
            hidden_dims: Tuple of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
        # For descriptor normalization
        self.register_buffer('mean', torch.zeros(input_dim))
        self.register_buffer('std', torch.ones(input_dim))
        self.fitted = False
    
    def fit_normalization(self, descriptors: torch.Tensor) -> None:
        """
        Fit normalization parameters from training data.
        
        Args:
            descriptors: Training descriptors [num_samples, input_dim]
        """
        self.mean = descriptors.mean(dim=0)
        self.std = descriptors.std(dim=0)
        self.std[self.std < 1e-6] = 1.0  # Avoid division by zero
        self.fitted = True
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input descriptors."""
        if self.fitted:
            return (x - self.mean) / self.std
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Molecular descriptors [batch_size, input_dim]
        
        Returns:
            Class logits [batch_size, num_classes]
        """
        x = self.normalize(x)
        return self.mlp(x)
    
    def __repr__(self) -> str:
        return (
            f"DescriptorMLP(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dims={self.hidden_dims},\n"
            f"  num_classes={self.num_classes}\n"
            f")"
        )


def print_baseline_summary(model: nn.Module, name: str = "Baseline") -> None:
    """
    Print summary for baseline models.
    
    Args:
        model: Baseline model
        name: Model name for display
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*60)
    print(f"{name} Summary")
    print("="*60)
    print(model)
    print("-"*60)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Demo and testing
    from torch_geometric.data import Data, Batch
    
    # Create dummy data
    num_nodes = 12
    node_feature_dim = 145
    descriptor_dim = 10
    
    # GNN Classifier test
    print("Testing GNN Classifier...")
    gnn_model = GNNClassifier(
        node_feature_dim=node_feature_dim,
        gnn_hidden_dim=64,
        gnn_embedding_dim=32,
        gnn_layers=3,
        num_classes=2,
    )
    print_baseline_summary(gnn_model, "GNN Classifier")
    
    x = torch.randn(num_nodes, node_feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, 25))
    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data] * 4)
    
    with torch.no_grad():
        logits = gnn_model.forward_batch(batch)
    print(f"GNN Classifier output shape: {logits.shape}\n")
    
    # Descriptor MLP test
    print("Testing Descriptor MLP...")
    mlp_model = DescriptorMLP(
        input_dim=descriptor_dim,
        hidden_dims=(64, 32, 16),
        num_classes=2,
    )
    print_baseline_summary(mlp_model, "Descriptor MLP")
    
    descriptors = torch.randn(16, descriptor_dim)
    
    # Fit normalization
    mlp_model.fit_normalization(descriptors)
    
    with torch.no_grad():
        logits = mlp_model(descriptors)
    print(f"Descriptor MLP output shape: {logits.shape}")
