"""
Graph Neural Network Encoder Module

Implements a message-passing GNN for molecular graph encoding.
Uses GCN/GAT layers with global pooling to produce fixed-size embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Batch
from typing import Optional, Literal


class GNNEncoder(nn.Module):
    """
    Message-passing Graph Neural Network encoder for molecular graphs.
    
    Architecture:
    - Multiple graph convolution layers (GCN or GAT)
    - ReLU activations with batch normalization
    - Global pooling to produce graph-level embeddings
    - Output projection to desired embedding dimension
    
    The encoder processes variable-size molecular graphs and produces
    fixed-size embedding vectors suitable for downstream classification.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        num_layers: int = 3,
        conv_type: Literal['gcn', 'gat'] = 'gcn',
        dropout: float = 0.2,
        pooling: Literal['mean', 'add', 'max'] = 'mean',
        use_batch_norm: bool = True,
    ):
        """
        Initialize the GNN encoder.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Hidden layer dimension
            embedding_dim: Output embedding dimension (32 for hybrid model)
            num_layers: Number of graph convolution layers (2-3 recommended)
            conv_type: Type of graph convolution ('gcn' or 'gat')
            dropout: Dropout rate for regularization
            pooling: Global pooling strategy
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim
            
            if conv_type == 'gcn':
                conv = GCNConv(in_channels, out_channels)
            elif conv_type == 'gat':
                # GAT with 4 attention heads
                conv = GATConv(in_channels, out_channels // 4, heads=4, concat=True)
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")
            
            self.convs.append(conv)
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(out_channels))
        
        # Global pooling function
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        # Output projection to embedding dimension
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the GNN encoder.
        
        Args:
            x: Node feature matrix [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]
        
        Returns:
            Graph embeddings [batch_size, embedding_dim]
        """
        # Handle single graph case (no batch vector)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Message passing layers
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            
            if self.use_batch_norm:
                h = self.batch_norms[i](h)
            
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Global pooling: aggregate node features to graph-level
        h = self.pool(h, batch)
        
        # Output projection
        embedding = self.output_proj(h)
        
        return embedding
    
    def forward_batch(self, data: Batch) -> torch.Tensor:
        """
        Convenience method for processing PyG Batch objects.
        
        Args:
            data: PyTorch Geometric Batch object
        
        Returns:
            Graph embeddings [batch_size, embedding_dim]
        """
        return self.forward(data.x, data.edge_index, data.batch)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  embedding_dim={self.embedding_dim},\n"
            f"  num_layers={self.num_layers},\n"
            f"  dropout={self.dropout}\n"
            f")"
        )


def print_gnn_summary(model: GNNEncoder) -> None:
    """
    Print a detailed summary of the GNN encoder architecture.
    
    Args:
        model: GNNEncoder instance
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*60)
    print("GNN Encoder Architecture Summary")
    print("="*60)
    print(f"\nInput dimension:     {model.input_dim}")
    print(f"Hidden dimension:    {model.hidden_dim}")
    print(f"Embedding dimension: {model.embedding_dim}")
    print(f"Number of layers:    {model.num_layers}")
    print(f"Dropout rate:        {model.dropout}")
    print(f"\nLayer-by-layer breakdown:")
    print("-"*60)
    
    print(f"1. Input Projection: Linear({model.input_dim} -> {model.hidden_dim})")
    
    for i, conv in enumerate(model.convs):
        print(f"{i+2}. Graph Conv {i+1}: {conv}")
        if model.use_batch_norm:
            print(f"   -> BatchNorm1d({model.hidden_dim})")
        print(f"   -> ReLU -> Dropout({model.dropout})")
    
    print(f"{len(model.convs)+2}. Global Pooling: {model.pool.__name__}")
    print(f"{len(model.convs)+3}. Output MLP: {model.hidden_dim} -> {model.hidden_dim} -> {model.embedding_dim}")
    
    print("-"*60)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Demo and testing
    from torch_geometric.data import Data, Batch
    
    # Create dummy molecular graph
    num_nodes = 10
    input_dim = 145  # Typical molecular feature dimension
    
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 20))
    data = Data(x=x, edge_index=edge_index)
    
    # Create batch of 4 graphs
    batch = Batch.from_data_list([data] * 4)
    
    # Initialize encoder
    encoder = GNNEncoder(
        input_dim=input_dim,
        hidden_dim=64,
        embedding_dim=32,
        num_layers=3,
        conv_type='gcn',
    )
    
    # Print summary
    print_gnn_summary(encoder)
    
    # Forward pass
    embeddings = encoder.forward_batch(batch)
    print(f"Input batch: {batch.num_graphs} graphs")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Expected: ({batch.num_graphs}, 32)")
