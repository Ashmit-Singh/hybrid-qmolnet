"""
Hybrid Quantum-Classical Model Module

Combines GNN encoder with variational quantum circuit for
molecular property prediction.

Architecture:
GNN → Linear Compression (32→8) → Normalization → Quantum Layer → Classifier
"""

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from typing import Optional

from .gnn_encoder import GNNEncoder
from .quantum_layer import VariationalQuantumLayer


class HybridQMolNet(nn.Module):
    """
    Hybrid Graph Neural Network + Variational Quantum Circuit model.
    
    This model combines:
    1. GNN Encoder: Extracts molecular graph embeddings (32-dim)
    2. Compression Layer: Reduces to qubit count (8-dim)
    3. Normalization: Prepares for quantum encoding
    4. VQC Layer: Processes through quantum circuit
    5. Classifier: Produces final logits
    
    The classical components (GNN, compression, classifier) can run on GPU,
    while the quantum layer runs on CPU simulator.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        gnn_hidden_dim: int = 64,
        gnn_embedding_dim: int = 32,
        gnn_layers: int = 3,
        n_qubits: int = 8,
        quantum_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.2,
        use_quantum: bool = True,
    ):
        """
        Initialize the hybrid model.
        
        Args:
            node_feature_dim: Input node feature dimension
            gnn_hidden_dim: GNN hidden layer dimension
            gnn_embedding_dim: GNN output embedding dimension
            gnn_layers: Number of GNN message-passing layers
            n_qubits: Number of qubits in quantum circuit
            quantum_layers: Number of quantum variational layers
            num_classes: Number of output classes
            dropout: Dropout rate
            use_quantum: Whether to use quantum layer (for ablation)
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.gnn_embedding_dim = gnn_embedding_dim
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.use_quantum = use_quantum
        
        # --- GNN Encoder ---
        self.gnn_encoder = GNNEncoder(
            input_dim=node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            embedding_dim=gnn_embedding_dim,
            num_layers=gnn_layers,
            conv_type='gcn',
            dropout=dropout,
            pooling='mean',
        )
        
        # --- Compression Layer ---
        # Reduce embedding dim to match qubit count: 32 → 8
        self.compression = nn.Sequential(
            nn.Linear(gnn_embedding_dim, n_qubits),
            nn.LayerNorm(n_qubits),  # Normalize for quantum encoding
            nn.Tanh(),  # Bound to [-1, 1] for angle encoding
        )
        
        if use_quantum:
            # --- Quantum Layer ---
            self.quantum_layer = VariationalQuantumLayer(
                n_qubits=n_qubits,
                n_layers=quantum_layers,
                diff_method='parameter-shift',
            )
            
            # --- Classifier Head ---
            # Takes quantum expectation values as input
            self.classifier = nn.Sequential(
                nn.Linear(n_qubits, n_qubits * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(n_qubits * 2, num_classes),
            )
        else:
            # Ablation: skip quantum layer
            self.quantum_layer = None
            self.classifier = nn.Sequential(
                nn.Linear(n_qubits, n_qubits * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(n_qubits * 2, num_classes),
            )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
        
        Returns:
            Class logits [batch_size, num_classes]
        """
        # Store original device
        device = x.device
        
        # 1. GNN encoding: graph → embedding
        embedding = self.gnn_encoder(x, edge_index, batch)  # [B, 32]
        
        # 2. Compression: 32 → 8
        compressed = self.compression(embedding)  # [B, 8]
        
        if self.use_quantum and self.quantum_layer is not None:
            # 3. Quantum processing (must be on CPU)
            compressed_cpu = compressed.cpu()
            quantum_out = self.quantum_layer(compressed_cpu)  # [B, 8]
            
            # Move back to original device
            quantum_out = quantum_out.to(device)
        else:
            quantum_out = compressed
        
        # 4. Classification
        logits = self.classifier(quantum_out)  # [B, num_classes]
        
        return logits
    
    def forward_batch(self, data: Batch) -> torch.Tensor:
        """
        Convenience method for PyG Batch objects.
        
        Args:
            data: PyTorch Geometric Batch
        
        Returns:
            Class logits [batch_size, num_classes]
        """
        return self.forward(data.x, data.edge_index, data.batch)
    
    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        layer: str = 'gnn',
    ) -> torch.Tensor:
        """
        Extract intermediate embeddings for visualization.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment
            layer: Which embedding to return ('gnn', 'compressed', 'quantum')
        
        Returns:
            Embeddings at specified layer
        """
        device = x.device
        
        embedding = self.gnn_encoder(x, edge_index, batch)
        if layer == 'gnn':
            return embedding
        
        compressed = self.compression(embedding)
        if layer == 'compressed':
            return compressed
        
        if self.use_quantum and self.quantum_layer is not None:
            compressed_cpu = compressed.cpu()
            quantum_out = self.quantum_layer(compressed_cpu)
            return quantum_out.to(device)
        
        return compressed
    
    def get_embeddings_batch(self, data: Batch, layer: str = 'gnn') -> torch.Tensor:
        """Convenience method for batch embedding extraction."""
        return self.get_embeddings(data.x, data.edge_index, data.batch, layer)
    
    def __repr__(self) -> str:
        quantum_info = f"quantum_layers={self.quantum_layer.n_layers}" if self.use_quantum else "quantum=disabled"
        return (
            f"HybridQMolNet(\n"
            f"  node_features={self.node_feature_dim},\n"
            f"  gnn_embedding={self.gnn_embedding_dim},\n"
            f"  n_qubits={self.n_qubits},\n"
            f"  {quantum_info},\n"
            f"  num_classes={self.num_classes}\n"
            f")"
        )


def print_hybrid_model_summary(model: HybridQMolNet) -> None:
    """
    Print a comprehensive summary of the hybrid model.
    
    Args:
        model: HybridQMolNet instance
    """
    print("\n" + "="*70)
    print("Hybrid QMolNet Architecture Summary")
    print("="*70)
    
    # Count parameters by component
    gnn_params = sum(p.numel() for p in model.gnn_encoder.parameters())
    compression_params = sum(p.numel() for p in model.compression.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    
    if model.use_quantum:
        quantum_params = sum(p.numel() for p in model.quantum_layer.parameters())
    else:
        quantum_params = 0
    
    total_params = gnn_params + compression_params + quantum_params + classifier_params
    
    print("\n[Pipeline Overview]")
    print("+-----------------------------------------------------------------+")
    print("|  SMILES -> Graph -> GNN -> Compress -> [Quantum] -> Classifier  |")
    print("|                    |         |          |            |          |")
    print(f"|             {model.node_feature_dim:>5}D -> {model.gnn_embedding_dim:>3}D ->    {model.n_qubits}D  ->      {model.n_qubits}D  ->       {model.num_classes}    |")
    print("+-----------------------------------------------------------------+")
    
    print("\n[Component Details]")
    print("-"*70)
    
    print(f"\n1. GNN Encoder ({gnn_params:,} params)")
    print(f"   Input: {model.node_feature_dim} node features")
    print(f"   Output: {model.gnn_embedding_dim}-dim graph embedding")
    print(f"   Layers: {model.gnn_encoder.num_layers} GCN layers")
    
    print(f"\n2. Compression Layer ({compression_params:,} params)")
    print(f"   Linear: {model.gnn_embedding_dim} -> {model.n_qubits}")
    print(f"   LayerNorm + Tanh activation")
    
    if model.use_quantum:
        print(f"\n3. Quantum Layer ({quantum_params:,} params)")
        print(f"   Qubits: {model.n_qubits}")
        print(f"   Variational layers: {model.quantum_layer.n_layers}")
        print(f"   Encoding: RY angle encoding")
        print(f"   Entanglement: CNOT ring")
        print(f"   Measurement: Pauli-Z expectations")
    else:
        print("\n3. Quantum Layer: DISABLED (ablation mode)")
    
    print(f"\n4. Classifier Head ({classifier_params:,} params)")
    print(f"   MLP: {model.n_qubits} -> {model.n_qubits * 2} -> {model.num_classes}")
    
    print("\n" + "-"*70)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"  Classical: {gnn_params + compression_params + classifier_params:,}")
    print(f"  Quantum:   {quantum_params:,}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Demo and testing
    from torch_geometric.data import Data, Batch
    
    # Create dummy molecular graphs
    num_nodes = 15
    node_feature_dim = 145
    
    x = torch.randn(num_nodes, node_feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, 30))
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([1]))
    
    # Create batch
    batch = Batch.from_data_list([data] * 4)
    
    # Initialize model
    print("Initializing Hybrid QMolNet...")
    model = HybridQMolNet(
        node_feature_dim=node_feature_dim,
        gnn_hidden_dim=64,
        gnn_embedding_dim=32,
        gnn_layers=3,
        n_qubits=8,
        quantum_layers=3,
        num_classes=2,
    )
    
    # Print summary
    print_hybrid_model_summary(model)
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        logits = model.forward_batch(batch)
    
    print(f"Input: {batch.num_graphs} graphs")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output: {logits}")
    
    # Test embedding extraction
    print("\nTesting embedding extraction...")
    with torch.no_grad():
        gnn_emb = model.get_embeddings_batch(batch, layer='gnn')
        comp_emb = model.get_embeddings_batch(batch, layer='compressed')
        quant_emb = model.get_embeddings_batch(batch, layer='quantum')
    
    print(f"GNN embeddings: {gnn_emb.shape}")
    print(f"Compressed embeddings: {comp_emb.shape}")
    print(f"Quantum embeddings: {quant_emb.shape}")
