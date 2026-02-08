"""
Unit Tests for Neural Network Models

Tests model initialization, forward passes, and output shapes.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Data, Batch


class TestGNNEncoder:
    """Tests for GNN Encoder."""
    
    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch of molecular graphs."""
        data_list = []
        for _ in range(4):
            x = torch.randn(10, 145)  # 10 atoms, 145 features
            edge_index = torch.randint(0, 10, (2, 20))
            data_list.append(Data(x=x, edge_index=edge_index))
        return Batch.from_data_list(data_list)
    
    def test_encoder_initialization(self):
        """Test GNN encoder initializes correctly."""
        from models.gnn_encoder import GNNEncoder
        
        encoder = GNNEncoder(
            input_dim=145,
            hidden_dim=64,
            embedding_dim=32,
            num_layers=3,
        )
        
        assert encoder is not None
        assert encoder.input_dim == 145
        assert encoder.embedding_dim == 32
    
    def test_encoder_forward_pass(self, sample_batch):
        """Test forward pass produces correct output shape."""
        from models.gnn_encoder import GNNEncoder
        
        encoder = GNNEncoder(
            input_dim=145,
            hidden_dim=64,
            embedding_dim=32,
            num_layers=3,
        )
        
        output = encoder.forward_batch(sample_batch)
        
        assert output.shape == (4, 32)  # 4 graphs, 32-dim embeddings
    
    def test_encoder_different_graph_sizes(self):
        """Test encoder handles graphs of different sizes."""
        from models.gnn_encoder import GNNEncoder
        
        encoder = GNNEncoder(input_dim=145, embedding_dim=32)
        
        # Create graphs with different numbers of nodes
        data_list = []
        for n_nodes in [5, 10, 15, 20]:
            x = torch.randn(n_nodes, 145)
            edge_index = torch.randint(0, n_nodes, (2, n_nodes * 2))
            data_list.append(Data(x=x, edge_index=edge_index))
        
        batch = Batch.from_data_list(data_list)
        output = encoder.forward_batch(batch)
        
        assert output.shape == (4, 32)


class TestHybridModel:
    """Tests for Hybrid QMolNet model."""
    
    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch of molecular graphs."""
        data_list = []
        for i in range(2):  # Small batch for quantum testing
            x = torch.randn(8, 145)
            edge_index = torch.randint(0, 8, (2, 15))
            y = torch.tensor([i % 2])
            data_list.append(Data(x=x, edge_index=edge_index, y=y))
        return Batch.from_data_list(data_list)
    
    def test_hybrid_model_initialization(self):
        """Test hybrid model initializes correctly."""
        from models.hybrid_model import HybridQMolNet
        
        model = HybridQMolNet(
            node_feature_dim=145,
            n_qubits=8,
            quantum_layers=2,
            num_classes=2,
        )
        
        assert model is not None
        assert model.n_qubits == 8
        assert model.num_classes == 2
        assert model.use_quantum == True
    
    def test_hybrid_model_forward_pass(self, sample_batch):
        """Test forward pass produces logits."""
        from models.hybrid_model import HybridQMolNet
        
        model = HybridQMolNet(
            node_feature_dim=145,
            n_qubits=8,
            quantum_layers=2,
            num_classes=2,
        )
        
        with torch.no_grad():
            logits = model.forward_batch(sample_batch)
        
        assert logits.shape == (2, 2)  # 2 graphs, 2 classes
    
    def test_hybrid_model_without_quantum(self, sample_batch):
        """Test model works without quantum layer."""
        from models.hybrid_model import HybridQMolNet
        
        model = HybridQMolNet(
            node_feature_dim=145,
            n_qubits=8,
            num_classes=2,
            use_quantum=False,
        )
        
        with torch.no_grad():
            logits = model.forward_batch(sample_batch)
        
        assert logits.shape == (2, 2)


class TestBaselineModels:
    """Tests for baseline models."""
    
    @pytest.fixture
    def sample_batch(self):
        """Create sample batch."""
        data_list = []
        for i in range(4):
            x = torch.randn(10, 145)
            edge_index = torch.randint(0, 10, (2, 20))
            data_list.append(Data(x=x, edge_index=edge_index))
        return Batch.from_data_list(data_list)
    
    def test_gnn_classifier(self, sample_batch):
        """Test GNN classifier baseline."""
        from models.baselines import GNNClassifier
        
        model = GNNClassifier(
            node_feature_dim=145,
            num_classes=2,
        )
        
        with torch.no_grad():
            logits = model.forward_batch(sample_batch)
        
        assert logits.shape == (4, 2)
    
    def test_descriptor_mlp(self):
        """Test descriptor MLP baseline."""
        from models.baselines import DescriptorMLP
        
        model = DescriptorMLP(
            input_dim=10,
            hidden_dims=(64, 32),
            num_classes=2,
        )
        
        x = torch.randn(16, 10)
        
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (16, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
