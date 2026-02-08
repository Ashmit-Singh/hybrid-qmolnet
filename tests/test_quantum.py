"""
Unit Tests for Quantum Layer

Tests the variational quantum circuit implementation.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestQuantumLayer:
    """Tests for Variational Quantum Layer."""
    
    def test_layer_initialization(self):
        """Test quantum layer initializes correctly."""
        from models.quantum_layer import VariationalQuantumLayer
        
        layer = VariationalQuantumLayer(
            n_qubits=8,
            n_layers=3,
        )
        
        assert layer is not None
        assert layer.n_qubits == 8
        assert layer.n_layers == 3
        assert layer.weights.shape == (3, 8, 3)
    
    def test_layer_forward_single(self):
        """Test forward pass for single sample."""
        from models.quantum_layer import VariationalQuantumLayer
        
        layer = VariationalQuantumLayer(n_qubits=4, n_layers=2)
        x = torch.randn(4)
        
        with torch.no_grad():
            output = layer.forward_single(x)
        
        assert output.shape == (4,)
        # Expectation values should be in [-1, 1]
        assert output.min() >= -1.0
        assert output.max() <= 1.0
    
    def test_layer_forward_batch(self):
        """Test forward pass for batch."""
        from models.quantum_layer import VariationalQuantumLayer
        
        layer = VariationalQuantumLayer(n_qubits=4, n_layers=2)
        x = torch.randn(3, 4)  # Batch of 3
        
        with torch.no_grad():
            output = layer(x)
        
        assert output.shape == (3, 4)
    
    def test_layer_gradients(self):
        """Test that gradients can be computed."""
        from models.quantum_layer import VariationalQuantumLayer
        
        layer = VariationalQuantumLayer(n_qubits=4, n_layers=2)
        x = torch.randn(2, 4, requires_grad=True)
        
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist and are not all zeros
        assert layer.weights.grad is not None
        assert layer.weights.grad.abs().sum() > 0
    
    def test_circuit_info(self):
        """Test circuit information method."""
        from models.quantum_layer import VariationalQuantumLayer
        
        layer = VariationalQuantumLayer(n_qubits=8, n_layers=3)
        info = layer.get_circuit_info()
        
        assert info['n_qubits'] == 8
        assert info['n_layers'] == 3
        assert info['n_parameters'] == 3 * 8 * 3  # layers * qubits * 3 rotations
    
    def test_output_range(self):
        """Test that outputs are valid expectation values."""
        from models.quantum_layer import VariationalQuantumLayer
        
        layer = VariationalQuantumLayer(n_qubits=8, n_layers=2)
        
        # Test with various inputs
        for _ in range(5):
            x = torch.randn(4, 8) * 2  # Random inputs
            
            with torch.no_grad():
                output = layer(x)
            
            # All outputs should be in [-1, 1]
            assert (output >= -1.0).all()
            assert (output <= 1.0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
