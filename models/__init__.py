# Models package initialization
"""
Neural network models for Hybrid QMolNet

- GNN Encoder: Message-passing graph neural network
- Quantum Layer: Variational quantum circuit
- Hybrid Model: Combined classical-quantum architecture
- Baselines: Classical comparison models
"""

from .gnn_encoder import GNNEncoder
from .quantum_layer import VariationalQuantumLayer, draw_quantum_circuit
from .hybrid_model import HybridQMolNet
from .baselines import GNNClassifier, DescriptorMLP

__all__ = [
    'GNNEncoder',
    'VariationalQuantumLayer',
    'draw_quantum_circuit',
    'HybridQMolNet',
    'GNNClassifier',
    'DescriptorMLP',
]
