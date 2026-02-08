"""
Variational Quantum Circuit Layer Module

Implements a parameterized quantum circuit using PennyLane that can be
integrated with PyTorch for end-to-end gradient-based training.

The circuit uses:
- Angle encoding (RY) to embed classical features
- Entangling CNOT layers for quantum correlations
- Parameterized rotations (RX, RY, RZ) for expressibility
- Pauli-Z expectation measurements for output

Mathematical Foundation:
-----------------------
The quantum layer implements a variational ansatz:

1. Feature Encoding: |ψ_0⟩ → Π_i RY(x_i) |0⟩^n
   - Encodes n classical features into n qubit rotation angles
   
2. Variational Blocks: |ψ_k+1⟩ = U_ent · U_rot(θ_k) |ψ_k⟩
   - U_ent: Layer of CNOT gates for entanglement
   - U_rot: Parameterized single-qubit rotations RX, RY, RZ
   
3. Measurement: ⟨ψ_final| Z_i |ψ_final⟩ for each qubit
   - Returns expectation values in [-1, 1]

The circuit is differentiable via the parameter-shift rule:
∂f/∂θ = (1/2)[f(θ + π/2) - f(θ - π/2)]
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from typing import Tuple, Optional, List, Callable
import matplotlib.pyplot as plt


def create_quantum_circuit(
    n_qubits: int = 8,
    n_layers: int = 3,
    diff_method: str = 'parameter-shift',
) -> Tuple[qml.QNode, qml.Device]:
    """
    Create a variational quantum circuit for classification.
    
    The circuit architecture:
    1. RY angle encoding layer (features → qubit rotations)
    2. Multiple variational blocks:
       - Entangling CNOT ring
       - Parameterized RX, RY, RZ rotations on each qubit
    3. Pauli-Z measurements on all qubits
    
    Args:
        n_qubits: Number of qubits (should match compressed embedding size)
        n_layers: Number of variational layers (blocks of rotations + entanglement)
        diff_method: Differentiation method ('parameter-shift' or 'backprop')
    
    Returns:
        Tuple of (quantum_node, device)
    """
    # Create quantum device (simulator)
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev, interface='torch', diff_method=diff_method)
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Execute the variational quantum circuit.
        
        Args:
            inputs: Feature vector [n_qubits] to encode
            weights: Variational parameters [n_layers, n_qubits, 3]
        
        Returns:
            Expectation values [n_qubits]
        """
        # --- Feature Encoding Layer ---
        # Apply RY rotations to encode classical features as quantum amplitudes
        # RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        # Features are scaled by π to span the full Bloch sphere
        for i in range(n_qubits):
            qml.RY(inputs[i] * np.pi, wires=i)
        
        # --- Variational Layers ---
        for layer_idx in range(n_layers):
            # Entanglement layer: CNOT ring connecting adjacent qubits
            # Creates quantum correlations between qubits
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            
            # Parameterized rotation layer
            # RX, RY, RZ rotations provide expressibility
            # Each qubit gets 3 independent rotation parameters
            for i in range(n_qubits):
                qml.RX(weights[layer_idx, i, 0], wires=i)
                qml.RY(weights[layer_idx, i, 1], wires=i)
                qml.RZ(weights[layer_idx, i, 2], wires=i)
        
        # --- Measurement ---
        # Return expectation value of Pauli-Z on each qubit
        # ⟨Z⟩ ∈ [-1, 1] representing quantum measurement outcome
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return circuit, dev


class VariationalQuantumLayer(nn.Module):
    """
    PyTorch module wrapping a variational quantum circuit.
    
    This layer:
    - Accepts batched classical embeddings
    - Encodes features using angle embedding
    - Applies variational quantum operations
    - Returns quantum expectation values
    
    Designed to be inserted between classical neural network layers
    for hybrid quantum-classical learning.
    """
    
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        diff_method: str = 'parameter-shift',
    ):
        """
        Initialize the variational quantum layer.
        
        Args:
            n_qubits: Number of qubits (input/output dimension)
            n_layers: Number of variational blocks
            diff_method: PennyLane differentiation method
        """
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.diff_method = diff_method
        
        # Create quantum circuit
        self.circuit, self.device = create_quantum_circuit(
            n_qubits, n_layers, diff_method
        )
        
        # Initialize variational parameters
        # Shape: [n_layers, n_qubits, 3] for RX, RY, RZ per qubit per layer
        # Initialize with small random values near zero
        weight_shape = (n_layers, n_qubits, 3)
        self.weights = nn.Parameter(
            torch.randn(weight_shape) * 0.1,
            requires_grad=True
        )
        
        # Count parameters
        self.n_params = self.weights.numel()
    
    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a single sample through the quantum circuit.
        
        Args:
            x: Input features [n_qubits]
        
        Returns:
            Quantum expectation values [n_qubits]
        """
        # Normalize input to reasonable range for angle encoding
        x_normalized = torch.tanh(x)  # Map to [-1, 1]
        
        # Run circuit
        expectations = self.circuit(x_normalized, self.weights)
        
        # PennyLane 0.28+ may return a tensor directly
        if isinstance(expectations, torch.Tensor):
            return expectations.float()
            
        # Stack results into tensor
        return torch.stack(expectations).float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a batch of samples through the quantum circuit.
        
        Note: Quantum circuits are executed sequentially per sample.
        This is a fundamental limitation of current quantum hardware/simulators.
        
        Args:
            x: Batched input features [batch_size, n_qubits]
        
        Returns:
            Quantum expectation values [batch_size, n_qubits]
        """
        batch_size = x.shape[0]
        outputs = []
        
        for i in range(batch_size):
            out = self.forward_single(x[i])
            outputs.append(out)
        
        return torch.stack(outputs)
    
    def get_circuit_info(self) -> dict:
        """
        Get information about the quantum circuit.
        
        Returns:
            Dictionary with circuit specifications
        """
        return {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'n_parameters': self.n_params,
            'diff_method': self.diff_method,
            'gate_count': {
                'RY_encoding': self.n_qubits,
                'CNOT_per_layer': self.n_qubits,
                'RX_per_layer': self.n_qubits,
                'RY_per_layer': self.n_qubits,
                'RZ_per_layer': self.n_qubits,
                'total_gates': self.n_qubits + self.n_layers * (4 * self.n_qubits),
            }
        }
    
    def __repr__(self) -> str:
        return (
            f"VariationalQuantumLayer(\n"
            f"  n_qubits={self.n_qubits},\n"
            f"  n_layers={self.n_layers},\n"
            f"  n_parameters={self.n_params},\n"
            f"  diff_method='{self.diff_method}'\n"
            f")"
        )


def draw_quantum_circuit(
    n_qubits: int = 8,
    n_layers: int = 3,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
) -> plt.Figure:
    """
    Draw and visualize the quantum circuit architecture.
    
    Creates a visual representation of the variational quantum circuit
    showing the encoding and variational layers.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        save_path: Optional path to save the figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    # Create a minimal circuit for visualization
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def visualization_circuit(inputs, weights):
        # Encoding layer
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # Variational layers
        for layer_idx in range(n_layers):
            # Entanglement
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            
            # Rotations
            for i in range(n_qubits):
                qml.RX(weights[layer_idx, i, 0], wires=i)
                qml.RY(weights[layer_idx, i, 1], wires=i)
                qml.RZ(weights[layer_idx, i, 2], wires=i)
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # Create dummy inputs
    inputs = np.zeros(n_qubits)
    weights = np.zeros((n_layers, n_qubits, 3))
    
    # Draw the circuit
    fig, ax = qml.draw_mpl(visualization_circuit)(inputs, weights)
    fig.set_size_inches(figsize)
    ax.set_title(f"Variational Quantum Circuit ({n_qubits} qubits, {n_layers} layers)", fontsize=14)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Circuit diagram saved to {save_path}")
    
    return fig


def print_quantum_layer_summary(layer: VariationalQuantumLayer) -> None:
    """
    Print a detailed summary of the quantum layer.
    
    Args:
        layer: VariationalQuantumLayer instance
    """
    info = layer.get_circuit_info()
    
    print("\n" + "="*60)
    print("Variational Quantum Layer Summary")
    print("="*60)
    print(f"\nQuantum System:")
    print(f"  Number of qubits:       {info['n_qubits']}")
    print(f"  Variational layers:     {info['n_layers']}")
    print(f"  Trainable parameters:   {info['n_parameters']}")
    print(f"  Differentiation method: {info['diff_method']}")
    
    print(f"\nGate Counts:")
    gates = info['gate_count']
    print(f"  RY encoding gates:      {gates['RY_encoding']}")
    print(f"  CNOT gates per layer:   {gates['CNOT_per_layer']}")
    print(f"  RX gates per layer:     {gates['RX_per_layer']}")
    print(f"  RY gates per layer:     {gates['RY_per_layer']}")
    print(f"  RZ gates per layer:     {gates['RZ_per_layer']}")
    print(f"  Total gates:            {gates['total_gates']}")
    
    print(f"\nCircuit Depth Analysis:")
    depth_per_layer = 1 + 3  # CNOT ring + 3 rotations
    total_depth = 1 + info['n_layers'] * depth_per_layer  # encoding + layers
    print(f"  Encoding depth:         1")
    print(f"  Depth per var. layer:   {depth_per_layer}")
    print(f"  Total circuit depth:    ~{total_depth}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Demo and testing
    print("Creating Variational Quantum Layer...")
    
    # Initialize layer
    vql = VariationalQuantumLayer(
        n_qubits=8,
        n_layers=3,
        diff_method='parameter-shift'
    )
    
    # Print summary
    print_quantum_layer_summary(vql)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 8)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = vql(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Expected range: [-1, 1] (Pauli-Z expectation values)\n")
    
    # Test gradient computation
    print("Testing gradient computation...")
    x_grad = torch.randn(2, 8, requires_grad=True)
    output = vql(x_grad)
    loss = output.sum()
    loss.backward()
    
    print(f"Gradient computation successful!")
    print(f"Weight gradient shape: {vql.weights.grad.shape}")
    print(f"Weight gradient norm: {vql.weights.grad.norm():.4f}")
