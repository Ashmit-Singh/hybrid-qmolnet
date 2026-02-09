"""
Optimized Variational Quantum Circuit Layer Module

PERFORMANCE OPTIMIZATIONS:
1. QNode defined once at module init (not per forward)
2. Device auto-selection (lightning.qubit preferred)
3. Pre-scaled encoding (π multiplication outside circuit)
4. Analytic mode (shots=None)
5. Timing profiler hooks
6. Micro-batching wrapper
7. Fast profile mode (reduced qubits/layers)
8. Cached tape execution where possible

The circuit uses:
- Angle encoding (RY) to embed classical features
- Entangling CNOT layers for quantum correlations
- Parameterized rotations (RX, RY, RZ) for expressibility
- Pauli-Z expectation measurements for output

Gradients computed via parameter-shift rule:
∂f/∂θ = (1/2)[f(θ + π/2) - f(θ - π/2)]
"""

import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from typing import Tuple, Optional, Dict, Any
from functools import lru_cache


# =============================================================================
# DEVICE AUTO-SELECTION
# =============================================================================

def get_best_device(n_qubits: int) -> Tuple[qml.Device, str]:
    """
    Auto-select the fastest available PennyLane device.
    
    Priority order:
    1. lightning.gpu (if available and beneficial)
    2. lightning.qubit (CPU optimized, ~2-5x faster)
    3. default.qubit (fallback)
    
    Args:
        n_qubits: Number of qubits needed
        
    Returns:
        Tuple of (device, device_name)
    """
    # Try lightning.gpu first (CUDA acceleration)
    try:
        import pennylane_lightning
        dev = qml.device('lightning.gpu', wires=n_qubits)
        return dev, 'lightning.gpu'
    except:
        pass
    
    # Try lightning.qubit (CPU optimized C++ backend)
    try:
        dev = qml.device('lightning.qubit', wires=n_qubits)
        return dev, 'lightning.qubit'
    except:
        pass
    
    # Fallback to default.qubit
    dev = qml.device('default.qubit', wires=n_qubits)
    return dev, 'default.qubit'


# =============================================================================
# TIMING PROFILER
# =============================================================================

class QuantumProfiler:
    """Track quantum layer execution times."""
    
    def __init__(self):
        self.forward_times = []
        self.backward_times = []
        self.batch_sizes = []
        self._forward_start = None
        self._backward_start = None
        
    def start_forward(self):
        self._forward_start = time.perf_counter()
        
    def end_forward(self, batch_size: int = 1):
        if self._forward_start:
            elapsed = time.perf_counter() - self._forward_start
            self.forward_times.append(elapsed)
            self.batch_sizes.append(batch_size)
            self._forward_start = None
            
    def start_backward(self):
        self._backward_start = time.perf_counter()
        
    def end_backward(self):
        if self._backward_start:
            elapsed = time.perf_counter() - self._backward_start
            self.backward_times.append(elapsed)
            self._backward_start = None
    
    def get_stats(self) -> Dict[str, float]:
        """Get timing statistics."""
        stats = {}
        if self.forward_times:
            stats['forward_mean_ms'] = np.mean(self.forward_times) * 1000
            stats['forward_total_s'] = np.sum(self.forward_times)
            stats['samples_processed'] = sum(self.batch_sizes)
            stats['ms_per_sample'] = (stats['forward_total_s'] * 1000) / max(1, stats['samples_processed'])
        if self.backward_times:
            stats['backward_mean_ms'] = np.mean(self.backward_times) * 1000
            stats['backward_total_s'] = np.sum(self.backward_times)
        return stats
    
    def reset(self):
        self.forward_times = []
        self.backward_times = []
        self.batch_sizes = []
        
    def summary(self) -> str:
        stats = self.get_stats()
        if not stats:
            return "No timing data collected"
        lines = ["Quantum Layer Timing:"]
        for k, v in stats.items():
            lines.append(f"  {k}: {v:.2f}")
        return "\n".join(lines)


# Global profiler instance
_profiler = QuantumProfiler()

def get_profiler() -> QuantumProfiler:
    return _profiler


# =============================================================================
# OPTIMIZED QUANTUM LAYER
# =============================================================================

class VariationalQuantumLayer(nn.Module):
    """
    Optimized PyTorch module wrapping a variational quantum circuit.
    
    Performance Features:
    - QNode created once at init
    - Device auto-selection
    - Pre-scaled angle encoding
    - Analytic expectation mode
    - Timing hooks
    - Optional fast profile mode
    """
    
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        diff_method: str = 'parameter-shift',
        fast_mode: bool = False,
        device_name: Optional[str] = None,
        enable_profiling: bool = True,
    ):
        """
        Initialize the optimized variational quantum layer.
        
        Args:
            n_qubits: Number of qubits (input/output dimension)
            n_layers: Number of variational blocks
            diff_method: PennyLane differentiation method
            fast_mode: If True, use reduced config (6 qubits, 2 layers)
            device_name: Force specific device (None for auto-select)
            enable_profiling: Track execution times
        """
        super().__init__()
        
        # Fast mode reduces circuit complexity
        if fast_mode:
            n_qubits = min(n_qubits, 6)
            n_layers = min(n_layers, 2)
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.diff_method = diff_method
        self.fast_mode = fast_mode
        self.enable_profiling = enable_profiling
        
        # Pre-compute pi constant as tensor
        self.register_buffer('_pi', torch.tensor(np.pi, dtype=torch.float32))
        
        # Auto-select or use specified device
        if device_name:
            self.device = qml.device(device_name, wires=n_qubits)
            self.device_name = device_name
        else:
            self.device, self.device_name = get_best_device(n_qubits)
        
        # Create quantum circuit ONCE at init
        self.circuit = self._build_circuit()
        
        # Initialize variational parameters
        # Shape: [n_layers, n_qubits, 3] for RX, RY, RZ per qubit per layer
        weight_shape = (n_layers, n_qubits, 3)
        self.weights = nn.Parameter(
            torch.randn(weight_shape, dtype=torch.float32) * 0.1,
            requires_grad=True
        )
        
        self.n_params = self.weights.numel()
        
    def _build_circuit(self) -> qml.QNode:
        """
        Build the quantum circuit once.
        
        The circuit is defined here and reused for all forward passes.
        This avoids the overhead of rebuilding the computational graph.
        """
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        
        @qml.qnode(
            self.device, 
            interface='torch', 
            diff_method=self.diff_method,
            # Caching optimization
            cache=True if self.diff_method == 'parameter-shift' else False,
        )
        def circuit(encoded_inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            """
            Execute the variational quantum circuit.
            
            Args:
                encoded_inputs: PRE-SCALED feature vector [n_qubits] (already multiplied by π)
                weights: Variational parameters [n_layers, n_qubits, 3]
            
            Returns:
                Expectation values [n_qubits]
            """
            # --- Feature Encoding Layer ---
            # Inputs are PRE-SCALED by π outside the circuit for efficiency
            for i in range(n_qubits):
                qml.RY(encoded_inputs[i], wires=i)
            
            # --- Variational Layers ---
            for layer_idx in range(n_layers):
                # Entanglement layer: CNOT ring
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
                
                # Parameterized rotation layer
                for i in range(n_qubits):
                    qml.RX(weights[layer_idx, i, 0], wires=i)
                    qml.RY(weights[layer_idx, i, 1], wires=i)
                    qml.RZ(weights[layer_idx, i, 2], wires=i)
            
            # --- Measurement ---
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        return circuit
    
    def _encode_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pre-encode input: normalize and scale by π.
        
        This is done OUTSIDE the circuit for efficiency.
        """
        # Normalize to [-1, 1] then scale by π
        x_normalized = torch.tanh(x)
        return x_normalized * self._pi
    
    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a single sample through the quantum circuit.
        
        Args:
            x: Input features [n_qubits]
        
        Returns:
            Quantum expectation values [n_qubits]
        """
        # Pre-encode (π scaling done here, not in circuit)
        encoded = self._encode_input(x)
        
        # Run circuit
        expectations = self.circuit(encoded, self.weights)
        
        # Handle output format
        if isinstance(expectations, torch.Tensor):
            return expectations.float()
        return torch.stack(expectations).float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a batch of samples through the quantum circuit.
        
        Args:
            x: Batched input features [batch_size, n_qubits]
        
        Returns:
            Quantum expectation values [batch_size, n_qubits]
        """
        if self.enable_profiling:
            _profiler.start_forward()
        
        batch_size = x.shape[0]
        
        # Pre-encode all inputs at once (vectorized)
        encoded_batch = self._encode_input(x)
        
        # Process samples (sequential - fundamental limitation)
        outputs = []
        for i in range(batch_size):
            out = self.circuit(encoded_batch[i], self.weights)
            if isinstance(out, torch.Tensor):
                outputs.append(out.float())
            else:
                outputs.append(torch.stack(out).float())
        
        result = torch.stack(outputs)
        
        if self.enable_profiling:
            _profiler.end_forward(batch_size)
        
        return result
    
    def get_circuit_info(self) -> Dict[str, Any]:
        """Get information about the quantum circuit."""
        return {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'n_parameters': self.n_params,
            'diff_method': self.diff_method,
            'device': self.device_name,
            'fast_mode': self.fast_mode,
            'gate_count': {
                'RY_encoding': self.n_qubits,
                'CNOT_per_layer': self.n_qubits,
                'rotation_gates_per_layer': self.n_qubits * 3,
                'total_gates': self.n_qubits + self.n_layers * (4 * self.n_qubits),
            },
            'gradient_evals_per_step': 2 * self.n_params,  # Parameter-shift cost
        }
    
    def __repr__(self) -> str:
        return (
            f"VariationalQuantumLayer(\n"
            f"  n_qubits={self.n_qubits},\n"
            f"  n_layers={self.n_layers},\n"
            f"  n_parameters={self.n_params},\n"
            f"  device='{self.device_name}',\n"
            f"  fast_mode={self.fast_mode}\n"
            f")"
        )


# =============================================================================
# MICRO-BATCH WRAPPER (for memory efficiency)
# =============================================================================

class MicroBatchQuantumLayer(nn.Module):
    """
    Wrapper that processes large batches in smaller micro-batches.
    
    Useful for:
    - Reducing peak memory usage
    - More consistent execution times
    - Better debugging
    """
    
    def __init__(
        self,
        quantum_layer: VariationalQuantumLayer,
        micro_batch_size: int = 8,
    ):
        super().__init__()
        self.quantum_layer = quantum_layer
        self.micro_batch_size = micro_batch_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        if batch_size <= self.micro_batch_size:
            return self.quantum_layer(x)
        
        # Process in chunks
        outputs = []
        for i in range(0, batch_size, self.micro_batch_size):
            chunk = x[i:i + self.micro_batch_size]
            out = self.quantum_layer(chunk)
            outputs.append(out)
        
        return torch.cat(outputs, dim=0)


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def create_quantum_circuit(
    n_qubits: int = 8,
    n_layers: int = 3,
    diff_method: str = 'parameter-shift',
) -> Tuple[qml.QNode, qml.Device]:
    """
    Legacy function for backward compatibility.
    
    Prefer using VariationalQuantumLayer directly.
    """
    warnings.warn(
        "create_quantum_circuit is deprecated. Use VariationalQuantumLayer instead.",
        DeprecationWarning
    )
    
    dev, _ = get_best_device(n_qubits)
    
    @qml.qnode(dev, interface='torch', diff_method=diff_method)
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        for i in range(n_qubits):
            qml.RY(inputs[i] * np.pi, wires=i)
        
        for layer_idx in range(n_layers):
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            
            for i in range(n_qubits):
                qml.RX(weights[layer_idx, i, 0], wires=i)
                qml.RY(weights[layer_idx, i, 1], wires=i)
                qml.RZ(weights[layer_idx, i, 2], wires=i)
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return circuit, dev


# =============================================================================
# DRAWING UTILITIES
# =============================================================================

def draw_quantum_circuit(
    n_qubits: int = 8,
    n_layers: int = 3,
    save_path: Optional[str] = None,
) -> str:
    """
    Generate ASCII circuit diagram.
    
    Returns text representation for console output.
    """
    try:
        from visualization.circuit_viz import get_circuit_ascii
        return get_circuit_ascii(n_qubits, n_layers)
    except ImportError:
        return f"Circuit: {n_qubits} qubits, {n_layers} layers"


def print_quantum_layer_summary(layer: VariationalQuantumLayer) -> None:
    """Print a detailed summary of the quantum layer."""
    info = layer.get_circuit_info()
    
    print("\n" + "="*60)
    print("Optimized Variational Quantum Layer")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Qubits:                 {info['n_qubits']}")
    print(f"  Variational layers:     {info['n_layers']}")
    print(f"  Trainable parameters:   {info['n_parameters']}")
    print(f"  Device:                 {info['device']}")
    print(f"  Fast mode:              {info['fast_mode']}")
    print(f"  Diff method:            {info['diff_method']}")
    print(f"\nGate count:               {info['gate_count']['total_gates']}")
    print(f"Gradient evals/step:      {info['gradient_evals_per_step']}")
    print("="*60 + "\n")


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def validate_layer(layer: VariationalQuantumLayer, tolerance: float = 1e-4) -> bool:
    """
    Validate that the layer produces correct outputs and gradients.
    
    Returns True if all tests pass.
    """
    print("Running validation tests...")
    
    # Test 1: Forward pass output shape
    x = torch.randn(2, layer.n_qubits)
    out = layer(x)
    assert out.shape == (2, layer.n_qubits), f"Shape mismatch: {out.shape}"
    print("  ✓ Output shape correct")
    
    # Test 2: Output range (expectation values in [-1, 1])
    assert out.min() >= -1.0 - tolerance, f"Output below -1: {out.min()}"
    assert out.max() <= 1.0 + tolerance, f"Output above 1: {out.max()}"
    print("  ✓ Output range correct [-1, 1]")
    
    # Test 3: Gradients exist and are non-zero
    x.requires_grad = True
    out = layer(x)
    loss = out.sum()
    loss.backward()
    
    assert layer.weights.grad is not None, "No gradient computed"
    assert layer.weights.grad.abs().sum() > 0, "Gradient is all zeros"
    print("  ✓ Gradients computed and non-zero")
    
    # Test 4: Deterministic output (same input = same output)
    layer.eval()
    with torch.no_grad():
        x_test = torch.randn(1, layer.n_qubits)
        out1 = layer(x_test)
        out2 = layer(x_test)
        diff = (out1 - out2).abs().max()
        assert diff < tolerance, f"Non-deterministic output: {diff}"
    print("  ✓ Deterministic outputs")
    
    print("All validation tests passed!")
    return True


# =============================================================================
# MAIN / DEMO
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("Optimized Quantum Layer Benchmark")
    print("="*60)
    
    # Parse fast mode flag
    fast_mode = '--fast_quantum' in sys.argv or '--fast' in sys.argv
    
    # Create layer
    layer = VariationalQuantumLayer(
        n_qubits=8,
        n_layers=3,
        fast_mode=fast_mode,
        enable_profiling=True,
    )
    
    print_quantum_layer_summary(layer)
    
    # Validate
    validate_layer(layer)
    
    # Benchmark
    print("\nBenchmarking forward pass...")
    batch_sizes = [1, 4, 8]
    
    for bs in batch_sizes:
        x = torch.randn(bs, layer.n_qubits)
        
        # Warmup
        _ = layer(x)
        
        # Timed run
        get_profiler().reset()
        start = time.perf_counter()
        for _ in range(3):
            _ = layer(x)
        elapsed = (time.perf_counter() - start) / 3
        
        print(f"  Batch size {bs}: {elapsed*1000:.1f} ms ({elapsed*1000/bs:.1f} ms/sample)")
    
    print("\n" + get_profiler().summary())
