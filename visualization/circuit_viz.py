"""
Quantum Circuit Visualization Module

Provides functions to visualize and explain the variational quantum circuit
used in the hybrid QMolNet architecture.
"""

import numpy as np
from typing import Optional, Tuple, List


def get_circuit_ascii(n_qubits: int = 8, n_layers: int = 3) -> str:
    """
    Generate ASCII representation of the variational quantum circuit.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
    
    Returns:
        ASCII art string of the circuit
    """
    lines = []
    lines.append("=" * 70)
    lines.append("VARIATIONAL QUANTUM CIRCUIT ARCHITECTURE")
    lines.append(f"Qubits: {n_qubits} | Variational Layers: {n_layers}")
    lines.append("=" * 70)
    lines.append("")
    
    # Encoding layer
    lines.append("┌─────────────────────────────────────────────────────────────────┐")
    lines.append("│                     ANGLE ENCODING LAYER                        │")
    lines.append("├─────────────────────────────────────────────────────────────────┤")
    
    for q in range(min(n_qubits, 4)):
        lines.append(f"│  q[{q}]: ─────[RY(x_{q})]─────────────────────────────────────── │")
    
    if n_qubits > 4:
        lines.append(f"│  ...  ({n_qubits - 4} more qubits)                                        │")
    
    lines.append("└─────────────────────────────────────────────────────────────────┘")
    lines.append("                              │")
    lines.append("                              ▼")
    
    # Variational layers
    for layer in range(min(n_layers, 2)):
        lines.append(f"┌─────────────────────────────────────────────────────────────────┐")
        lines.append(f"│                   VARIATIONAL LAYER {layer + 1}                           │")
        lines.append("├─────────────────────────────────────────────────────────────────┤")
        lines.append("│  Entanglement Block:                                           │")
        lines.append("│    q[0]: ───●───────────────────────────────────────           │")
        lines.append("│            │                                                   │")
        lines.append("│    q[1]: ──⊕──●────────────────────────────────────            │")
        lines.append("│               │                                                │")
        lines.append("│    q[2]: ─────⊕──●─────────────────────────────────            │")
        lines.append("│                  │         ... (ring topology)                 │")
        lines.append("│    q[n]: ────────⊕─────────────────────────────────            │")
        lines.append("│                                                                │")
        lines.append("│  Rotation Block:                                               │")
        lines.append("│    All qubits: ─[RX(θ)]─[RY(θ)]─[RZ(θ)]─                       │")
        lines.append("└─────────────────────────────────────────────────────────────────┘")
        if layer < n_layers - 1:
            lines.append("                              │")
            lines.append("                              ▼")
    
    if n_layers > 2:
        lines.append("                              │")
        lines.append(f"                    ... ({n_layers - 2} more layers)")
        lines.append("                              │")
    
    lines.append("                              ▼")
    lines.append("┌─────────────────────────────────────────────────────────────────┐")
    lines.append("│                       MEASUREMENT                               │")
    lines.append("├─────────────────────────────────────────────────────────────────┤")
    lines.append("│  All qubits: ────────────────────────────────[⟨Z⟩]────○         │")
    lines.append("│                                                       │         │")
    lines.append("│  Output: Expectation values <Z_0>, <Z_1>, ..., <Z_n>  │         │")
    lines.append("└───────────────────────────────────────────────────────┼─────────┘")
    lines.append("                                                        │")
    lines.append("                                                        ▼")
    lines.append("                                              [Classical Output]")
    lines.append("")
    
    return "\n".join(lines)


def get_circuit_parameters_info(n_qubits: int = 8, n_layers: int = 3) -> dict:
    """
    Get information about circuit parameters.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
    
    Returns:
        Dictionary with parameter information
    """
    # Parameters per layer: 3 rotation gates (RX, RY, RZ) per qubit
    params_per_layer = n_qubits * 3
    total_params = n_layers * params_per_layer
    
    # Gates count
    encoding_gates = n_qubits  # RY gates
    entangling_gates_per_layer = n_qubits  # CNOT ring
    rotation_gates_per_layer = n_qubits * 3  # RX, RY, RZ per qubit
    
    total_gates = (
        encoding_gates + 
        n_layers * (entangling_gates_per_layer + rotation_gates_per_layer)
    )
    
    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "trainable_parameters": total_params,
        "encoding_gates": encoding_gates,
        "entangling_gates": n_layers * entangling_gates_per_layer,
        "rotation_gates": n_layers * rotation_gates_per_layer,
        "total_gates": total_gates,
        "circuit_depth_estimate": 1 + n_layers * 2,  # encoding + (entangle + rotate) * layers
        "gradient_method": "parameter-shift",
    }


def get_circuit_explanation() -> str:
    """
    Get technical explanation of the quantum circuit.
    
    Returns:
        Markdown-formatted explanation text
    """
    return """
## Variational Quantum Circuit (VQC) Architecture

The hybrid model uses an **8-qubit Variational Quantum Circuit** as a feature transformation layer.

### Circuit Design

#### 1. Angle Encoding Layer
Classical features (compressed GNN embeddings) are encoded into quantum states using **RY rotation gates**:

```
|0⟩ → RY(2 × arctan(x_i)) → encoded state
```

Each of the 8 features maps to one qubit's rotation angle.

#### 2. Variational Layers (×3)
Each layer consists of:

**Entanglement Block:**
- CNOT gates in a ring topology
- Creates quantum correlations between qubits
- Pattern: CNOT(0,1) → CNOT(1,2) → ... → CNOT(7,0)

**Rotation Block:**
- Three parameterized rotations per qubit: RX(θ), RY(θ), RZ(θ)
- Parameters are learned during training

#### 3. Measurement
- Pauli-Z expectation values measured on all qubits
- Outputs 8-dimensional quantum-transformed features
- Formula: ⟨ψ|Z_i|ψ⟩ for i = 0, 1, ..., 7

### Training Integration

The circuit parameters are trained end-to-end with the GNN using the **parameter-shift rule**:

```
∂f/∂θ = (1/2)[f(θ + π/2) - f(θ - π/2)]
```

This allows gradient-based optimization through the quantum layer.

### Hardware Compatibility

This circuit is designed for **NISQ (Noisy Intermediate-Scale Quantum)** compatibility:
- Shallow depth (≤7 layers)
- Local connectivity (ring topology)
- Standard gate set

**Note:** Current implementation runs on a classical simulator (PennyLane default.qubit).
"""


def print_circuit_summary(n_qubits: int = 8, n_layers: int = 3):
    """
    Print a formatted summary of the quantum circuit.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
    """
    info = get_circuit_parameters_info(n_qubits, n_layers)
    
    print("=" * 50)
    print("QUANTUM CIRCUIT SUMMARY")
    print("=" * 50)
    print(f"Qubits:                 {info['n_qubits']}")
    print(f"Variational Layers:     {info['n_layers']}")
    print(f"Trainable Parameters:   {info['trainable_parameters']}")
    print("-" * 50)
    print(f"Encoding Gates (RY):    {info['encoding_gates']}")
    print(f"Entangling Gates (CNOT):{info['entangling_gates']}")
    print(f"Rotation Gates:         {info['rotation_gates']}")
    print(f"Total Gates:            {info['total_gates']}")
    print("-" * 50)
    print(f"Est. Circuit Depth:     {info['circuit_depth_estimate']}")
    print(f"Gradient Method:        {info['gradient_method']}")
    print("=" * 50)


def draw_circuit_matplotlib(
    n_qubits: int = 8,
    n_layers: int = 3,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
):
    """
    Draw the quantum circuit using matplotlib.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Matplotlib not available for circuit drawing")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Layout parameters
    qubit_spacing = 0.8
    gate_width = 0.4
    gate_height = 0.5
    layer_spacing = 1.5
    
    n_display = min(n_qubits, 6)  # Display up to 6 qubits for clarity
    
    # Draw qubit lines
    total_width = 2 + (1 + n_layers) * layer_spacing + 2
    for i in range(n_display):
        y = -i * qubit_spacing
        ax.plot([0, total_width], [y, y], 'k-', linewidth=1, alpha=0.5)
        ax.text(-0.3, y, f'q[{i}]', ha='right', va='center', fontsize=10)
    
    if n_qubits > n_display:
        ax.text(-0.3, -(n_display - 0.5) * qubit_spacing, 
                f'... ({n_qubits - n_display} more)', 
                ha='right', va='center', fontsize=8, style='italic')
    
    x_pos = 1
    
    # Encoding layer
    for i in range(n_display):
        y = -i * qubit_spacing
        rect = mpatches.FancyBboxPatch(
            (x_pos - gate_width/2, y - gate_height/2),
            gate_width, gate_height,
            boxstyle="round,pad=0.02",
            facecolor='lightblue',
            edgecolor='blue',
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x_pos, y, 'RY', ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.text(x_pos, 0.6, 'Encoding', ha='center', fontsize=10, fontweight='bold')
    x_pos += layer_spacing
    
    # Variational layers
    for layer in range(min(n_layers, 2)):
        # CNOT gates (simplified representation)
        for i in range(n_display - 1):
            y1 = -i * qubit_spacing
            y2 = -(i + 1) * qubit_spacing
            ax.plot([x_pos, x_pos], [y1, y2], 'k-', linewidth=2)
            ax.plot(x_pos, y1, 'ko', markersize=8)
            ax.plot(x_pos, y2, 'ko', markersize=12, fillstyle='none', markeredgewidth=2)
        
        x_pos += 0.5
        
        # Rotation gates
        for i in range(n_display):
            y = -i * qubit_spacing
            rect = mpatches.FancyBboxPatch(
                (x_pos - gate_width/2, y - gate_height/2),
                gate_width, gate_height,
                boxstyle="round,pad=0.02",
                facecolor='lightgreen',
                edgecolor='green',
                linewidth=1.5
            )
            ax.add_patch(rect)
            ax.text(x_pos, y, 'Rθ', ha='center', va='center', fontsize=9, fontweight='bold')
        
        ax.text(x_pos - 0.25, 0.6, f'Layer {layer + 1}', ha='center', fontsize=10, fontweight='bold')
        x_pos += layer_spacing
    
    if n_layers > 2:
        ax.text(x_pos - 0.5, -n_display * qubit_spacing / 2, 
                f'... ({n_layers - 2} more layers)', 
                ha='center', fontsize=10, style='italic')
        x_pos += layer_spacing / 2
    
    # Measurement
    for i in range(n_display):
        y = -i * qubit_spacing
        rect = mpatches.FancyBboxPatch(
            (x_pos - gate_width/2, y - gate_height/2),
            gate_width, gate_height,
            boxstyle="round,pad=0.02",
            facecolor='lightyellow',
            edgecolor='orange',
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x_pos, y, '⟨Z⟩', ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.text(x_pos, 0.6, 'Measure', ha='center', fontsize=10, fontweight='bold')
    
    # Title and styling
    ax.set_xlim(-1, total_width + 0.5)
    ax.set_ylim(-(n_display + 0.5) * qubit_spacing, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Variational Quantum Circuit ({n_qubits} qubits, {n_layers} layers)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', edgecolor='blue', label='Encoding (RY)'),
        mpatches.Patch(facecolor='lightgreen', edgecolor='green', label='Variational (Rθ)'),
        mpatches.Patch(facecolor='lightyellow', edgecolor='orange', label='Measurement (⟨Z⟩)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved circuit diagram to: {save_path}")
    
    return fig


if __name__ == "__main__":
    # Demo
    print(get_circuit_ascii())
    print()
    print_circuit_summary()
    print()
    print(get_circuit_explanation())
