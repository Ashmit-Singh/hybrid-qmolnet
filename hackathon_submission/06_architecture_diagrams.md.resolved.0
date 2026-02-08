# Hybrid QMolNet — Architecture Diagrams

## ASCII Diagrams + Python Code for Visual Generation

---

## 1. Full Architecture Diagram (ASCII)

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                              HYBRID QMOLNET ARCHITECTURE                              ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                       ║
║  ┌─────────────┐     ┌──────────────────┐     ┌───────────────────────────────────┐  ║
║  │   SMILES    │     │   RDKit Parsing  │     │         MOLECULAR GRAPH           │  ║
║  │   String    │ ──▶ │   145-dim nodes  │ ──▶ │   Atoms = Nodes, Bonds = Edges    │  ║
║  │   "CCO"     │     │   6-dim edges    │     │   Bidirectional connectivity      │  ║
║  └─────────────┘     └──────────────────┘     └───────────────────────────────────┘  ║
║                                                              │                        ║
║                                                              ▼                        ║
║  ┌────────────────────────────────────────────────────────────────────────────────┐  ║
║  │                           GNN ENCODER (Classical GPU)                          │  ║
║  │  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐            │  ║
║  │  │Input Proj  │   │  GCN #1    │   │  GCN #2    │   │  GCN #3    │            │  ║
║  │  │145 → 64    │──▶│  64 → 64   │──▶│  64 → 64   │──▶│  64 → 64   │            │  ║
║  │  │+ ReLU      │   │  + BN+ReLU │   │  + BN+ReLU │   │  + BN+ReLU │            │  ║
║  │  └────────────┘   └────────────┘   └────────────┘   └────────────┘            │  ║
║  │                                                              │                 │  ║
║  │                                                              ▼                 │  ║
║  │                              ┌─────────────────────────────────────┐           │  ║
║  │                              │     GLOBAL MEAN POOLING            │           │  ║
║  │                              │     Aggregate node → graph         │           │  ║
║  │                              └─────────────────────────────────────┘           │  ║
║  │                                                              │                 │  ║
║  │                                                              ▼                 │  ║
║  │                              ┌─────────────────────────────────────┐           │  ║
║  │                              │     OUTPUT MLP: 64 → 64 → 32       │           │  ║
║  │                              │     + ReLU + Dropout               │           │  ║
║  │                              └─────────────────────────────────────┘           │  ║
║  └────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                              │                        ║
║                                          32-dim embedding    │                        ║
║                                                              ▼                        ║
║  ┌────────────────────────────────────────────────────────────────────────────────┐  ║
║  │                         COMPRESSION LAYER (Classical)                          │  ║
║  │                                                                                │  ║
║  │                     Linear(32 → 8) + LayerNorm + Tanh                          │  ║
║  │                     Output bounded to [-1, 1] for quantum encoding             │  ║
║  │                                                                                │  ║
║  └────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                              │                        ║
║                                          8-dim normalized    │                        ║
║                                                              ▼                        ║
║  ┌────────────────────────────────────────────────────────────────────────────────┐  ║
║  │                     VARIATIONAL QUANTUM CIRCUIT (Quantum CPU)                  │  ║
║  │                                                                                │  ║
║  │  |0⟩ ─ RY(πx₀) ─●───────────── RX(θ) ─ RY(θ) ─ RZ(θ) ─ ... ─ ⟨Z⟩               │  ║
║  │                 │                                                              │  ║
║  │  |0⟩ ─ RY(πx₁) ─X──●────────── RX(θ) ─ RY(θ) ─ RZ(θ) ─ ... ─ ⟨Z⟩               │  ║
║  │                    │                                                           │  ║
║  │  |0⟩ ─ RY(πx₂) ────X──●─────── RX(θ) ─ RY(θ) ─ RZ(θ) ─ ... ─ ⟨Z⟩               │  ║
║  │                       │                                                        │  ║
║  │      ...              ... (CNOT ring wraps to qubit 0)                         │  ║
║  │                                                                                │  ║
║  │  8 qubits × 3 layers = 72 trainable parameters                                 │  ║
║  │                                                                                │  ║
║  └────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                              │                        ║
║                                          8-dim ⟨Z⟩ values    │                        ║
║                                                              ▼                        ║
║  ┌────────────────────────────────────────────────────────────────────────────────┐  ║
║  │                        CLASSIFIER HEAD (Classical GPU)                         │  ║
║  │                                                                                │  ║
║  │                     Linear(8 → 16) + ReLU + Dropout(0.2)                       │  ║
║  │                     Linear(16 → 2) → Softmax → Prediction                      │  ║
║  │                                                                                │  ║
║  └────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                              │                        ║
║                                                              ▼                        ║
║                                                       [Class 0, Class 1]              ║
║                                                                                       ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Pipeline Flow Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  SMILES           Molecular          Node              Graph              Class    │
│  String           Graph              Embeddings        Embedding          Logits   │
│                                                                                     │
│  "c1ccccc1O"      7 atoms            [7, 64]           [1, 32]            [1, 2]   │
│  (Phenol)         14 edges           per layer         compressed         final    │
│      │                │                  │                 │                 │      │
│      ▼                ▼                  ▼                 ▼                 ▼      │
│  ┌───────┐       ┌─────────┐       ┌──────────┐       ┌─────────┐       ┌───────┐  │
│  │ Parse │ ───▶  │ Feature │ ───▶  │   GNN    │ ───▶  │Compress │ ───▶  │ VQC + │  │
│  │ RDKit │       │ Extract │       │ Message  │       │ + Norm  │       │Classify│  │
│  │       │       │ 145-dim │       │ Passing  │       │ 32 → 8  │       │        │  │
│  └───────┘       └─────────┘       └──────────┘       └─────────┘       └───────┘  │
│                                                                                     │
│  ────────────────────────────────────────────────────────────────────────────────   │
│  Time:   ~1ms          ~1ms            ~5ms              ~1ms          ~50ms       │
│  Device: CPU           CPU              GPU               GPU         Quantum/CPU   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Quantum Circuit Detail (ASCII)

```
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                     VARIATIONAL QUANTUM CIRCUIT (8 Qubits, 3 Layers)                  ║
╠═══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                       ║
║        ENCODING              LAYER 1                   LAYER 2-3           MEASURE   ║
║       ───────────       ─────────────────           ─────────────         ─────────  ║
║                                                                                       ║
║  q₀: ─|0⟩─ RY(πx₀) ─●───────── RX(θ₀) ─ RY(θ₁) ─ RZ(θ₂) ─ [ ... ] ─ ⟨Z₀⟩ ────▶ y₀   ║
║                     │                                                                 ║
║  q₁: ─|0⟩─ RY(πx₁) ─┼─●─────── RX(θ₃) ─ RY(θ₄) ─ RZ(θ₅) ─ [ ... ] ─ ⟨Z₁⟩ ────▶ y₁   ║
║                     │ │                                                               ║
║  q₂: ─|0⟩─ RY(πx₂) ─┼─┼─●───── RX(θ₆) ─ RY(θ₇) ─ RZ(θ₈) ─ [ ... ] ─ ⟨Z₂⟩ ────▶ y₂   ║
║                     │ │ │                                                             ║
║  q₃: ─|0⟩─ RY(πx₃) ─┼─┼─┼─●─── RX(θ₉) ─ RY(θ₁₀)─ RZ(θ₁₁)─ [ ... ] ─ ⟨Z₃⟩ ────▶ y₃   ║
║                     │ │ │ │                                                           ║
║  q₄: ─|0⟩─ RY(πx₄) ─┼─┼─┼─┼─●─ RX(θ₁₂)─ RY(θ₁₃)─ RZ(θ₁₄)─ [ ... ] ─ ⟨Z₄⟩ ────▶ y₄   ║
║                     │ │ │ │ │                                                         ║
║  q₅: ─|0⟩─ RY(πx₅) ─┼─┼─┼─┼─┼─ RX(θ₁₅)─ RY(θ₁₆)─ RZ(θ₁₇)─ [ ... ] ─ ⟨Z₅⟩ ────▶ y₅   ║
║                     │ │ │ │ │ │                                                       ║
║  q₆: ─|0⟩─ RY(πx₆) ─┼─┼─┼─┼─┼─┼─RX(θ₁₈)─ RY(θ₁₉)─ RZ(θ₂₀)─ [ ... ] ─ ⟨Z₆⟩ ────▶ y₆   ║
║                     │ │ │ │ │ │ │                                                     ║
║  q₇: ─|0⟩─ RY(πx₇) ─┴─┴─┴─┴─┴─┴─┴ RX(θ₂₁)─ RY(θ₂₂)─ RZ(θ₂₃)─ [ ... ] ─⟨Z₇⟩ ────▶ y₇   ║
║                     ▲                                                                 ║
║                     └── CNOT ring (q₇ → q₀ wraps)                                     ║
║                                                                                       ║
║  ═════════════════════════════════════════════════════════════════════════════════   ║
║  Gate Summary:                                                                        ║
║    • RY Encoding:        8 gates  (features × π scaling)                             ║
║    • CNOT per layer:     8 gates  (circular entanglement)                            ║
║    • Rotations/layer:   24 gates  (RX, RY, RZ × 8 qubits)                            ║
║    • Total (3 layers): 104 gates                                                     ║
║    • Parameters:        72        (3 rotations × 8 qubits × 3 layers)                ║
║    • Circuit depth:    ~13        (encoding + 3×[CNOT + rotations])                  ║
║                                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 4. Component Interaction Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           COMPONENT INTERACTION                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│                          FORWARD PASS                                               │
│    ┌──────────────────────────────────────────────────────────────────────────┐    │
│    │                                                                          │    │
│    │  ┌─────────┐      ┌─────────────┐      ┌───────────┐      ┌──────────┐  │    │
│    │  │MolGraph │ ───▶ │ GNNEncoder  │ ───▶ │Compression│ ───▶ │VQCLayer  │  │    │
│    │  │Builder  │      │   (GPU)     │      │  (GPU)    │      │  (CPU)   │  │    │
│    │  └─────────┘      └─────────────┘      └───────────┘      └──────────┘  │    │
│    │                                                                   │      │    │
│    │                                                                   ▼      │    │
│    │                                                            ┌──────────┐  │    │
│    │                                                            │Classifier│  │    │
│    │                                                            │  (GPU)   │  │    │
│    │                                                            └──────────┘  │    │
│    │                                                                   │      │    │
│    │                                                                   ▼      │    │
│    │                                                            [logits]      │    │
│    └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                     │
│                          BACKWARD PASS                                              │
│    ┌──────────────────────────────────────────────────────────────────────────┐    │
│    │                                                                          │    │
│    │  ┌──────────┐      ┌───────────────────────────────────────────────────┐ │    │
│    │  │CrossEntry│ ───▶ │             PyTorch Autograd                      │ │    │
│    │  │Loss      │      │  ┌─────────────────────────────────────────────┐  │ │    │
│    │  └──────────┘      │  │  Classical Components:                      │  │ │    │
│    │                    │  │    • Chain rule through GNN, MLP            │  │ │    │
│    │                    │  │    • Standard backprop                      │  │ │    │
│    │                    │  └─────────────────────────────────────────────┘  │ │    │
│    │                    │  ┌─────────────────────────────────────────────┐  │ │    │
│    │                    │  │  Quantum Components (via PennyLane):        │  │ │    │
│    │                    │  │    • Parameter-shift rule                   │  │ │    │
│    │                    │  │    • 2× circuit evaluations per param       │  │ │    │
│    │                    │  │    • ∂f/∂θ = ½[f(θ+π/2) - f(θ-π/2)]        │  │ │    │
│    │                    │  └─────────────────────────────────────────────┘  │ │    │
│    │                    └───────────────────────────────────────────────────┘ │    │
│    │                                                                          │    │
│    └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Quantum Math Flow (ASCII)

```
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                          QUANTUM LAYER MATHEMATICAL FLOW                              ║
╠═══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                       ║
║  INPUT: x = [x₀, x₁, x₂, x₃, x₄, x₅, x₆, x₇] ∈ ℝ⁸  (compressed embedding)            ║
║                                                                                       ║
║  ─────────────────────────────────────────────────────────────────────────────────   ║
║                                                                                       ║
║  STEP 1: Feature Encoding                                                            ║
║  ─────────────────────────                                                           ║
║                                                                                       ║
║    |ψ₀⟩ = ⊗ᵢ RY(π·xᵢ)|0⟩                                                             ║
║                                                                                       ║
║    Single qubit:  RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩                               ║
║                                                                                       ║
║    Full state:    |ψ₀⟩ ∈ ℂ^256           (2⁸-dimensional Hilbert space)              ║
║                                                                                       ║
║  ─────────────────────────────────────────────────────────────────────────────────   ║
║                                                                                       ║
║  STEP 2: Variational Layers (×3)                                                     ║
║  ────────────────────────────────                                                    ║
║                                                                                       ║
║    For layer l = 1,2,3:                                                              ║
║                                                                                       ║
║      2a. Entanglement Layer:    U_ent = ∏ᵢ CNOT(i, (i+1) mod 8)                      ║
║                                                                                       ║
║          CNOT: |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩                            ║
║                                                                                       ║
║      2b. Rotation Layer:        U_rot(θₗ) = ⊗ᵢ RZ(θₗ,ᵢ,₂)·RY(θₗ,ᵢ,₁)·RX(θₗ,ᵢ,₀)      ║
║                                                                                       ║
║    Combined:  |ψₗ⟩ = U_rot(θₗ) · U_ent · |ψₗ₋₁⟩                                       ║
║                                                                                       ║
║  ─────────────────────────────────────────────────────────────────────────────────   ║
║                                                                                       ║
║  STEP 3: Measurement                                                                 ║
║  ───────────────────                                                                 ║
║                                                                                       ║
║    Output:  yᵢ = ⟨ψ_final| Zᵢ |ψ_final⟩    for i = 0,...,7                           ║
║                                                                                       ║
║    Property:  yᵢ ∈ [-1, 1]     (expectation of ±1 eigenvalues)                       ║
║                                                                                       ║
║  ─────────────────────────────────────────────────────────────────────────────────   ║
║                                                                                       ║
║  GRADIENT: Parameter-Shift Rule                                                      ║
║  ──────────────────────────────                                                      ║
║                                                                                       ║
║    ∂f/∂θ = ½ [ f(θ + π/2) - f(θ - π/2) ]                                             ║
║                                                                                       ║
║    Cost: 2 × circuit evaluations per parameter                                       ║
║          72 parameters → 144 circuit calls per gradient step                         ║
║                                                                                       ║
║  ─────────────────────────────────────────────────────────────────────────────────   ║
║                                                                                       ║
║  OUTPUT: y = [y₀, y₁, y₂, y₃, y₄, y₅, y₆, y₇] ∈ [-1,1]⁸                              ║
║                                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 6. Python Code for Diagram Generation

### 6.1 Architecture Diagram with Matplotlib

```python
"""
Generate publication-quality architecture diagram using matplotlib.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram(save_path='architecture_diagram.png'):
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#E3F2FD',
        'gnn': '#C8E6C9', 
        'compress': '#FFF9C4',
        'quantum': '#F3E5F5',
        'classifier': '#FFCCBC',
        'arrow': '#37474F'
    }
    
    # Component boxes
    components = [
        (0.5, 5, 2, 1.5, 'SMILES\nInput', colors['input']),
        (3, 5, 2.5, 1.5, 'RDKit\nGraph', colors['input']),
        (6, 4, 3.5, 3, 'GNN Encoder\n3×GCN Layers\n64-dim hidden', colors['gnn']),
        (10, 5, 2, 1.5, 'Compress\n32→8', colors['compress']),
        (12.5, 4, 3, 3, 'Quantum\nCircuit\n8 qubits', colors['quantum']),
        (12.5, 0.5, 3, 2, 'Classifier\n8→16→2', colors['classifier']),
    ]
    
    for x, y, w, h, label, color in components:
        box = FancyBboxPatch((x, y), w, h, 
                             boxstyle="round,pad=0.1,rounding_size=0.2",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Arrows
    arrows = [
        (2.5, 5.75, 3, 5.75),
        (5.5, 5.75, 6, 5.75),
        (9.5, 5.5, 10, 5.5),
        (12, 5.75, 12.5, 5.75),
        (14, 4, 14, 2.5),
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    
    # Dimension labels
    dims = [
        (2.75, 5.2, '145-dim'),
        (5.75, 5.2, 'Graph'),
        (9.75, 5.2, '32-dim'),
        (12.25, 5.2, '8-dim'),
        (14.5, 3.2, '8-dim'),
    ]
    
    for x, y, label in dims:
        ax.text(x, y, label, fontsize=8, style='italic', ha='center')
    
    # Title
    ax.text(8, 9.5, 'Hybrid QMolNet Architecture', fontsize=16, 
            fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    create_architecture_diagram()
```

### 6.2 Quantum Circuit Diagram with Graphviz

```python
"""
Generate quantum circuit diagram using graphviz.
Requires: pip install graphviz
"""
from graphviz import Digraph

def create_circuit_diagram(save_path='quantum_circuit'):
    dot = Digraph(comment='Quantum Circuit', format='png')
    dot.attr(rankdir='LR', size='12,6', dpi='300')
    
    # Qubit lines
    for i in range(8):
        dot.node(f'q{i}_start', f'|0⟩', shape='plaintext')
        dot.node(f'q{i}_ry', f'RY(πx{i})', shape='box', style='rounded,filled', 
                 fillcolor='#E8F5E9')
        dot.node(f'q{i}_cnot', '●' if i < 7 else '○', shape='circle', 
                 width='0.3', height='0.3', style='filled', fillcolor='#BBDEFB')
        dot.node(f'q{i}_rx', f'RX(θ)', shape='box', style='rounded,filled',
                 fillcolor='#FFF3E0')
        dot.node(f'q{i}_ry2', f'RY(θ)', shape='box', style='rounded,filled',
                 fillcolor='#FFF3E0')
        dot.node(f'q{i}_rz', f'RZ(θ)', shape='box', style='rounded,filled',
                 fillcolor='#FFF3E0')
        dot.node(f'q{i}_measure', f'⟨Z{i}⟩', shape='box', style='rounded,filled',
                 fillcolor='#FCE4EC')
        dot.node(f'q{i}_out', f'y{i}', shape='plaintext')
        
        # Connect horizontally
        dot.edge(f'q{i}_start', f'q{i}_ry')
        dot.edge(f'q{i}_ry', f'q{i}_cnot')
        dot.edge(f'q{i}_cnot', f'q{i}_rx')
        dot.edge(f'q{i}_rx', f'q{i}_ry2')
        dot.edge(f'q{i}_ry2', f'q{i}_rz')
        dot.edge(f'q{i}_rz', f'q{i}_measure')
        dot.edge(f'q{i}_measure', f'q{i}_out')
    
    # CNOT connections (vertical)
    for i in range(7):
        dot.edge(f'q{i}_cnot', f'q{i+1}_cnot', style='dashed', color='blue')
    dot.edge(f'q7_cnot', f'q0_cnot', style='dashed', color='blue', 
             constraint='false')
    
    dot.render(save_path, view=False, cleanup=True)
    print(f"Saved: {save_path}.png")

if __name__ == "__main__":
    create_circuit_diagram()
```

### 6.3 Pipeline Flow with Matplotlib

```python
"""
Generate pipeline flow diagram.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as PathEffects

def create_pipeline_diagram(save_path='pipeline_diagram.png'):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Stages
    stages = [
        (0.5, 1.5, 'SMILES\n"CCO"', '#E3F2FD'),
        (3, 1.5, 'Molecular\nGraph', '#C8E6C9'),
        (5.5, 1.5, 'GNN\nEncoder', '#A5D6A7'),
        (8, 1.5, 'Compress\n+ Norm', '#FFF59D'),
        (10.5, 1.5, 'Quantum\nCircuit', '#CE93D8'),
        (13, 1.5, 'Class\nLogits', '#FFAB91'),
    ]
    
    for x, y, label, color in stages:
        box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    for i in range(len(stages)-1):
        x1 = stages[i][0] + 0.8
        x2 = stages[i+1][0] - 0.8
        ax.annotate('', xy=(x2, 1.5), xytext=(x1, 1.5),
                    arrowprops=dict(arrowstyle='->', color='#37474F', lw=2))
    
    # Dimension annotations
    dims = ['145-dim', 'Nodes+Edges', '32-dim', '8-dim', '8-dim', '[0.1, 0.9]']
    for i, dim in enumerate(dims):
        ax.text(stages[i][0], 0.6, dim, ha='center', fontsize=7, style='italic')
    
    plt.title('Hybrid QMolNet Pipeline', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    create_pipeline_diagram()
```

---

## Usage Instructions

1. **ASCII Diagrams**: Copy directly into presentations, README files, or terminal displays
2. **Python Scripts**: Run to generate PNG files for slides and documentation
3. **Requirements**: `pip install matplotlib graphviz`

For graphviz, also install the system package:
- Windows: `choco install graphviz`
- Mac: `brew install graphviz`
- Linux: `apt install graphviz`
