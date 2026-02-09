# Hybrid QMolNet

**Hybrid Quantum-Classical Graph Neural Network for Molecular Property Prediction**

A research-grade implementation combining Graph Neural Networks (GNNs) with Variational Quantum Circuits (VQCs) for drug discovery applications.

```
┌─────────────────────────────────────────────────────────────┐
│                  Hybrid QMolNet Architecture                │
├─────────────────────────────────────────────────────────────┤
│   SMILES ─▶ [Molecular Graph] ─▶ [GNN Encoder] ─▶ (32-dim) │
│                                        │                    │
│                                        ▼                    │
│                              [Linear Compression]           │
│                                        │                    │
│                                        ▼                    │
│                              [8-Qubit VQC] ◇────────────── │
│                                        │    │ Angle Embed  │
│                                        │    │ Var. Layers  │
│                                        │    │ Measurements │
│                                        ▼    └──────────────┘
│                              [Classifier Head]              │
│                                        │                    │
│                                        ▼                    │
│                                  Prediction                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Features

- **Hybrid Architecture**: GCN encoder + 8-qubit Variational Quantum Circuit (PennyLane)
- **End-to-End Pipeline**: SMILES → Graph → GNN → VQC → Prediction
- **Web Demo**: Interactive Streamlit application
- **Model Comparison**: Toggle between hybrid and classical baselines
- **Comprehensive Evaluation**: ROC-AUC, accuracy, confusion matrices
- **Safe Scientific Language**: No exaggerated quantum claims

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Ashmit-Singh/hybrid-qmolnet.git
cd hybrid-qmolnet

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Quick Start

### Run Web Demo
```bash
streamlit run app.py
```

This launches an interactive web interface where you can:
- Enter any SMILES molecular string
- Visualize the molecule structure
- Get property predictions with confidence scores
- Compare hybrid vs classical model outputs
- View technical explanations of the pipeline

### Train Models
```bash
# Quick verification (synthetic data)
python run_all.py --quick

# Full training
python run_all.py --samples 500 --epochs 50 --batch_size 32

# Train on BBBP dataset
python run_all.py --data_path data/bbbp.csv --smiles_col smiles --label_col p_np --output_dir outputs_bbbp
```

### Generate Reports
```bash
python generate_report.py --output_dir outputs_bbbp
```

### Run Tests
```bash
python -m pytest tests/ -v
```

## 📂 Project Structure

```
hybrid-qmolnet/
├── app.py                  # Streamlit web demo
├── run_all.py              # Training pipeline runner
├── generate_report.py      # Evaluation report generator
│
├── models/
│   ├── hybrid_model.py     # HybridQMolNet (GNN + VQC)
│   ├── gnn_encoder.py      # Graph Convolutional Network
│   ├── quantum_layer.py    # Variational Quantum Circuit
│   └── baselines.py        # Classical baselines
│
├── training/
│   └── trainer.py          # Training loop
│
├── evaluation/
│   ├── evaluator.py        # Model evaluation
│   └── metrics.py          # Metric computation
│
├── visualization/
│   ├── plots.py            # Training curves, ROC, confusion matrix
│   ├── molecule_viz.py     # Molecule structure visualization
│   └── embedding_viz.py    # Embedding projections
│
├── utils/
│   ├── smiles_to_graph.py  # SMILES → PyG Data conversion
│   ├── data_loader.py      # Dataset loading
│   ├── formatters.py       # Prediction output formatting
│   ├── explanation.py      # Technical explanations
│   └── helpers.py          # Utility functions
│
├── tests/                  # Unit tests
├── outputs/                # Training outputs
└── data/                   # Datasets
```

## 📊 Model Components

### Hybrid Model Pipeline
1. **SMILES Parsing**: RDKit converts SMILES to molecule objects
2. **Graph Construction**: Atoms → nodes (145 features), bonds → edges
3. **GNN Encoding**: 3-layer GCN produces 32-dim molecular embedding
4. **Compression**: Linear layer maps to 8 dimensions (qubit count)
5. **Quantum Transform**: 8-qubit VQC with angle encoding and variational layers
6. **Classification**: Final linear layer outputs class probabilities

### Baseline Models
- **GNNClassifier**: Same GNN encoder with classical MLP head
- **DescriptorMLP**: Pre-computed molecular descriptors + MLP

## 📈 Expected Outputs

After training, you'll find in the output directory:
- `checkpoints/best.pt` - Best model weights
- `figures/` - Training curves, ROC curves, confusion matrices
- `reports/` - Markdown and CSV evaluation reports

## 🧪 Example Usage

### Python API
```python
from models.hybrid_model import HybridQMolNet
from utils.smiles_to_graph import smiles_to_graph
from utils.formatters import format_prediction_output
import torch

# Load model
model = HybridQMolNet(node_feature_dim=145, n_qubits=8)
model.load_state_dict(torch.load('outputs_bbbp/checkpoints/best.pt')['model_state_dict'])
model.eval()

# Predict
smiles = "CC(=O)Nc1ccc(O)cc1"  # Paracetamol
graph = smiles_to_graph(smiles)
from torch_geometric.data import Batch
batch = Batch.from_data_list([graph])

with torch.no_grad():
    logits = model.forward_batch(batch)
    prob = torch.softmax(logits, dim=1)[0, 1].item()

# Format output
result = format_prediction_output(prob, task_type="bbbp", model_name="hybrid")
print(f"{result['label']} ({result['confidence']} confidence)")
```

## ⚠️ Scientific Disclaimer

This model provides computational predictions based on molecular structure analysis. Results are estimates and should not replace experimental validation. The hybrid quantum-classical approach is a research methodology; no claims of quantum advantage are made without rigorous benchmarking.

## 📝 License

MIT License

## 🙏 Acknowledgments

- PyTorch & PyTorch Geometric
- PennyLane (Xanadu)
- RDKit
- Streamlit
