# Hybrid QMolNet: Quantum-Classical Graph Neural Network for Drug Discovery

![Hybrid QMolNet Architecture](outputs/figures/roc_comparison.png)

A research-grade implementation of a hybrid quantum-classical neural network for molecular property prediction. This project combines Graph Neural Networks (GNNs) for molecular feature extraction with Variational Quantum Circuits (VQCs) for enhanced expressibility.

## 🚀 Key Features

*   **Hybrid Architecture**: GCN-based encoder + 8-qubit Variational Quantum Circuit (PennyLane).
*   **End-to-End Pipeline**: From SMILES strings to training, evaluation, and visualization.
*   **Robust Environment**: Fully tested compatibility stack for PyTorch + PennyLane + RDKit.
*   **Comprehensive Evaluation**: Compare against classical MLP and GNN baselines.
*   **Visualization**: Automated generation of training curves, confusion matrices, and embedding clusters.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Ashmit-Singh/hybrid-qmolnet.git
    cd hybrid-qmolnet
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1  # Windows
    # source .venv/bin/activate # Linux/Mac
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This project requires specific versions of `numpy` and `pennylane` to ensure compatibility. The `requirements.txt` handles this.*

## 🏃 Usage

### Quick Verification
Run a fast test on a small synthetic dataset:
```bash
python run_all.py --quick
```

### Full Experiment
Run the full training pipeline with custom parameters:
```bash
python run_all.py --samples 500 --epochs 50 --batch_size 32
```

### Run Unit Tests
```bash
python -m pytest tests/
```

## 📂 Project Structure

```
hybrid-qmolnet/
├── data/               # Dataset storage
├── models/             # PyTorch models (GNN, Hybrid, Quantum Layer)
├── training/           # Training loops and evaluators
├── utils/              # Data loading and graph conversion
├── outputs/            # Generated figures and results
├── tests/              # Unit tests
├── requirements.txt    # Dependency list
├── run_all.py          # Main execution script
└── README.md           # This file
```

## 📊 Results

The model generates real-time training metrics and saves figures to `outputs/figures/`.
- **Hybrid vs Classical**: Compare ROC-AUC and Accuracy.
- **Quantum Advantage**: Analyze if the VQC layer improves embedding separation.


