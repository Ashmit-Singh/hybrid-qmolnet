# Hybrid QMolNet

**Hybrid Graph Neural Network with Variational Quantum Circuit for Molecular Property Prediction**

---

## Abstract

Hybrid QMolNet is a research-oriented hybrid quantum–classical learning framework for molecular property prediction. The model integrates a graph neural network (GNN) molecular encoder with a variational quantum circuit (VQC) layer to study hybrid representation learning under near-term quantum (NISQ) constraints. Molecular structures are converted from SMILES strings into graphs, encoded via message-passing neural networks, compressed into qubit-compatible embeddings, and processed through a parameterized quantum circuit before final classification. The repository provides a complete, reproducible implementation including baselines, evaluation tooling, and an interactive demonstration interface.

This project is intended for experimental and educational use and does not claim quantum computational advantage.

---

## Problem Setting

Molecular property prediction is commonly formulated as a supervised learning problem where a model learns a mapping:

```
molecular structure → predicted property
```

Graph neural networks are well-suited for this task because molecules naturally form graphs (atoms as nodes, bonds as edges). This project investigates whether inserting a shallow variational quantum circuit as a learnable nonlinear transformation layer on top of graph embeddings is practically trainable and performance-competitive within a hybrid pipeline.

The focus is on architectural feasibility and empirical behavior rather than quantum speedup claims.

---

## Method Overview

The Hybrid QMolNet pipeline is:

```
SMILES → Molecular Graph → GNN Encoder → Embedding (32d)
        → Linear Compression → Variational Quantum Circuit
        → Classifier → Prediction
```

### Components

**Graph Construction**

* RDKit-based SMILES parsing
* Atom-level feature vectors (~145 dimensions)
* Bidirectional bond edges
* PyTorch Geometric data objects

**GNN Encoder**

* Multi-layer graph convolution network
* Message passing over molecular topology
* Global pooling → fixed-length graph embedding

**Compression Layer**

* Linear projection from embedding dimension to qubit count
* Normalization and bounded activation for angle encoding

**Variational Quantum Circuit**

* 8-qubit circuit implemented in PennyLane
* Angle encoding of compressed features
* Shallow entangling ansatz
* Pauli expectation measurements
* Parameter-shift gradient computation

**Classifier Head**

* Small classical MLP on quantum outputs

The quantum circuit functions as a parameterized feature transformation layer, not a chemistry simulator.

---

## Baseline Models

To isolate the effect of the quantum layer, the repository includes:

* **GNN + MLP baseline** — identical graph encoder with purely classical head
* **Descriptor + MLP baseline** — fixed molecular descriptors with MLP

All models share comparable training and evaluation procedures.

---

## Implementation Details

* Framework: PyTorch + PyTorch Geometric
* Quantum simulation: PennyLane `default.qubit`
* Chemistry toolkit: RDKit
* Training: Adam/AdamW optimizers with early stopping
* Metrics: Accuracy, ROC-AUC, F1, confusion matrix
* Visualization: ROC curves, loss curves, embedding projections

The quantum circuit depth and qubit count are intentionally limited to remain compatible with near-term hardware assumptions.

---

## Repository Structure

```
models/              Hybrid and baseline model definitions
training/            Training loop and utilities
evaluation/          Metrics and evaluation runners
visualization/       Plotting and molecule visualization
utils/               Data processing and formatting
app.py               Streamlit demo interface
run_all.py           End-to-end training entry point
generate_report.py   Evaluation report generator
tests/               Unit tests
```

---

## Installation

```bash
git clone https://github.com/Ashmit-Singh/hybrid-qmolnet.git
cd hybrid-qmolnet

python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

RDKit may require conda installation on some platforms.

---

## Reproducible Training

Quick verification run:

```bash
python run_all.py --quick
```

Example training run:

```bash
python run_all.py \
  --data_path data/bbbp.csv \
  --smiles_col smiles \
  --label_col p_np \
  --epochs 50 \
  --batch_size 32 \
  --output_dir outputs_bbbp
```

---

## Evaluation Outputs

Training runs produce:

* model checkpoints
* ROC curves
* confusion matrices
* training loss plots
* CSV and markdown metric summaries

Reports can be regenerated with:

```bash
python generate_report.py --output_dir outputs_bbbp
```

---

## Interactive Demonstration

A lightweight Streamlit interface is provided for demonstration and testing:

```bash
streamlit run app.py
```

The demo supports:

* SMILES input
* molecule visualization
* hybrid vs classical model toggle
* probability + text label output
* technical explanation panel

---

## Limitations

* Quantum layer evaluated via classical simulation
* Small qubit count and shallow circuit only
* No claim of quantum computational advantage
* Dataset scale may be limited by simulation cost
* Performance gains, where observed, are empirical and task-dependent

The GNN encoder provides the majority of representational capacity; the quantum layer acts as a nonlinear refinement stage.

---

## Intended Use

This repository is intended for:

* hybrid quantum–classical ML experimentation
* educational use
* hackathon and prototype demonstrations
* architecture studies

It is **not** intended for clinical or industrial drug discovery decisions.

---

## License

MIT License

