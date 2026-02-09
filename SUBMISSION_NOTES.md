# Hybrid QMolNet — Hackathon Submission Notes

## For Judges: Quick Summary

**Project:** Hybrid Quantum-Classical Neural Network for Molecular Property Prediction

**Core Innovation:** Combines Graph Neural Networks (GNN) for molecular structure encoding with a Variational Quantum Circuit (VQC) for feature transformation, enabling end-to-end gradient-based training of a hybrid quantum-classical model.

---

## Technical Highlights

### Architecture
```
SMILES → Molecular Graph → GNN (3-layer GCN) → Compress → 8-Qubit VQC → Classifier
```

### Quantum Component
- **8-qubit** variational circuit
- **Angle encoding** for classical-to-quantum mapping
- **3 variational layers** with trainable rotation gates
- **Ring entanglement** topology (NISQ-compatible)
- **Parameter-shift rule** for quantum gradient computation

### Training
- End-to-end gradient-based optimization
- Hybrid backpropagation through quantum layer
- PennyLane + PyTorch integration

---

## What Makes This Hackathon-Ready

1. **Complete Pipeline**: From raw SMILES to prediction with one command
2. **Interactive Demo**: Streamlit web app for live testing
3. **Model Comparison**: Side-by-side hybrid vs classical
4. **Proper Baselines**: Classical GNN and MLP comparisons
5. **Visualization**: Molecules, circuits, metrics all visualized
6. **Reproducibility**: Seeds, configs, and clear instructions

---

## Scientific Honesty Statement

We make **no claims of quantum advantage**. This project demonstrates:

- Feasibility of hybrid quantum-classical molecular ML
- Integration of VQCs with graph neural networks
- End-to-end trainability of quantum parameters

The quantum circuit runs on a **classical simulator** (PennyLane default.qubit). Real hardware execution would require:
- Noise mitigation
- Hardware-specific transpilation
- Potentially different circuit design

---

## Limitations

1. **Simulation Only**: No actual quantum hardware used
2. **Dataset Size**: Limited by quantum layer computational cost
3. **Binary Classification**: Currently supports binary tasks only
4. **CPU Training**: Quantum simulation is slow on CPU

---

## Demo Commands

```bash
# Launch web demo
streamlit run app.py

# Run training (quick mode)
python run_all.py --quick

# Generate evaluation report
python generate_report.py --output_dir outputs_bbbp
```

---

## Repository Structure

| Directory | Purpose |
|-----------|---------|
| `models/` | Neural network architectures |
| `training/` | Training loop and optimization |
| `evaluation/` | Metrics and model evaluation |
| `visualization/` | All plotting and visualization |
| `utils/` | Data loading and helpers |
| `tests/` | Unit tests |
| `app.py` | Streamlit demo application |

---

## Key Files for Review

1. **`models/hybrid_model.py`** — Main hybrid architecture
2. **`models/quantum_layer.py`** — VQC implementation
3. **`app.py`** — Interactive demo
4. **`README.md`** — Full documentation

---

## Contact

Repository: https://github.com/Ashmit-Singh/hybrid-qmolnet
