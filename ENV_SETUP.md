# Environment Setup for Hybrid QMolNet

## Issue Detected

The current Python environment has deep incompatibility between package versions. 
Basic `import pennylane` fails with a `NumpyMimic` AttributeError.

## Tested Combinations (All Failed)

| PyTorch | numpy | PennyLane | PyG | Error |
|---------|-------|-----------|-----|-------|
| 2.9.0 | 1.26.4 | 0.44.0 | 2.7.0 | `torch.Device` AttributeError |
| 2.5.1 | 1.26.4 | 0.42.0 | 2.7.0 | `NumpyMimic` AttributeError |
| 2.3.1 | 1.24.4 | 0.42.0 | 2.7.0 | `NumpyMimic` AttributeError |

## Recommended Solution: Fresh Virtual Environment

```bash
# Create new virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install locked, compatible versions
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric==2.5.2 -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install pennylane==0.35.1
pip install rdkit==2023.9.4
pip install scikit-learn pandas matplotlib seaborn networkx pytest

# Verify
python -c "import torch; import pennylane; import torch_geometric; print('All imports OK')"
```

## Alternative: Use requirements-lock.txt

Create `requirements-lock.txt`:
```
torch==2.2.1
torch_geometric==2.5.2
pennylane==0.35.1
numpy==1.24.3
rdkit==2023.9.4
scikit-learn==1.4.0
pandas==2.2.0
matplotlib==3.8.3
seaborn==0.13.2
networkx==3.2.1
pytest==8.0.0
```

Install:
```bash
pip install -r requirements-lock.txt -f https://download.pytorch.org/whl/cpu
```

## After Environment Fix

Run the full pipeline:
```bash
python run_all.py --quick  # Quick test
python run_all.py          # Full training
```
