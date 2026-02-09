# Visualization package initialization
"""
Visualization utilities for Hybrid QMolNet

- Plots: Training curves, ROC, confusion matrix
- Molecule visualization
- Quantum circuit diagrams
- Embedding projections
"""

from .plots import (
    plot_training_curves,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_pr_curve,
    plot_multiple_roc_curves,
)
from .molecule_viz import (
    plot_molecule,
    plot_molecular_graph,
    plot_molecule_gallery,
    smiles_to_image,
)
from .embedding_viz import plot_embeddings_pca, plot_embeddings_tsne
from .circuit_viz import (
    get_circuit_ascii,
    get_circuit_explanation,
    get_circuit_parameters_info,
    draw_circuit_matplotlib,
    print_circuit_summary,
)

__all__ = [
    # Plotting
    'plot_training_curves',
    'plot_roc_curve',
    'plot_confusion_matrix',
    'plot_metrics_comparison',
    'plot_pr_curve',
    'plot_multiple_roc_curves',
    # Molecule visualization
    'plot_molecule',
    'plot_molecular_graph',
    'plot_molecule_gallery',
    'smiles_to_image',
    # Embeddings
    'plot_embeddings_pca',
    'plot_embeddings_tsne',
    # Quantum circuit
    'get_circuit_ascii',
    'get_circuit_explanation',
    'get_circuit_parameters_info',
    'draw_circuit_matplotlib',
    'print_circuit_summary',
]

