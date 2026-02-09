"""
Technical Explanation Generator Module

Generates structured technical explanations for model predictions.
Describes the graph structure, GNN processing, quantum transforms,
and hybrid inference pipeline in accessible terms.

This is NOT an interpretability module (like SHAP or LIME).
It provides static but technically accurate explanations.
"""

from typing import Dict, Optional, Any


def generate_explanation(
    smiles: str,
    model_type: str = "hybrid",
    n_atoms: Optional[int] = None,
    n_bonds: Optional[int] = None,
    prediction_prob: Optional[float] = None,
    task_type: str = "binary",
) -> Dict[str, str]:
    """
    Generate a structured technical explanation for a prediction.
    
    Args:
        smiles: SMILES string of the molecule
        model_type: Type of model used (hybrid, gnn, mlp)
        n_atoms: Number of atoms in molecule (optional)
        n_bonds: Number of bonds in molecule (optional)
        prediction_prob: Prediction probability (optional)
        task_type: Type of prediction task
    
    Returns:
        Dictionary with explanation sections
    """
    explanations = {}
    
    # Section 1: Graph Structure
    explanations["graph_structure"] = _explain_graph_structure(
        smiles, n_atoms, n_bonds
    )
    
    # Section 2: Model-specific processing
    if model_type.lower() in ["hybrid", "hybridqmolnet"]:
        explanations["gnn_processing"] = _explain_gnn_processing()
        explanations["quantum_transform"] = _explain_quantum_transform()
        explanations["hybrid_inference"] = _explain_hybrid_inference()
    elif model_type.lower() in ["gnn", "gnnclassifier"]:
        explanations["gnn_processing"] = _explain_gnn_processing()
        explanations["classical_inference"] = _explain_classical_inference()
    else:
        explanations["mlp_processing"] = _explain_mlp_processing()
    
    # Section 3: Output interpretation
    explanations["output_interpretation"] = _explain_output(
        prediction_prob, task_type
    )
    
    # Full explanation text
    explanations["full_explanation"] = _compile_full_explanation(
        explanations, model_type
    )
    
    return explanations


def _explain_graph_structure(
    smiles: str,
    n_atoms: Optional[int],
    n_bonds: Optional[int],
) -> str:
    """Generate explanation of molecular graph structure."""
    
    base = (
        "**Molecular Graph Representation**\n\n"
        f"The input molecule (SMILES: `{smiles}`) is represented as a graph "
        "where:\n"
        "- **Nodes** represent atoms in the molecule\n"
        "- **Edges** represent chemical bonds between atoms\n\n"
    )
    
    if n_atoms is not None:
        base += f"This molecule contains **{n_atoms} atoms** "
    if n_bonds is not None:
        if n_atoms is not None:
            base += f"and **{n_bonds} bonds**.\n\n"
        else:
            base += f"This molecule contains **{n_bonds} bonds**.\n\n"
    
    base += (
        "Each atom (node) is encoded with a 145-dimensional feature vector "
        "capturing:\n"
        "- Atomic number (element type)\n"
        "- Degree (number of connected atoms)\n"
        "- Formal charge\n"
        "- Hybridization state (sp, sp2, sp3, etc.)\n"
        "- Aromaticity\n"
        "- Ring membership\n"
    )
    
    return base


def _explain_gnn_processing() -> str:
    """Generate explanation of GNN message passing."""
    
    return (
        "**Graph Neural Network Processing**\n\n"
        "The molecular graph is processed through a Graph Convolutional "
        "Network (GCN) with multiple layers of message passing:\n\n"
        "1. **Message Passing**: Each atom aggregates information from its "
        "neighboring atoms (bonded atoms) through learned transformations.\n\n"
        "2. **Feature Updating**: Atom features are updated based on the "
        "aggregated neighborhood information, capturing local structural "
        "patterns.\n\n"
        "3. **Graph Pooling**: Node features across the entire molecule are "
        "combined into a single graph-level representation (32-dimensional "
        "embedding).\n\n"
        "This process allows the network to learn hierarchical structural "
        "features relevant to the molecular property being predicted."
    )


def _explain_quantum_transform() -> str:
    """Generate explanation of quantum variational circuit."""
    
    return (
        "**Variational Quantum Circuit Transform**\n\n"
        "The GNN embeddings are processed through a parameterized quantum "
        "circuit:\n\n"
        "1. **Dimensionality Compression**: The 32-dimensional GNN embedding "
        "is compressed to 8 dimensions to match the number of qubits.\n\n"
        "2. **Angle Encoding**: Features are encoded into quantum states "
        "using rotation gates (RY), mapping classical values to quantum "
        "amplitudes.\n\n"
        "3. **Variational Layers**: The circuit applies alternating layers of:\n"
        "   - **Entangling Gates** (CNOT): Create correlations between qubits\n"
        "   - **Parameterized Rotations** (RX, RY, RZ): Learned transformations\n\n"
        "4. **Measurement**: Expectation values of Pauli-Z operators on each "
        "qubit are computed, producing 8-dimensional quantum-transformed "
        "features.\n\n"
        "Note: This is a quantum-classical *hybrid* approach running on a "
        "classical simulator. The quantum circuit provides a different feature "
        "transformation that may capture certain structural patterns."
    )


def _explain_hybrid_inference() -> str:
    """Generate explanation of hybrid model inference."""
    
    return (
        "**Hybrid Inference Pipeline**\n\n"
        "The complete prediction pipeline:\n\n"
        "```\n"
        "SMILES → Graph → GNN Encoder → Compression → VQC → Classifier → Prediction\n"
        "```\n\n"
        "1. SMILES string is parsed and converted to a molecular graph\n"
        "2. GNN extracts graph-level molecular embeddings (32-dim)\n"
        "3. Linear layer compresses to qubit count (8-dim)\n"
        "4. Variational Quantum Circuit transforms features (8-dim)\n"
        "5. Classification head produces final prediction\n\n"
        "The model is trained end-to-end using gradient-based optimization, "
        "with quantum gradients computed via the parameter-shift rule."
    )


def _explain_classical_inference() -> str:
    """Generate explanation of classical GNN inference."""
    
    return (
        "**Classical GNN Inference**\n\n"
        "The prediction pipeline:\n\n"
        "```\n"
        "SMILES → Graph → GNN Encoder → MLP Classifier → Prediction\n"
        "```\n\n"
        "1. SMILES string is parsed and converted to a molecular graph\n"
        "2. GNN extracts graph-level molecular embeddings\n"
        "3. Multi-layer perceptron produces final prediction\n\n"
        "This classical baseline uses the same GNN architecture but replaces "
        "the quantum circuit with a fully classical processing layer."
    )


def _explain_mlp_processing() -> str:
    """Generate explanation of MLP descriptor-based processing."""
    
    return (
        "**Descriptor-based MLP Processing**\n\n"
        "This simpler baseline uses pre-computed molecular descriptors:\n\n"
        "1. **Descriptor Computation**: RDKit computes standard molecular "
        "descriptors (molecular weight, LogP, H-bond donors/acceptors, etc.)\n\n"
        "2. **MLP Classification**: A multi-layer perceptron processes "
        "the descriptor vector to produce predictions.\n\n"
        "This approach does not use graph structure directly but relies on "
        "established cheminformatics descriptors."
    )


def _explain_output(
    prediction_prob: Optional[float],
    task_type: str,
) -> str:
    """Generate explanation of output interpretation."""
    
    base = (
        "**Output Interpretation**\n\n"
        "The model outputs a probability estimate between 0 and 1, "
        "representing the estimated likelihood of the positive class.\n\n"
    )
    
    if prediction_prob is not None:
        prob_pct = prediction_prob * 100
        base += f"For this molecule, the model estimates **{prob_pct:.1f}%** "
        base += "probability for the positive class.\n\n"
    
    base += (
        "**Important**: These predictions are computational estimates. "
        "Confidence levels reflect model certainty, not ground truth. "
        "Experimental validation is always recommended for critical "
        "applications."
    )
    
    return base


def _compile_full_explanation(
    sections: Dict[str, str],
    model_type: str,
) -> str:
    """Compile all sections into full explanation text."""
    
    parts = [sections["graph_structure"]]
    
    if "gnn_processing" in sections:
        parts.append(sections["gnn_processing"])
    if "quantum_transform" in sections:
        parts.append(sections["quantum_transform"])
    if "hybrid_inference" in sections:
        parts.append(sections["hybrid_inference"])
    if "classical_inference" in sections:
        parts.append(sections["classical_inference"])
    if "mlp_processing" in sections:
        parts.append(sections["mlp_processing"])
    
    parts.append(sections["output_interpretation"])
    
    return "\n\n---\n\n".join(parts)


def get_architecture_summary(model_type: str = "hybrid") -> str:
    """
    Get a concise architecture summary for display.
    
    Args:
        model_type: Type of model
    
    Returns:
        ASCII architecture diagram
    """
    if model_type.lower() in ["hybrid", "hybridqmolnet"]:
        return """
┌─────────────────────────────────────────────────────────────┐
│                  Hybrid QMolNet Architecture                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
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
"""
    else:
        return """
┌─────────────────────────────────────────────────────────────┐
│              Classical GNN Classifier Architecture          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   SMILES ─▶ [Molecular Graph] ─▶ [GNN Encoder] ─▶ (32-dim) │
│                                        │                    │
│                                        ▼                    │
│                                 [MLP Classifier]            │
│                                        │                    │
│                                        ▼                    │
│                                  Prediction                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
"""


def get_pipeline_steps(model_type: str = "hybrid") -> list:
    """
    Get list of pipeline steps for display.
    
    Args:
        model_type: Type of model
    
    Returns:
        List of step descriptions
    """
    base_steps = [
        ("1. Input", "SMILES molecular string"),
        ("2. Parse", "Convert SMILES to RDKit molecule object"),
        ("3. Featurize", "Extract 145-dim atom features"),
        ("4. Graph Build", "Create PyTorch Geometric Data object"),
        ("5. GNN Encode", "Process graph through 3-layer GCN"),
    ]
    
    if model_type.lower() in ["hybrid", "hybridqmolnet"]:
        base_steps.extend([
            ("6. Compress", "Linear layer: 32-dim → 8-dim"),
            ("7. Quantum", "8-qubit VQC with 3 variational layers"),
            ("8. Classify", "Linear head produces class logits"),
            ("9. Output", "Softmax probability for each class"),
        ])
    else:
        base_steps.extend([
            ("6. Classify", "MLP head produces class logits"),
            ("7. Output", "Softmax probability for each class"),
        ])
    
    return base_steps


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Technical Explanation Generator Demo")
    print("=" * 60)
    
    explanation = generate_explanation(
        smiles="CC(=O)Nc1ccc(O)cc1",
        model_type="hybrid",
        n_atoms=11,
        n_bonds=11,
        prediction_prob=0.85,
        task_type="toxicity",
    )
    
    print("\nFull Explanation:")
    print("-" * 60)
    print(explanation["full_explanation"])
    
    print("\n" + "=" * 60)
    print("Architecture Summary:")
    print(get_architecture_summary("hybrid"))
    
    print("\nPipeline Steps:")
    for step, desc in get_pipeline_steps("hybrid"):
        print(f"  {step}: {desc}")
