#!/usr/bin/env python
"""
Hybrid QMolNet Streamlit Demo Application

A clean web interface for molecular property prediction using
the hybrid quantum-classical GNN model.

Features:
- SMILES input with molecule visualization
- Model selection (Hybrid / Classical)
- Side-by-side comparison mode
- Technical explanation panel
- Safe scientific language throughout

Usage:
    streamlit run app.py

Author: Hybrid QMolNet Team
"""

import os
import sys
import torch
import streamlit as st
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from utils.smiles_to_graph import smiles_to_graph, MoleculeGraphBuilder
from utils.formatters import (
    format_prediction_output,
    format_comparison_output,
    get_confidence_level,
    get_safe_disclaimer,
    get_model_description,
)
from utils.explanation import (
    generate_explanation,
    get_architecture_summary,
    get_pipeline_steps,
)

# Page configuration
st.set_page_config(
    page_title="Hybrid QMolNet - Molecular Property Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #546E7A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-label {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .confidence-high { color: #4CAF50; }
    .confidence-moderate { color: #FF9800; }
    .confidence-low { color: #f44336; }
    .model-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        background-color: #E3F2FD;
        color: #1565C0;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
    }
    .disclaimer {
        font-size: 0.85rem;
        color: #78909C;
        padding: 1rem;
        background-color: #ECEFF1;
        border-radius: 8px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# Cache model loading
@st.cache_resource
def load_models():
    """Load trained models with caching."""
    models = {}
    device = torch.device('cpu')
    
    # Model configurations
    model_config = {
        'node_feature_dim': 145,
        'gnn_hidden_dim': 64,
        'gnn_embedding_dim': 32,
        'gnn_layers': 3,
        'n_qubits': 8,
        'quantum_layers': 3,
        'num_classes': 2,
        'dropout': 0.2,
    }
    
    # Try to load hybrid model
    hybrid_checkpoint = None
    checkpoint_paths = [
        'outputs_bbbp/checkpoints/best.pt',
        'outputs/checkpoints/best.pt',
        'checkpoints/best.pt',
    ]
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            hybrid_checkpoint = path
            break
    
    try:
        from models.hybrid_model import HybridQMolNet
        from models.baselines import GNNClassifier
        
        # Initialize hybrid model
        hybrid_model = HybridQMolNet(
            node_feature_dim=model_config['node_feature_dim'],
            gnn_hidden_dim=model_config['gnn_hidden_dim'],
            gnn_embedding_dim=model_config['gnn_embedding_dim'],
            gnn_layers=model_config['gnn_layers'],
            n_qubits=model_config['n_qubits'],
            quantum_layers=model_config['quantum_layers'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout'],
            use_quantum=True,
        )
        
        if hybrid_checkpoint and os.path.exists(hybrid_checkpoint):
            checkpoint = torch.load(hybrid_checkpoint, map_location=device)
            if 'model_state_dict' in checkpoint:
                hybrid_model.load_state_dict(checkpoint['model_state_dict'])
            st.sidebar.success(f"✓ Loaded trained hybrid model")
        else:
            st.sidebar.warning("⚠ Using untrained hybrid model")
        
        hybrid_model.to(device)
        hybrid_model.eval()
        models['hybrid'] = hybrid_model
        
        # Initialize classical GNN model
        gnn_model = GNNClassifier(
            node_feature_dim=model_config['node_feature_dim'],
            gnn_hidden_dim=model_config['gnn_hidden_dim'],
            gnn_embedding_dim=model_config['gnn_embedding_dim'],
            gnn_layers=model_config['gnn_layers'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout'],
        )
        gnn_model.to(device)
        gnn_model.eval()
        models['classical'] = gnn_model
        st.sidebar.info("ℹ Classical GNN baseline loaded (untrained)")
        
    except Exception as e:
        st.sidebar.error(f"Error loading models: {str(e)}")
        return None
    
    return models


def visualize_molecule(smiles: str):
    """Create molecule visualization from SMILES."""
    try:
        from visualization.molecule_viz import smiles_to_image
        img = smiles_to_image(smiles, size=(350, 350))
        return img
    except Exception as e:
        st.warning(f"Could not visualize molecule: {str(e)}")
        return None


def predict_with_model(smiles: str, model, model_type: str):
    """Run prediction with a model."""
    try:
        # Convert SMILES to graph
        graph_data = smiles_to_graph(smiles)
        if graph_data is None:
            return None, "Invalid SMILES string"
        
        # Create batch
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([graph_data])
        
        # Run inference
        device = torch.device('cpu')
        with torch.no_grad():
            logits = model.forward_batch(batch.to(device))
            probs = torch.softmax(logits, dim=1)
            positive_prob = probs[0, 1].item()
        
        return positive_prob, None
        
    except Exception as e:
        return None, str(e)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🧬 Hybrid QMolNet</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Hybrid Quantum-Classical Neural Network for Molecular Property Prediction</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Load models
    models = load_models()
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    model_choice = st.sidebar.radio(
        "Select Model",
        ["Hybrid Quantum-Classical", "Classical GNN (Baseline)"],
        index=0,
        help="Choose between the hybrid quantum-classical model or classical GNN baseline"
    )
    
    # Comparison mode
    comparison_mode = st.sidebar.checkbox(
        "Enable Comparison Mode",
        value=False,
        help="Show predictions from both models side-by-side"
    )
    
    # Task type
    st.sidebar.subheader("Prediction Task")
    task_type = st.sidebar.selectbox(
        "Property to Predict",
        ["bbbp", "toxicity", "binary"],
        format_func=lambda x: {
            "bbbp": "Blood-Brain Barrier Permeability",
            "toxicity": "Molecular Toxicity",
            "binary": "Binary Classification"
        }[x]
    )
    
    # Show explanation toggle
    show_explanation = st.sidebar.checkbox(
        "Show Technical Explanation",
        value=False,
        help="Display detailed technical explanation of the prediction pipeline"
    )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input")
        
        # SMILES input
        smiles_input = st.text_input(
            "Enter SMILES String",
            value="CC(=O)Nc1ccc(O)cc1",
            help="Enter a valid SMILES molecular representation"
        )
        
        # Example molecules
        st.markdown("**Example Molecules:**")
        example_cols = st.columns(3)
        examples = [
            ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
            ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
            ("Ethanol", "CCO"),
        ]
        
        for i, (name, smiles) in enumerate(examples):
            with example_cols[i]:
                if st.button(name, key=f"example_{i}"):
                    smiles_input = smiles
                    st.experimental_rerun()
        
        # Predict button
        predict_clicked = st.button("🔮 Predict", type="primary")
        
        # Molecule visualization
        st.subheader("🔬 Molecule Structure")
        mol_img = visualize_molecule(smiles_input)
        if mol_img:
            st.image(mol_img, caption=f"SMILES: {smiles_input}")
        else:
            st.warning("Could not render molecule structure")
    
    with col2:
        st.subheader("📊 Prediction Results")
        
        if predict_clicked and models:
            # Determine which model to use
            model_key = 'hybrid' if "Hybrid" in model_choice else 'classical'
            model_type_str = 'hybrid' if "Hybrid" in model_choice else 'gnn'
            
            if comparison_mode:
                # Run both models
                st.markdown("### Model Comparison")
                
                result_cols = st.columns(2)
                
                # Hybrid prediction
                with result_cols[0]:
                    st.markdown("**Hybrid Quantum-Classical**")
                    if 'hybrid' in models:
                        prob, error = predict_with_model(smiles_input, models['hybrid'], 'hybrid')
                        if error:
                            st.error(f"Error: {error}")
                        elif prob is not None:
                            result = format_prediction_output(prob, task_type, 'hybrid', smiles_input)
                            level, level_class = get_confidence_level(result['confidence_value'])
                            
                            st.metric(
                                label="Prediction",
                                value=result['label'],
                                delta=f"{result['confidence']} confidence"
                            )
                            st.progress(prob)
                    else:
                        st.warning("Hybrid model not available")
                
                # Classical prediction
                with result_cols[1]:
                    st.markdown("**Classical GNN Baseline**")
                    if 'classical' in models:
                        prob, error = predict_with_model(smiles_input, models['classical'], 'gnn')
                        if error:
                            st.error(f"Error: {error}")
                        elif prob is not None:
                            result = format_prediction_output(prob, task_type, 'gnn', smiles_input)
                            level, level_class = get_confidence_level(result['confidence_value'])
                            
                            st.metric(
                                label="Prediction",
                                value=result['label'],
                                delta=f"{result['confidence']} confidence"
                            )
                            st.progress(prob)
                    else:
                        st.warning("Classical model not available")
            
            else:
                # Single model prediction
                if model_key in models:
                    prob, error = predict_with_model(smiles_input, models[model_key], model_type_str)
                    
                    if error:
                        st.error(f"Prediction Error: {error}")
                    elif prob is not None:
                        result = format_prediction_output(prob, task_type, model_type_str, smiles_input)
                        level, level_class = get_confidence_level(result['confidence_value'])
                        
                        # Display model being used
                        st.markdown(f'<div class="model-badge">🤖 {result["model_name"]}</div>', unsafe_allow_html=True)
                        
                        # Main prediction display
                        st.markdown(f"""
                        <div class="prediction-box">
                            <div class="prediction-label">{result['label']}</div>
                            <div>Confidence: {result['confidence']} ({level})</div>
                            <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                                {result['task_name']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Progress bar
                        st.progress(prob)
                        st.caption(f"Probability of positive class: {prob:.4f}")
                        
                        # Model description
                        with st.expander("ℹ️ About this Model"):
                            st.markdown(get_model_description(model_type_str))
                else:
                    st.warning(f"Model '{model_key}' not available")
        
        elif not models:
            st.info("⏳ Loading models... Please wait.")
        else:
            st.info("👆 Enter a SMILES string and click Predict to start")
    
    # Technical explanation section
    if show_explanation and smiles_input:
        st.markdown("---")
        st.subheader("🔍 Technical Explanation")
        
        # Get graph info
        graph_data = smiles_to_graph(smiles_input)
        n_atoms = graph_data.num_nodes if graph_data else None
        n_bonds = graph_data.num_edges // 2 if graph_data else None
        
        model_type = 'hybrid' if "Hybrid" in model_choice else 'gnn'
        explanation = generate_explanation(
            smiles=smiles_input,
            model_type=model_type,
            n_atoms=n_atoms,
            n_bonds=n_bonds,
            task_type=task_type,
        )
        
        # Show explanation tabs
        exp_tabs = st.tabs(["Graph Structure", "GNN Processing", "Quantum Transform", "Full Pipeline"])
        
        with exp_tabs[0]:
            st.markdown(explanation.get('graph_structure', 'N/A'))
        
        with exp_tabs[1]:
            st.markdown(explanation.get('gnn_processing', 'N/A'))
        
        with exp_tabs[2]:
            if 'quantum_transform' in explanation:
                st.markdown(explanation['quantum_transform'])
            else:
                st.info("Quantum transform is only used in the Hybrid model")
        
        with exp_tabs[3]:
            pipeline_name = 'hybrid_inference' if model_type == 'hybrid' else 'classical_inference'
            st.markdown(explanation.get(pipeline_name, 'N/A'))
        
        # Architecture diagram
        with st.expander("📐 Architecture Diagram"):
            st.code(get_architecture_summary(model_type), language=None)
    
    # Disclaimer
    st.markdown("---")
    st.markdown(f'<div class="disclaimer">⚠️ {get_safe_disclaimer()}</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #9E9E9E; font-size: 0.85rem;">
            <strong>Hybrid QMolNet</strong> | Quantum-Classical Neural Network for Drug Discovery<br>
            Built with PyTorch, PennyLane, PyTorch Geometric, and Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
