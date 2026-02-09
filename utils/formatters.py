"""
Prediction Output Formatter Module

Converts raw model outputs into human-readable text with safe scientific language.
Avoids terms like "quantum advantage" or "guaranteed accuracy".
Uses careful, scientifically accurate phrasing.
"""

from typing import Dict, Optional, Tuple


# Task type configurations
TASK_CONFIGS = {
    "toxicity": {
        "name": "Toxicity Prediction",
        "positive_label": "Likely Toxic",
        "negative_label": "Likely Non-Toxic",
        "positive_class": 1,
        "description": "Estimated toxicity based on molecular structure",
    },
    "bbbp": {
        "name": "Blood-Brain Barrier Permeability",
        "positive_label": "Likely Permeable",
        "negative_label": "Likely Non-Permeable",
        "positive_class": 1,
        "description": "Estimated BBB permeability based on molecular structure",
    },
    "solubility": {
        "name": "Aqueous Solubility",
        "positive_label": "Likely Soluble",
        "negative_label": "Likely Insoluble",
        "positive_class": 1,
        "description": "Estimated aqueous solubility based on molecular structure",
    },
    "binary": {
        "name": "Binary Classification",
        "positive_label": "Positive Class",
        "negative_label": "Negative Class",
        "positive_class": 1,
        "description": "Binary molecular property prediction",
    },
}

# Model name mappings for display
MODEL_DISPLAY_NAMES = {
    "hybrid": "Hybrid Quantum-Classical GNN",
    "gnn": "Classical GNN",
    "mlp": "Descriptor-based MLP",
    "HybridQMolNet": "Hybrid Quantum-Classical GNN",
    "GNNClassifier": "Classical GNN",
    "DescriptorMLP": "Descriptor-based MLP",
}


def format_prediction_output(
    probability: float,
    task_type: str = "binary",
    model_name: str = "hybrid",
    smiles: Optional[str] = None,
) -> Dict[str, str]:
    """
    Format raw prediction probability into human-readable output.
    
    Uses safe scientific language - avoids exaggerated claims.
    
    Args:
        probability: Predicted probability for the positive class (0-1)
        task_type: Type of prediction task (toxicity, bbbp, solubility, binary)
        model_name: Name of the model used for prediction
        smiles: Optional SMILES string of the molecule
    
    Returns:
        Dictionary with formatted output fields:
        - label: Human-readable prediction label
        - confidence: Confidence percentage string
        - confidence_value: Raw confidence value (0-100)
        - task_name: Display name of the task
        - model_name: Display name of the model
        - description: Task description
        - probability: Original probability value
        - predicted_class: Predicted class (0 or 1)
    """
    # Get task configuration
    config = TASK_CONFIGS.get(task_type, TASK_CONFIGS["binary"])
    
    # Determine prediction and confidence
    if probability >= 0.5:
        predicted_class = config["positive_class"]
        label = config["positive_label"]
        confidence = probability
    else:
        predicted_class = 1 - config["positive_class"]
        label = config["negative_label"]
        confidence = 1 - probability
    
    # Format confidence as percentage
    confidence_pct = confidence * 100
    confidence_str = f"{confidence_pct:.1f}%"
    
    # Get display model name
    display_model = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    
    # Build output dictionary
    output = {
        "label": label,
        "confidence": confidence_str,
        "confidence_value": confidence_pct,
        "task_name": config["name"],
        "model_name": display_model,
        "description": config["description"],
        "probability": probability,
        "predicted_class": predicted_class,
    }
    
    if smiles:
        output["smiles"] = smiles
    
    return output


def format_comparison_output(
    hybrid_prob: float,
    classical_prob: float,
    task_type: str = "binary",
) -> Dict[str, Dict[str, str]]:
    """
    Format side-by-side comparison of hybrid and classical model predictions.
    
    Args:
        hybrid_prob: Probability from hybrid quantum-classical model
        classical_prob: Probability from classical GNN model
        task_type: Type of prediction task
    
    Returns:
        Dictionary with 'hybrid' and 'classical' formatted outputs
    """
    return {
        "hybrid": format_prediction_output(
            hybrid_prob, task_type, "hybrid"
        ),
        "classical": format_prediction_output(
            classical_prob, task_type, "gnn"
        ),
    }


def get_confidence_level(confidence_value: float) -> Tuple[str, str]:
    """
    Get a qualitative confidence level description.
    
    Args:
        confidence_value: Confidence percentage (0-100)
    
    Returns:
        Tuple of (level_name, css_color_class)
    """
    if confidence_value >= 90:
        return "High", "high-confidence"
    elif confidence_value >= 70:
        return "Moderate", "moderate-confidence"
    elif confidence_value >= 55:
        return "Low", "low-confidence"
    else:
        return "Very Low", "very-low-confidence"


def format_batch_predictions(
    probabilities: list,
    smiles_list: list,
    task_type: str = "binary",
    model_name: str = "hybrid",
) -> list:
    """
    Format batch predictions for multiple molecules.
    
    Args:
        probabilities: List of prediction probabilities
        smiles_list: List of SMILES strings
        task_type: Type of prediction task
        model_name: Model name
    
    Returns:
        List of formatted prediction dictionaries
    """
    results = []
    for prob, smiles in zip(probabilities, smiles_list):
        result = format_prediction_output(
            prob, task_type, model_name, smiles
        )
        results.append(result)
    return results


def get_safe_disclaimer() -> str:
    """
    Return a safe scientific disclaimer for predictions.
    
    Returns:
        Disclaimer text string
    """
    return (
        "Note: These predictions are computational estimates based on "
        "molecular structure analysis. They should not be used as the "
        "sole basis for any medical, pharmaceutical, or safety decisions. "
        "Always consult with qualified professionals and conduct "
        "appropriate experimental validation."
    )


def get_model_description(model_name: str) -> str:
    """
    Get a safe description of the model architecture.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Description string with safe scientific language
    """
    descriptions = {
        "hybrid": (
            "This prediction was made using a hybrid quantum-classical "
            "neural network that combines Graph Neural Networks (GNN) for "
            "molecular feature extraction with a Variational Quantum Circuit "
            "(VQC) for feature transformation. The model was trained on "
            "molecular property data to estimate property values."
        ),
        "gnn": (
            "This prediction was made using a classical Graph Neural Network "
            "(GNN) that processes molecular structures as graphs, with atoms "
            "as nodes and chemical bonds as edges. The model was trained on "
            "molecular property data to estimate property values."
        ),
        "mlp": (
            "This prediction was made using a Multi-Layer Perceptron (MLP) "
            "that processes pre-computed molecular descriptors. The model "
            "was trained on molecular property data to estimate property values."
        ),
    }
    
    # Normalize model name
    normalized = model_name.lower()
    if "hybrid" in normalized or "quantum" in normalized:
        return descriptions["hybrid"]
    elif "gnn" in normalized:
        return descriptions["gnn"]
    elif "mlp" in normalized or "descriptor" in normalized:
        return descriptions["mlp"]
    else:
        return descriptions["hybrid"]


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Prediction Output Formatter Demo")
    print("=" * 60)
    
    # Test toxicity prediction
    result = format_prediction_output(
        probability=0.85,
        task_type="toxicity",
        model_name="hybrid",
        smiles="CC(=O)Nc1ccc(O)cc1"
    )
    
    print("\nToxicity Prediction:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Test BBB permeability
    result = format_prediction_output(
        probability=0.35,
        task_type="bbbp",
        model_name="gnn",
    )
    
    print("\nBBB Permeability Prediction:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Test comparison
    comparison = format_comparison_output(
        hybrid_prob=0.78,
        classical_prob=0.72,
        task_type="toxicity",
    )
    
    print("\nModel Comparison:")
    print("  Hybrid:", comparison["hybrid"]["label"], 
          f"({comparison['hybrid']['confidence']})")
    print("  Classical:", comparison["classical"]["label"],
          f"({comparison['classical']['confidence']})")
    
    print("\n" + get_safe_disclaimer())
