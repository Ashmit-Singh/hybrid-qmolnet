"""
Data Loading and Dataset Management Module

Handles dataset loading, preprocessing, train/val/test splits,
and DataLoader creation for molecular property prediction.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Dict, Any
from tqdm import tqdm

from .smiles_to_graph import MoleculeGraphBuilder, compute_molecular_descriptors


class MoleculeDataset(Dataset):
    """
    PyTorch Dataset for molecular graphs.
    
    Stores pre-computed PyG Data objects for efficient training.
    Supports both graph-based and descriptor-based representations.
    """
    
    def __init__(
        self,
        smiles_list: List[str],
        labels: List[int],
        graph_builder: Optional[MoleculeGraphBuilder] = None,
        compute_descriptors: bool = False,
    ):
        """
        Initialize the molecule dataset.
        
        Args:
            smiles_list: List of SMILES strings
            labels: List of binary labels (0 or 1)
            graph_builder: MoleculeGraphBuilder instance for graph conversion
            compute_descriptors: Whether to compute molecular descriptors
        """
        self.smiles_list = smiles_list
        self.labels = labels
        self.graph_builder = graph_builder or MoleculeGraphBuilder()
        self.compute_descriptors = compute_descriptors
        
        # Pre-compute graphs and descriptors
        self.graphs: List[Optional[Data]] = []
        self.descriptors: List[Optional[np.ndarray]] = []
        self.valid_indices: List[int] = []
        
        self._precompute()
    
    def _precompute(self):
        """Pre-compute all molecular representations."""
        print(f"Processing {len(self.smiles_list)} molecules...")
        
        for idx, (smiles, label) in enumerate(tqdm(
            zip(self.smiles_list, self.labels),
            total=len(self.smiles_list),
            desc="Building graphs"
        )):
            # Build graph
            graph = self.graph_builder.build(smiles, label=label)
            
            # Compute descriptors if requested
            desc = None
            if self.compute_descriptors:
                desc = compute_molecular_descriptors(smiles)
            
            # Only keep valid molecules
            if graph is not None:
                self.graphs.append(graph)
                self.descriptors.append(desc)
                self.valid_indices.append(idx)
        
        print(f"Successfully processed {len(self.valid_indices)}/{len(self.smiles_list)} molecules")
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with 'graph' and optionally 'descriptors' keys
        """
        item = {'graph': self.graphs[idx]}
        
        if self.compute_descriptors and self.descriptors[idx] is not None:
            item['descriptors'] = torch.tensor(self.descriptors[idx], dtype=torch.float)
        
        return item
    
    @property
    def node_feature_dim(self) -> int:
        """Get node feature dimension."""
        return self.graph_builder.node_feature_dim
    
    @property
    def descriptor_dim(self) -> int:
        """Get descriptor dimension."""
        if self.descriptors and self.descriptors[0] is not None:
            return len(self.descriptors[0])
        return 10  # Default descriptor count


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for batching molecular data.
    
    Args:
        batch: List of sample dictionaries
    
    Returns:
        Batched dictionary with PyG Batch for graphs
    """
    graphs = [item['graph'] for item in batch]
    batched = {'graph': Batch.from_data_list(graphs)}
    
    if 'descriptors' in batch[0]:
        descriptors = torch.stack([item['descriptors'] for item in batch])
        batched['descriptors'] = descriptors
    
    return batched


def generate_synthetic_dataset(
    n_samples: int = 1000,
    seed: int = 42
) -> Tuple[List[str], List[int]]:
    """
    Generate a synthetic dataset for demonstration.
    
    Creates molecules with property labels based on structural features.
    Uses common drug-like scaffolds and functional groups.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (smiles_list, labels)
    """
    np.random.seed(seed)
    
    # Define molecule templates with associated properties
    # Class 0: Hydrophobic/non-polar molecules
    class_0_templates = [
        'CCCCCC',           # Hexane
        'CCCCCCC',          # Heptane
        'c1ccccc1',          # Benzene
        'Cc1ccccc1',         # Toluene
        'c1ccc2ccccc2c1',    # Naphthalene
        'CCc1ccccc1',        # Ethylbenzene
        'Cc1ccc(C)cc1',      # Xylene
        'c1ccc(cc1)c2ccccc2', # Biphenyl
        'C1CCCCC1',          # Cyclohexane
        'CC(C)C',            # Isobutane
        'CCCCCCCC',          # Octane
        'CCC(C)C',           # Isopentane
    ]
    
    # Class 1: Polar/hydrogen-bonding molecules
    class_1_templates = [
        'CCO',              # Ethanol
        'CCCO',             # Propanol
        'CC(=O)O',          # Acetic acid
        'CC(=O)N',          # Acetamide
        'c1ccc(O)cc1',      # Phenol
        'CC(=O)Nc1ccccc1',  # Acetanilide
        'OCC(O)CO',         # Glycerol
        'c1ccc(N)cc1',      # Aniline
        'OC(=O)c1ccccc1',   # Benzoic acid
        'NCCc1ccc(O)cc1',   # Tyramine
        'CC(O)C(=O)O',      # Lactic acid
        'NCCO',             # Ethanolamine
    ]
    
    smiles_list = []
    labels = []
    
    samples_per_class = n_samples // 2
    
    # Generate class 0 samples
    for _ in range(samples_per_class):
        template = np.random.choice(class_0_templates)
        smiles_list.append(template)
        labels.append(0)
    
    # Generate class 1 samples
    for _ in range(n_samples - samples_per_class):
        template = np.random.choice(class_1_templates)
        smiles_list.append(template)
        labels.append(1)
    
    # Shuffle the dataset
    indices = np.random.permutation(len(smiles_list))
    smiles_list = [smiles_list[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return smiles_list, labels


def load_dataset(
    data_path: Optional[str] = None,
    smiles_col: str = 'smiles',
    label_col: str = 'label',
    n_samples: int = 1000,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    """
    Load or generate molecular dataset.
    
    If data_path is provided, loads from CSV file.
    Otherwise, generates synthetic dataset for demonstration.
    
    Args:
        data_path: Path to CSV file with SMILES and labels
        smiles_col: Column name for SMILES strings
        label_col: Column name for labels
        n_samples: Number of samples for synthetic dataset
        seed: Random seed
    
    Returns:
        Tuple of (smiles_list, labels)
    """
    if data_path is not None and os.path.exists(data_path):
        print(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        smiles_list = df[smiles_col].tolist()
        labels = df[label_col].tolist()
    else:
        print("Generating synthetic dataset for demonstration...")
        smiles_list, labels = generate_synthetic_dataset(n_samples, seed)
    
    print(f"Dataset size: {len(smiles_list)} molecules")
    print(f"Class distribution: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    return smiles_list, labels


def create_data_loaders(
    smiles_list: List[str],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    seed: int = 42,
    compute_descriptors: bool = False,
    use_cache: bool = True,
    num_workers: int = 0,
) -> Tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader, MoleculeDataset]:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        smiles_list: List of SMILES strings
        labels: List of labels
        train_ratio: Training set proportion
        val_ratio: Validation set proportion
        test_ratio: Test set proportion
        batch_size: Batch size for DataLoaders
        seed: Random seed for reproducibility
        compute_descriptors: Whether to compute molecular descriptors
        use_cache: Cache preprocessed graphs to disk
        num_workers: DataLoader workers (0 for Windows compatibility)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, full_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # First split: train+val vs test
    train_val_smiles, test_smiles, train_val_labels, test_labels = train_test_split(
        smiles_list, labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels
    )
    
    # Second split: train vs val
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_smiles, val_smiles, train_labels, val_labels = train_test_split(
        train_val_smiles, train_val_labels,
        test_size=relative_val_ratio,
        random_state=seed,
        stratify=train_val_labels
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_smiles)} molecules")
    print(f"  Val:   {len(val_smiles)} molecules")
    print(f"  Test:  {len(test_smiles)} molecules")
    
    # Create datasets
    graph_builder = MoleculeGraphBuilder()
    
    print("\nProcessing training set...")
    train_dataset = MoleculeDataset(
        train_smiles, train_labels,
        graph_builder=graph_builder,
        compute_descriptors=compute_descriptors
    )
    
    print("\nProcessing validation set...")
    val_dataset = MoleculeDataset(
        val_smiles, val_labels,
        graph_builder=graph_builder,
        compute_descriptors=compute_descriptors
    )
    
    print("\nProcessing test set...")
    test_dataset = MoleculeDataset(
        test_smiles, test_labels,
        graph_builder=graph_builder,
        compute_descriptors=compute_descriptors
    )
    
    # Create DataLoaders
    train_loader = PyGDataLoader(
        [item['graph'] for item in train_dataset],
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = PyGDataLoader(
        [item['graph'] for item in val_dataset],
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = PyGDataLoader(
        [item['graph'] for item in test_dataset],
        batch_size=batch_size,
        shuffle=False
    )
    
    # Also return full dataset for descriptor-based models
    full_dataset = MoleculeDataset(
        smiles_list, labels,
        graph_builder=graph_builder,
        compute_descriptors=compute_descriptors
    )
    
    return train_loader, val_loader, test_loader, full_dataset


if __name__ == "__main__":
    # Demo usage
    smiles_list, labels = load_dataset(n_samples=100)
    
    train_loader, val_loader, test_loader, dataset = create_data_loaders(
        smiles_list, labels,
        batch_size=16,
        compute_descriptors=True
    )
    
    print(f"\nNode feature dimension: {dataset.node_feature_dim}")
    print(f"Descriptor dimension: {dataset.descriptor_dim}")
    
    # Test batch
    for batch in train_loader:
        print(f"\nBatch info:")
        print(f"  Batch size: {batch.num_graphs}")
        print(f"  Total nodes: {batch.x.shape[0]}")
        print(f"  Node features: {batch.x.shape}")
        print(f"  Labels: {batch.y}")
        break
