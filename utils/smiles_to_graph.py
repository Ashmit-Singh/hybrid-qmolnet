"""
SMILES to Molecular Graph Conversion Module

Converts SMILES strings to PyTorch Geometric Data objects using RDKit.
Node features include atomic properties; edges are bidirectional bonds.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import Optional, List, Tuple


# Atom feature encoding constants
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),  # H to Og
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
}


def one_hot_encode(value, allowable_set: List, include_unknown: bool = True) -> List[int]:
    """
    One-hot encode a value given an allowable set.
    
    Args:
        value: Value to encode
        allowable_set: List of allowed values
        include_unknown: If True, add an extra bit for unknown values
    
    Returns:
        One-hot encoded list
    """
    encoding = [0] * len(allowable_set)
    
    try:
        idx = allowable_set.index(value)
        encoding[idx] = 1
    except ValueError:
        if include_unknown:
            encoding.append(1)
        else:
            # Use last valid category as fallback
            encoding[-1] = 1
    
    if include_unknown and len(encoding) == len(allowable_set):
        encoding.append(0)
    
    return encoding


def get_atom_features(atom) -> List[float]:
    """
    Extract comprehensive features for a single atom.
    
    Features include:
    - Atomic number (one-hot, 118 + 1 = 119 dims)
    - Degree (one-hot, 7 + 1 = 8 dims)
    - Formal charge (one-hot, 7 + 1 = 8 dims)
    - Hybridization (one-hot, 6 + 1 = 7 dims)
    - Is aromatic (binary, 1 dim)
    - Number of hydrogens (normalized, 1 dim)
    - Is in ring (binary, 1 dim)
    
    Total: 119 + 8 + 8 + 7 + 1 + 1 + 1 = 145 dimensions
    
    Args:
        atom: RDKit Atom object
    
    Returns:
        List of atom features
    """
    features = []
    
    # Atomic number (one-hot encoded)
    features.extend(one_hot_encode(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']))
    
    # Degree (number of directly bonded neighbors)
    features.extend(one_hot_encode(atom.GetDegree(), ATOM_FEATURES['degree']))
    
    # Formal charge
    features.extend(one_hot_encode(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']))
    
    # Hybridization state
    features.extend(one_hot_encode(atom.GetHybridization(), ATOM_FEATURES['hybridization']))
    
    # Aromaticity (binary)
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    
    # Number of hydrogens (normalized by max typical value of 4)
    features.append(atom.GetTotalNumHs() / 4.0)
    
    # Is in ring (binary)
    features.append(1.0 if atom.IsInRing() else 0.0)
    
    return features


def get_bond_features(bond) -> List[float]:
    """
    Extract features for a single bond.
    
    Features include:
    - Bond type (single, double, triple, aromatic) - one-hot
    - Is conjugated (binary)
    - Is in ring (binary)
    
    Args:
        bond: RDKit Bond object
    
    Returns:
        List of bond features
    """
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    
    features = one_hot_encode(bond.GetBondType(), bond_types, include_unknown=False)
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    features.append(1.0 if bond.IsInRing() else 0.0)
    
    return features


class MoleculeGraphBuilder:
    """
    Builder class for converting molecules to graph representations.
    
    Handles SMILES parsing, feature extraction, and PyG Data creation.
    Provides consistent feature dimensions across all molecules.
    """
    
    def __init__(self, add_hydrogens: bool = False):
        """
        Initialize the molecule graph builder.
        
        Args:
            add_hydrogens: Whether to explicitly add hydrogen atoms
        """
        self.add_hydrogens = add_hydrogens
        self._node_feature_dim = None
        self._edge_feature_dim = None
    
    @property
    def node_feature_dim(self) -> int:
        """Return the dimension of node features."""
        if self._node_feature_dim is None:
            # Calculate feature dimension from dummy atom
            dummy_mol = Chem.MolFromSmiles('C')
            dummy_atom = dummy_mol.GetAtomWithIdx(0)
            self._node_feature_dim = len(get_atom_features(dummy_atom))
        return self._node_feature_dim
    
    @property 
    def edge_feature_dim(self) -> int:
        """Return the dimension of edge features."""
        if self._edge_feature_dim is None:
            # Calculate from dummy bond
            dummy_mol = Chem.MolFromSmiles('CC')
            dummy_bond = dummy_mol.GetBondWithIdx(0)
            self._edge_feature_dim = len(get_bond_features(dummy_bond))
        return self._edge_feature_dim
    
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES string to RDKit Mol object.
        
        Args:
            smiles: SMILES string representation of molecule
        
        Returns:
            RDKit Mol object or None if parsing fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                return None
            
            if self.add_hydrogens:
                mol = Chem.AddHs(mol)
            
            return mol
        except Exception:
            return None
    
    def mol_to_graph(self, mol: Chem.Mol, label: Optional[int] = None) -> Data:
        """
        Convert RDKit Mol object to PyTorch Geometric Data.
        
        Args:
            mol: RDKit Mol object
            label: Optional class label for the molecule
        
        Returns:
            PyTorch Geometric Data object
        """
        # Extract node features for all atoms
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(get_atom_features(atom))
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Extract edge indices and features (bidirectional)
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            bond_feat = get_bond_features(bond)
            
            # Add both directions for bidirectional edges
            edge_indices.extend([[i, j], [j, i]])
            edge_features.extend([bond_feat, bond_feat])
        
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            # Handle single-atom molecules with no bonds
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, self.edge_feature_dim), dtype=torch.float)
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Add label if provided
        if label is not None:
            data.y = torch.tensor([label], dtype=torch.long)
        
        return data
    
    def build(self, smiles: str, label: Optional[int] = None) -> Optional[Data]:
        """
        Convert SMILES string directly to PyG Data object.
        
        Args:
            smiles: SMILES string
            label: Optional class label
        
        Returns:
            PyTorch Geometric Data object or None if conversion fails
        """
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        return self.mol_to_graph(mol, label)


def smiles_to_graph(smiles: str, label: Optional[int] = None) -> Optional[Data]:
    """
    Convenience function to convert SMILES to graph.
    
    Args:
        smiles: SMILES string representation
        label: Optional class label
    
    Returns:
        PyTorch Geometric Data object or None if parsing fails
    
    Example:
        >>> data = smiles_to_graph('CCO', label=1)
        >>> print(data.x.shape)  # Node features
        >>> print(data.edge_index.shape)  # Edge connectivity
    """
    builder = MoleculeGraphBuilder()
    return builder.build(smiles, label)


def compute_molecular_descriptors(smiles: str) -> Optional[np.ndarray]:
    """
    Compute classical molecular descriptors for baseline models.
    
    Descriptors include:
    - Molecular weight
    - LogP (lipophilicity)
    - Number of hydrogen bond donors/acceptors
    - TPSA (topological polar surface area)
    - Number of rotatable bonds
    - Number of aromatic rings
    - Fraction of sp3 carbons
    
    Args:
        smiles: SMILES string
    
    Returns:
        NumPy array of descriptors or None if computation fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Define descriptors to compute
        descriptor_funcs = [
            Descriptors.MolWt,
            Descriptors.MolLogP,
            Descriptors.NumHDonors,
            Descriptors.NumHAcceptors,
            Descriptors.TPSA,
            Descriptors.NumRotatableBonds,
            Descriptors.NumAromaticRings,
            Descriptors.FractionCSP3,
            Descriptors.HeavyAtomCount,
            Descriptors.RingCount,
        ]
        
        values = []
        for func in descriptor_funcs:
            try:
                val = func(mol)
                # Ensure it's a single float, not a sequence or None
                if val is None:
                    values.append(0.0)
                else:
                    values.append(float(val))
            except:
                values.append(0.0)
        
        return np.array(values, dtype=np.float32)
    except Exception:
        return None


if __name__ == "__main__":
    # Demo and testing
    test_smiles = [
        ('CCO', 'Ethanol'),
        ('c1ccccc1', 'Benzene'),
        ('CC(=O)O', 'Acetic acid'),
        ('CC(=O)Nc1ccc(O)cc1', 'Paracetamol'),
        ('invalid_smiles', 'Invalid'),
    ]
    
    builder = MoleculeGraphBuilder()
    print(f"Node feature dimension: {builder.node_feature_dim}")
    print(f"Edge feature dimension: {builder.edge_feature_dim}")
    print()
    
    for smiles, name in test_smiles:
        data = builder.build(smiles, label=1)
        if data is not None:
            print(f"{name} ({smiles}):")
            print(f"  Atoms: {data.x.shape[0]}, Bonds: {data.edge_index.shape[1] // 2}")
            print(f"  Node features shape: {data.x.shape}")
            print(f"  Edge index shape: {data.edge_index.shape}")
        else:
            print(f"{name} ({smiles}): Failed to parse")
        print()
