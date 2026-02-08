"""
Unit Tests for SMILES Parsing and Graph Building

Tests the molecular graph conversion pipeline.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.smiles_to_graph import (
    smiles_to_graph,
    MoleculeGraphBuilder,
    get_atom_features,
    compute_molecular_descriptors,
)


class TestSmilesParser:
    """Tests for SMILES parsing functionality."""
    
    def test_valid_smiles_ethanol(self):
        """Test parsing a simple valid SMILES (ethanol)."""
        data = smiles_to_graph('CCO', label=0)
        
        assert data is not None
        assert data.x.shape[0] == 3  # 3 atoms: C, C, O
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == 4  # 2 bonds * 2 (bidirectional)
        assert data.y.item() == 0
    
    def test_valid_smiles_benzene(self):
        """Test parsing benzene (cyclic aromatic)."""
        data = smiles_to_graph('c1ccccc1', label=1)
        
        assert data is not None
        assert data.x.shape[0] == 6  # 6 carbon atoms
        # Benzene has 6 bonds * 2 = 12 directed edges
        assert data.edge_index.shape[1] == 12
    
    def test_invalid_smiles(self):
        """Test that invalid SMILES returns None."""
        data = smiles_to_graph('not_a_valid_smiles')
        assert data is None
    
    def test_empty_smiles(self):
        """Test that empty SMILES returns None."""
        data = smiles_to_graph('')
        assert data is None
    
    def test_single_atom(self):
        """Test single atom molecule."""
        data = smiles_to_graph('[He]')  # Helium
        
        assert data is not None
        assert data.x.shape[0] == 1
        assert data.edge_index.shape[1] == 0  # No bonds


class TestMoleculeGraphBuilder:
    """Tests for MoleculeGraphBuilder class."""
    
    def test_builder_initialization(self):
        """Test builder creates correctly."""
        builder = MoleculeGraphBuilder()
        assert builder is not None
        assert builder.node_feature_dim > 0
    
    def test_feature_dimension_consistency(self):
        """Test that feature dimensions are consistent across molecules."""
        builder = MoleculeGraphBuilder()
        
        data1 = builder.build('CCO')
        data2 = builder.build('c1ccccc1')
        data3 = builder.build('CC(=O)O')
        
        assert data1.x.shape[1] == data2.x.shape[1] == data3.x.shape[1]
        assert data1.x.shape[1] == builder.node_feature_dim
    
    def test_edge_bidirectionality(self):
        """Test that edges are bidirectional."""
        data = smiles_to_graph('CC')  # Ethane
        
        edge_index = data.edge_index
        assert edge_index.shape[1] == 2  # 1 bond * 2 directions
        
        # Check both directions exist
        edges_set = set()
        for i in range(edge_index.shape[1]):
            edges_set.add((edge_index[0, i].item(), edge_index[1, i].item()))
        
        assert (0, 1) in edges_set
        assert (1, 0) in edges_set
    
    def test_label_assignment(self):
        """Test that labels are correctly assigned."""
        data0 = smiles_to_graph('CCO', label=0)
        data1 = smiles_to_graph('CCC', label=1)
        
        assert data0.y.item() == 0
        assert data1.y.item() == 1


class TestMolecularDescriptors:
    """Tests for molecular descriptor computation."""
    
    def test_descriptor_computation(self):
        """Test that descriptors are computed for valid SMILES."""
        desc = compute_molecular_descriptors('CCO')
        
        assert desc is not None
        assert len(desc) == 10  # Expected number of descriptors
    
    def test_descriptor_invalid_smiles(self):
        """Test that invalid SMILES returns None."""
        desc = compute_molecular_descriptors('invalid')
        assert desc is None
    
    def test_descriptor_values_reasonable(self):
        """Test that descriptor values are reasonable."""
        desc = compute_molecular_descriptors('c1ccccc1')  # Benzene
        
        # Molecular weight should be around 78
        assert 70 < desc[0] < 85
        
        # LogP should be positive for benzene
        assert desc[1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
