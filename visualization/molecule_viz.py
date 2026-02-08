"""
Molecule Visualization Module

Visualize molecular structures and their graph representations.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from typing import Optional, Tuple, List
from io import BytesIO
from PIL import Image


def smiles_to_image(
    smiles: str,
    size: Tuple[int, int] = (300, 300),
    highlight_atoms: Optional[List[int]] = None,
) -> Optional[Image.Image]:
    """
    Convert SMILES to a PIL Image.
    
    Args:
        smiles: SMILES string
        size: Image size (width, height)
        highlight_atoms: Optional list of atom indices to highlight
    
    Returns:
        PIL Image or None if conversion fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Compute 2D coordinates
    AllChem.Compute2DCoords(mol)
    
    # Create drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    
    # Set drawing options
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().addAtomIndices = False
    
    if highlight_atoms:
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
    else:
        drawer.DrawMolecule(mol)
    
    drawer.FinishDrawing()
    
    # Convert to PIL Image
    img_data = drawer.GetDrawingText()
    img = Image.open(BytesIO(img_data))
    
    return img


def plot_molecule(
    smiles: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 6),
) -> plt.Figure:
    """
    Plot a molecule from SMILES.
    
    Args:
        smiles: SMILES string
        title: Optional plot title
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        ax.text(0.5, 0.5, f"Invalid SMILES:\n{smiles}", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    
    # Use RDKit's matplotlib drawing
    img = smiles_to_image(smiles, size=(400, 400))
    
    if img is not None:
        ax.imshow(img)
    
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(smiles, fontsize=10, family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Molecule plot saved to {save_path}")
    
    return fig


def plot_molecular_graph(
    smiles: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    node_size: int = 1500,
    font_size: int = 12,
) -> plt.Figure:
    """
    Plot the molecular graph structure with atom labels.
    
    Shows the graph representation used by the GNN.
    
    Args:
        smiles: SMILES string
        title: Optional plot title
        save_path: Optional path to save
        figsize: Figure size
        node_size: Size of nodes
        font_size: Font size for labels
    
    Returns:
        Matplotlib figure
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Invalid SMILES:\n{smiles}", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes (atoms)
    atom_labels = {}
    atom_colors = []
    
    # Color scheme for common elements
    element_colors = {
        'C': '#7F7F7F',   # Gray
        'N': '#0000FF',   # Blue
        'O': '#FF0000',   # Red
        'S': '#FFFF00',   # Yellow
        'F': '#00FF00',   # Green
        'Cl': '#00FF00',  # Green
        'Br': '#A52A2A',  # Brown
        'P': '#FFA500',   # Orange
        'H': '#FFFFFF',   # White
    }
    
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        G.add_node(idx)
        atom_labels[idx] = symbol
        atom_colors.append(element_colors.get(symbol, '#808080'))
    
    # Add edges (bonds)
    edge_widths = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        
        # Set width based on bond type
        if bond_type == Chem.rdchem.BondType.DOUBLE:
            width = 3
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            width = 4
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            width = 2
        else:
            width = 1.5
        
        G.add_edge(i, j, width=width)
        edge_widths.append(width)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use spring layout
    pos = nx.spring_layout(G, seed=42, k=2)
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color='#404040',
        alpha=0.8,
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=atom_colors,
        node_size=node_size,
        edgecolors='black',
        linewidths=2,
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels=atom_labels,
        font_size=font_size,
        font_weight='bold',
        font_color='white',
    )
    
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"Molecular Graph: {smiles}", fontsize=12)
    
    # Add legend
    legend_elements = []
    for element, color in [('C', '#7F7F7F'), ('N', '#0000FF'), 
                           ('O', '#FF0000'), ('S', '#FFFF00')]:
        from matplotlib.patches import Patch
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=element))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Molecular graph saved to {save_path}")
    
    return fig


def plot_molecule_gallery(
    smiles_list: List[str],
    titles: Optional[List[str]] = None,
    ncols: int = 4,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot a gallery of molecules.
    
    Args:
        smiles_list: List of SMILES strings
        titles: Optional list of titles for each molecule
        ncols: Number of columns in the gallery
        save_path: Optional path to save
        figsize: Figure size (auto-computed if None)
    
    Returns:
        Matplotlib figure
    """
    n_mols = len(smiles_list)
    nrows = (n_mols + ncols - 1) // ncols
    
    if figsize is None:
        figsize = (3 * ncols, 3 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()
    
    for i, smiles in enumerate(smiles_list):
        ax = axes[i]
        
        img = smiles_to_image(smiles, size=(200, 200))
        if img is not None:
            ax.imshow(img)
        
        ax.axis('off')
        
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=10)
        else:
            ax.set_title(smiles[:20] + ('...' if len(smiles) > 20 else ''), 
                        fontsize=8, family='monospace')
    
    # Hide unused axes
    for i in range(n_mols, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Molecule gallery saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Demo
    test_molecules = [
        ('CCO', 'Ethanol'),
        ('c1ccccc1', 'Benzene'),
        ('CC(=O)O', 'Acetic Acid'),
        ('CC(=O)Nc1ccc(O)cc1', 'Paracetamol'),
    ]
    
    # Plot individual molecules
    for smiles, name in test_molecules[:2]:
        fig = plot_molecule(smiles, title=name)
        plt.show()
    
    # Plot molecular graph
    fig = plot_molecular_graph('CC(=O)Nc1ccc(O)cc1', title='Paracetamol Graph')
    plt.show()
    
    # Plot gallery
    smiles_list = [s for s, _ in test_molecules]
    titles = [n for _, n in test_molecules]
    fig = plot_molecule_gallery(smiles_list, titles, ncols=2)
    plt.show()
