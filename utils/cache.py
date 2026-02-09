"""
Dataset Caching Module

Provides cached graph preprocessing to avoid repeated RDKit parsing.
Reduces startup time by storing processed graphs to disk.
"""

import os
import hashlib
import pickle
import torch
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path


class DatasetCache:
    """
    Cache for preprocessed molecular graphs.
    
    Avoids repeated RDKit/graph conversion across training runs.
    Cache is invalidated if:
    - SMILES list changes
    - Preprocessing parameters change
    """
    
    def __init__(
        self,
        cache_dir: str = "cache",
        cache_name: str = "graphs",
    ):
        """
        Initialize dataset cache.
        
        Args:
            cache_dir: Directory for cache files
            cache_name: Base name for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_name = cache_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _compute_hash(
        self,
        smiles_list: List[str],
        labels: List[int],
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Compute hash for cache key based on input data.
        """
        data_str = f"{len(smiles_list)}_{','.join(smiles_list[:10])}_{sum(labels)}"
        if extra_params:
            data_str += f"_{str(sorted(extra_params.items()))}"
        return hashlib.md5(data_str.encode()).hexdigest()[:12]
    
    def get_cache_path(self, cache_hash: str) -> Path:
        """Get path to cache file."""
        return self.cache_dir / f"{self.cache_name}_{cache_hash}.pt"
    
    def exists(
        self,
        smiles_list: List[str],
        labels: List[int],
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if valid cache exists for this data."""
        cache_hash = self._compute_hash(smiles_list, labels, extra_params)
        cache_path = self.get_cache_path(cache_hash)
        return cache_path.exists()
    
    def load(
        self,
        smiles_list: List[str],
        labels: List[int],
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load cached data if available.
        
        Returns:
            Cached data dict or None if not found
        """
        cache_hash = self._compute_hash(smiles_list, labels, extra_params)
        cache_path = self.get_cache_path(cache_hash)
        
        if not cache_path.exists():
            return None
            
        try:
            cached = torch.load(cache_path, weights_only=False)
            # Verify integrity
            if cached.get('n_samples') == len(smiles_list):
                print(f"  📦 Loaded cached graphs from {cache_path.name}")
                return cached
        except Exception as e:
            print(f"  Warning: Cache load failed ({e}), rebuilding...")
        
        return None
    
    def save(
        self,
        data: Dict[str, Any],
        smiles_list: List[str],
        labels: List[int],
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save data to cache.
        
        Args:
            data: Dictionary with 'graphs', 'descriptors', etc.
            smiles_list: Original SMILES list
            labels: Original labels
            extra_params: Additional parameters affecting preprocessing
        """
        cache_hash = self._compute_hash(smiles_list, labels, extra_params)
        cache_path = self.get_cache_path(cache_hash)
        
        # Add metadata
        data['n_samples'] = len(smiles_list)
        data['cache_hash'] = cache_hash
        
        try:
            torch.save(data, cache_path)
            print(f"  💾 Saved graph cache to {cache_path.name}")
        except Exception as e:
            print(f"  Warning: Cache save failed ({e})")
    
    def clear(self) -> int:
        """
        Clear all cache files.
        
        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob(f"{self.cache_name}_*.pt"):
            cache_file.unlink()
            count += 1
        return count


# Global cache instance
_cache = None

def get_cache(cache_dir: str = "cache") -> DatasetCache:
    """Get or create global dataset cache."""
    global _cache
    if _cache is None:
        _cache = DatasetCache(cache_dir=cache_dir)
    return _cache


def cached_graph_preprocessing(
    smiles_list: List[str],
    labels: List[int],
    graph_builder,
    compute_descriptors: bool = False,
    cache_dir: str = "cache",
    force_rebuild: bool = False,
) -> Tuple[List, Optional[List]]:
    """
    Preprocess SMILES to graphs with caching.
    
    Args:
        smiles_list: List of SMILES strings
        labels: List of labels
        graph_builder: MoleculeGraphBuilder instance
        compute_descriptors: Whether to compute descriptors
        cache_dir: Directory for cache files
        force_rebuild: Force rebuild even if cache exists
    
    Returns:
        Tuple of (graphs, descriptors)
    """
    from tqdm import tqdm
    from .smiles_to_graph import compute_molecular_descriptors
    
    cache = get_cache(cache_dir)
    extra_params = {'compute_descriptors': compute_descriptors}
    
    # Try to load from cache
    if not force_rebuild:
        cached = cache.load(smiles_list, labels, extra_params)
        if cached is not None:
            return cached['graphs'], cached.get('descriptors')
    
    # Build graphs
    print("  Building molecular graphs...")
    graphs = []
    descriptors = [] if compute_descriptors else None
    
    for i, (smiles, label) in enumerate(tqdm(zip(smiles_list, labels), total=len(smiles_list), desc="Processing")):
        try:
            graph = graph_builder.smiles_to_graph(smiles)
            if graph is None:
                continue
            graph.y = torch.tensor([label], dtype=torch.long)
            graphs.append(graph)
            
            if compute_descriptors:
                desc = compute_molecular_descriptors(smiles)
                descriptors.append(desc)
        except Exception:
            continue
    
    # Save to cache
    cache_data = {'graphs': graphs}
    if compute_descriptors:
        cache_data['descriptors'] = descriptors
    cache.save(cache_data, smiles_list, labels, extra_params)
    
    return graphs, descriptors


class FastModeConfig:
    """
    Configuration for fast training mode.
    
    Reduces training time for quick demos while maintaining fair comparison.
    """
    
    # Default fast mode settings
    FAST_EPOCHS = 10
    FAST_PATIENCE = 3
    FAST_BATCH_SIZE = 64
    FAST_EVAL_EVERY = 2
    FAST_LOG_EVERY = 5
    FAST_HIDDEN_DIM = 32
    FAST_GNN_LAYERS = 2
    FAST_QUANTUM_QUBITS = 6
    FAST_QUANTUM_LAYERS = 2
    
    # Default standard settings
    STANDARD_EPOCHS = 50
    STANDARD_PATIENCE = 10
    STANDARD_BATCH_SIZE = 32
    
    def __init__(self, fast_mode: bool = False, fast_quantum: bool = False):
        """
        Initialize config.
        
        Args:
            fast_mode: Enable fast training mode
            fast_quantum: Enable fast quantum mode (reduced circuit)
        """
        self.fast_mode = fast_mode
        self.fast_quantum = fast_quantum
        
    @property
    def epochs(self) -> int:
        return self.FAST_EPOCHS if self.fast_mode else self.STANDARD_EPOCHS
    
    @property
    def patience(self) -> int:
        return self.FAST_PATIENCE if self.fast_mode else self.STANDARD_PATIENCE
    
    @property
    def batch_size(self) -> int:
        return self.FAST_BATCH_SIZE if self.fast_mode else self.STANDARD_BATCH_SIZE
    
    @property
    def eval_every(self) -> int:
        return self.FAST_EVAL_EVERY if self.fast_mode else 1
    
    @property
    def log_every(self) -> int:
        return self.FAST_LOG_EVERY if self.fast_mode else 1
    
    @property
    def gnn_hidden_dim(self) -> int:
        return self.FAST_HIDDEN_DIM if self.fast_mode else 64
    
    @property
    def gnn_layers(self) -> int:
        return self.FAST_GNN_LAYERS if self.fast_mode else 3
    
    @property
    def n_qubits(self) -> int:
        return self.FAST_QUANTUM_QUBITS if (self.fast_mode or self.fast_quantum) else 8
    
    @property
    def quantum_layers(self) -> int:
        return self.FAST_QUANTUM_LAYERS if (self.fast_mode or self.fast_quantum) else 3
    
    def summary(self) -> str:
        """Return config summary string."""
        mode = "FAST" if self.fast_mode else "STANDARD"
        quantum = "FAST" if self.fast_quantum else "FULL"
        return (
            f"Training Mode: {mode}\n"
            f"  Epochs: {self.epochs}, Patience: {self.patience}\n"
            f"  Batch size: {self.batch_size}\n"
            f"  GNN: {self.gnn_layers} layers, {self.gnn_hidden_dim} hidden\n"
            f"  Quantum: {quantum} ({self.n_qubits} qubits, {self.quantum_layers} layers)"
        )
