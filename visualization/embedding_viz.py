"""
Embedding Visualization Module

Visualize high-dimensional embeddings using dimensionality reduction:
- PCA (Principal Component Analysis)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional, Tuple, List


def plot_embeddings_pca(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Embedding Visualization (PCA)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.7,
    marker_size: int = 100,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Visualize embeddings using PCA.
    
    Args:
        embeddings: Embedding matrix [N, D]
        labels: Class labels [N]
        class_names: Optional list of class names
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        alpha: Point transparency
        marker_size: Size of scatter points
    
    Returns:
        Tuple of (figure, pca_result)
    """
    if class_names is None:
        class_names = ['Inactive', 'Active']
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each class
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=class_names[int(label)] if int(label) < len(class_names) else f'Class {label}',
            alpha=alpha,
            s=marker_size,
            edgecolors='white',
            linewidths=0.5,
        )
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"PCA embedding plot saved to {save_path}")
    
    return fig, embeddings_2d


def plot_embeddings_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Embedding Visualization (t-SNE)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    perplexity: float = 30.0,
    n_iter: int = 1000,
    alpha: float = 0.7,
    marker_size: int = 100,
    random_state: int = 42,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Visualize embeddings using t-SNE.
    
    Args:
        embeddings: Embedding matrix [N, D]
        labels: Class labels [N]
        class_names: Optional list of class names
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        alpha: Point transparency
        marker_size: Size of scatter points
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (figure, tsne_result)
    """
    if class_names is None:
        class_names = ['Inactive', 'Active']
    
    # Apply t-SNE
    # Adjust perplexity if needed
    n_samples = embeddings.shape[0]
    adjusted_perplexity = min(perplexity, n_samples - 1)
    
    tsne = TSNE(
        n_components=2,
        perplexity=adjusted_perplexity,
        n_iter=n_iter,
        random_state=random_state,
    )
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each class
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=class_names[int(label)] if int(label) < len(class_names) else f'Class {label}',
            alpha=alpha,
            s=marker_size,
            edgecolors='white',
            linewidths=0.5,
        )
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"t-SNE embedding plot saved to {save_path}")
    
    return fig, embeddings_2d


def plot_embedding_comparison(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Embedding Visualization",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 6),
) -> plt.Figure:
    """
    Plot both PCA and t-SNE side by side.
    
    Args:
        embeddings: Embedding matrix [N, D]
        labels: Class labels [N]
        class_names: Optional list of class names
        title: Overall plot title
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = ['Inactive', 'Active']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        axes[0].scatter(
            pca_result[mask, 0],
            pca_result[mask, 1],
            c=[color],
            label=class_names[int(label)] if int(label) < len(class_names) else f'Class {label}',
            alpha=0.7,
            s=80,
            edgecolors='white',
            linewidths=0.5,
        )
    
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    axes[0].set_title('PCA Projection', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE
    n_samples = embeddings.shape[0]
    adjusted_perplexity = min(30, n_samples - 1)
    
    tsne = TSNE(n_components=2, perplexity=adjusted_perplexity, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        axes[1].scatter(
            tsne_result[mask, 0],
            tsne_result[mask, 1],
            c=[color],
            label=class_names[int(label)] if int(label) < len(class_names) else f'Class {label}',
            alpha=0.7,
            s=80,
            edgecolors='white',
            linewidths=0.5,
        )
    
    axes[1].set_xlabel('t-SNE 1', fontsize=11)
    axes[1].set_ylabel('t-SNE 2', fontsize=11)
    axes[1].set_title('t-SNE Projection', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Embedding comparison saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Demo
    np.random.seed(42)
    
    # Generate synthetic embeddings
    n_samples = 200
    n_features = 32
    
    # Class 0: centered at one location
    emb_0 = np.random.randn(n_samples // 2, n_features) + np.array([2] * n_features)
    # Class 1: centered at another location
    emb_1 = np.random.randn(n_samples // 2, n_features) + np.array([-2] * n_features)
    
    embeddings = np.vstack([emb_0, emb_1])
    labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    embeddings = embeddings[idx]
    labels = labels[idx]
    
    # Plot PCA
    fig, pca_result = plot_embeddings_pca(embeddings, labels)
    plt.show()
    
    # Plot t-SNE
    fig, tsne_result = plot_embeddings_tsne(embeddings, labels)
    plt.show()
    
    # Plot comparison
    fig = plot_embedding_comparison(embeddings, labels, title="Demo Embeddings")
    plt.show()
