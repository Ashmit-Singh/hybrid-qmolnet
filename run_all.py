#!/usr/bin/env python
"""
Hybrid QMolNet - Complete Pipeline Runner

Single-command execution script that:
1. Prepares the dataset (with caching)
2. Trains baseline models (GNN, MLP)
3. Trains hybrid quantum-classical model
4. Evaluates all models
5. Generates comparison plots
6. Saves all outputs

Speed Modes:
  --fast_mode     : Fast training (10 epochs, smaller networks)
  --fast_quantum  : Fast quantum (6 qubits, 2 layers)
  --quick         : Legacy quick mode (100 samples, 10 epochs)
  --max_samples N : Limit dataset size

Author: Hybrid QMolNet Team
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

import torch
import numpy as np

# Ensure reproducibility
SEED = 42


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hybrid QMolNet: Train and evaluate quantum-classical molecular models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Speed Profiles:
  Standard:    python run_all.py --epochs 50
  Fast Mode:   python run_all.py --fast_mode
  Fast+Limit:  python run_all.py --fast_mode --max_samples 500
  Full Fast:   python run_all.py --fast_mode --fast_quantum
  Quick Demo:  python run_all.py --quick
        """
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs (default: 50, or 10 in fast_mode)'
    )
    parser.add_argument(
        '--samples', type=int, default=500,
        help='Number of molecules in synthetic dataset (default: 500)'
    )
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help='Limit dataset to N samples (stratified subsampling)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size for training (default: 32, or 64 in fast_mode)'
    )
    
    # Speed modes
    parser.add_argument(
        '--fast_mode', action='store_true',
        help='Fast training: smaller networks, fewer epochs, larger batches'
    )
    parser.add_argument(
        '--fast_quantum', action='store_true',
        help='Fast quantum: 6 qubits, 2 layers (3-5x faster)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Legacy quick mode: 100 samples, 10 epochs'
    )
    parser.add_argument(
        '--no-quantum', action='store_true',
        help='Skip quantum model training'
    )
    
    # Caching
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Disable dataset caching (rebuild graphs each run)'
    )
    parser.add_argument(
        '--clear-cache', action='store_true',
        help='Clear cache before starting'
    )
    
    # I/O
    parser.add_argument(
        '--output-dir', type=str, default='outputs',
        help='Output directory (default: outputs)'
    )
    parser.add_argument(
        '--data-path', type=str, default=None,
        help='Path to CSV dataset (default: None, uses synthetic)'
    )
    parser.add_argument(
        '--smiles-col', type=str, default='smiles',
        help='Column name for SMILES (default: smiles)'
    )
    parser.add_argument(
        '--label-col', type=str, default='label',
        help='Column name for labels (default: label)'
    )
    
    # Verbosity
    parser.add_argument(
        '--quiet', action='store_true',
        help='Reduce logging output'
    )
    
    return parser.parse_args()


def main():
    """Main execution pipeline."""
    args = parse_args()
    
    # Import config
    from utils.cache import FastModeConfig
    
    # Build configuration
    config = FastModeConfig(
        fast_mode=args.fast_mode,
        fast_quantum=args.fast_quantum
    )
    
    # Quick mode overrides
    if args.quick:
        args.samples = 100
        config.fast_mode = True
    
    # Apply config defaults (only if not explicitly set)
    if args.epochs is None:
        args.epochs = config.epochs
    if args.batch_size is None:
        args.batch_size = config.batch_size
    
    print("\n" + "="*70)
    print("   Hybrid QMolNet: Quantum-Classical Drug Molecule Prediction")
    print("="*70)
    
    # Print mode
    mode_str = []
    if args.fast_mode:
        mode_str.append("FAST")
    if args.fast_quantum:
        mode_str.append("FAST_QUANTUM")
    if args.quick:
        mode_str.append("QUICK")
    mode_display = "+".join(mode_str) if mode_str else "STANDARD"
    
    print(f"\n⚡ Mode: {mode_display}")
    print(f"\nConfiguration:")
    print(f"  Samples:      {args.samples}" + (f" (max: {args.max_samples})" if args.max_samples else ""))
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Train quantum: {not args.no_quantum}")
    print(f"  Output dir:   {args.output_dir}")
    if args.fast_mode:
        print(f"  GNN:          {config.gnn_layers} layers, {config.gnn_hidden_dim} hidden")
        print(f"  Quantum:      {config.n_qubits} qubits, {config.quantum_layers} layers")
    print()
    
    start_time = time.time()
    
    # Import modules
    print("[1/7] Importing modules...")
    from utils.helpers import set_seed, get_device, print_model_summary
    from utils.data_loader import load_dataset, create_data_loaders, MoleculeDataset
    from utils.smiles_to_graph import MoleculeGraphBuilder, compute_molecular_descriptors
    from utils.cache import get_cache
    from models.gnn_encoder import GNNEncoder, print_gnn_summary
    from models.hybrid_model import HybridQMolNet, print_hybrid_model_summary
    from models.baselines import GNNClassifier, DescriptorMLP, print_baseline_summary
    from training.trainer import Trainer, DescriptorTrainer
    from training.callbacks import EarlyStoppingCallback, CheckpointCallback
    from evaluation.metrics import compute_metrics, print_metrics
    from evaluation.evaluator import ModelEvaluator, compare_models
    from visualization.plots import (
        plot_training_curves, plot_roc_curve, plot_confusion_matrix,
        plot_metrics_comparison, plot_multiple_roc_curves
    )
    from visualization.embedding_viz import plot_embedding_comparison
    
    # Set random seed
    set_seed(SEED)
    device = get_device()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)
    
    # Clear cache if requested
    if args.clear_cache:
        cache = get_cache()
        cleared = cache.clear()
        print(f"  Cleared {cleared} cache files")
    
    # =========================================================================
    # Step 2: Prepare Dataset
    # =========================================================================
    print("\n[2/7] Preparing dataset...")
    
    smiles_list, labels = load_dataset(
        data_path=args.data_path,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        n_samples=args.samples, 
        seed=SEED
    )
    
    # Apply max_samples limit with stratified subsampling
    if args.max_samples and args.max_samples < len(smiles_list):
        from sklearn.model_selection import train_test_split
        smiles_list, _, labels, _ = train_test_split(
            smiles_list, labels,
            train_size=args.max_samples,
            random_state=SEED,
            stratify=labels
        )
        print(f"  Subsampled to {len(smiles_list)} molecules (stratified)")
    
    train_loader, val_loader, test_loader, dataset = create_data_loaders(
        smiles_list, labels,
        batch_size=args.batch_size,
        seed=SEED,
        compute_descriptors=True,
        use_cache=not args.no_cache,
    )
    
    node_feature_dim = dataset.node_feature_dim
    print(f"\nNode feature dimension: {node_feature_dim}")
    
    # Store results
    results = {
        'config': {
            **vars(args),
            'fast_mode': args.fast_mode,
            'fast_quantum': args.fast_quantum,
        },
        'dataset': {
            'total_samples': len(smiles_list),
            'train_size': len(train_loader.dataset),
            'val_size': len(val_loader.dataset),
            'test_size': len(test_loader.dataset),
        },
        'models': {}
    }
    
    # Determine hidden dims based on mode
    gnn_hidden = config.gnn_hidden_dim
    gnn_layers = config.gnn_layers
    patience = config.patience
    
    # =========================================================================
    # Step 3: Train GNN Baseline
    # =========================================================================
    print("\n[3/7] Training GNN Baseline...")
    
    gnn_model = GNNClassifier(
        node_feature_dim=node_feature_dim,
        gnn_hidden_dim=gnn_hidden,
        gnn_embedding_dim=32,
        gnn_layers=gnn_layers,
        num_classes=2,
    )
    if not args.quiet:
        print_baseline_summary(gnn_model, "GNN Classifier")
    
    gnn_trainer = Trainer(
        model=gnn_model,
        device=device,
        callbacks=[
            EarlyStoppingCallback(patience=patience, monitor='val_loss'),
            CheckpointCallback(
                save_dir=os.path.join(args.output_dir, 'checkpoints'),
                monitor='val_loss'
            ),
        ],
        model_name="GNN_Baseline",
    )
    
    gnn_history = gnn_trainer.fit(
        train_loader, val_loader,
        num_epochs=args.epochs,
        verbose=not args.quiet
    )
    
    # Plot training curves (skip in fast mode)
    if not args.fast_mode:
        plot_training_curves(
            gnn_history.to_dict(),
            save_path=os.path.join(args.output_dir, 'figures', 'gnn_training.png'),
            title='GNN Baseline Training'
        )
    
    # =========================================================================
    # Step 4: Train Descriptor MLP Baseline
    # =========================================================================
    print("\n[4/7] Training Descriptor MLP Baseline...")
    
    # Prepare descriptor data
    if not args.quiet:
        print("Computing molecular descriptors...")
    train_descriptors = []
    train_labels = []
    for item in dataset:
        if 'descriptors' in item and item['descriptors'] is not None:
            train_descriptors.append(item['descriptors'])
            train_labels.append(item['graph'].y.item())
    
    if len(train_descriptors) > 0:
        # Stack tensors directly
        cleaned_descriptors = []
        for d in train_descriptors:
            if isinstance(d, torch.Tensor):
                cleaned_descriptors.append(d)
            elif isinstance(d, np.ndarray):
                cleaned_descriptors.append(torch.from_numpy(d).float())
            else:
                cleaned_descriptors.append(torch.tensor(d).float())
        
        X_all = torch.stack(cleaned_descriptors)
        y_all = torch.tensor(train_labels, dtype=torch.long)
        
        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_all, y_all, test_size=0.3, random_state=SEED, stratify=y_all
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
        )
        
        mlp_hidden = (64, 32, 16) if not args.fast_mode else (32, 16)
        mlp_model = DescriptorMLP(
            input_dim=X_train.shape[1],
            hidden_dims=mlp_hidden,
            num_classes=2,
        )
        mlp_model.fit_normalization(X_train)
        if not args.quiet:
            print_baseline_summary(mlp_model, "Descriptor MLP")
        
        mlp_trainer = DescriptorTrainer(
            model=mlp_model,
            device=device,
            model_name="Descriptor_MLP",
        )
        
        mlp_history = mlp_trainer.fit(
            X_train, y_train, X_val, y_val,
            num_epochs=args.epochs,
            verbose=not args.quiet
        )
        
        if not args.fast_mode:
            plot_training_curves(
                mlp_history.to_dict(),
                save_path=os.path.join(args.output_dir, 'figures', 'mlp_training.png'),
                title='MLP Baseline Training'
            )
    else:
        print("Warning: No descriptors computed, skipping MLP baseline")
        mlp_model = None
    
    # =========================================================================
    # Step 5: Train Hybrid Quantum Model
    # =========================================================================
    if not args.no_quantum:
        print("\n[5/7] Training Hybrid Quantum-Classical Model...")
        
        hybrid_model = HybridQMolNet(
            node_feature_dim=node_feature_dim,
            gnn_hidden_dim=gnn_hidden,
            gnn_embedding_dim=32,
            gnn_layers=gnn_layers,
            n_qubits=config.n_qubits,
            quantum_layers=config.quantum_layers,
            num_classes=2,
            fast_quantum=args.fast_quantum or args.fast_mode,
        )
        if not args.quiet:
            print_hybrid_model_summary(hybrid_model)
        
        hybrid_trainer = Trainer(
            model=hybrid_model,
            device=device,
            callbacks=[
                EarlyStoppingCallback(patience=patience + 5, monitor='val_loss'),
                CheckpointCallback(
                    save_dir=os.path.join(args.output_dir, 'checkpoints'),
                    monitor='val_loss'
                ),
            ],
            model_name="Hybrid_QMolNet",
        )
        
        hybrid_history = hybrid_trainer.fit(
            train_loader, val_loader,
            num_epochs=args.epochs,
            verbose=not args.quiet
        )
        
        if not args.fast_mode:
            plot_training_curves(
                hybrid_history.to_dict(),
                save_path=os.path.join(args.output_dir, 'figures', 'hybrid_training.png'),
                title='Hybrid QMolNet Training'
            )
    else:
        print("\n[5/7] Skipping Hybrid Model (--no-quantum flag)")
        hybrid_model = None
    
    # =========================================================================
    # Step 6: Evaluate All Models
    # =========================================================================
    print("\n[6/7] Evaluating Models on Test Set...")
    
    all_metrics = {}
    roc_data = {}
    
    # Evaluate GNN
    if not args.quiet:
        print("\nEvaluating GNN Baseline...")
    gnn_evaluator = ModelEvaluator(gnn_model, device=device, model_name="GNN Baseline")
    gnn_metrics = gnn_evaluator.evaluate(test_loader)
    if not args.quiet:
        gnn_evaluator.print_results()
    all_metrics['GNN Baseline'] = gnn_metrics
    
    fpr, tpr, _ = gnn_evaluator.get_roc_curve()
    roc_data['GNN Baseline'] = (fpr, tpr, gnn_metrics.get('roc_auc', 0.5))
    
    # Save confusion matrix (skip in fast mode)
    if not args.fast_mode:
        cm = gnn_evaluator.get_confusion_matrix()
        plot_confusion_matrix(
            cm,
            save_path=os.path.join(args.output_dir, 'figures', 'gnn_confusion.png'),
            title='GNN Baseline Confusion Matrix'
        )
    
    results['models']['gnn'] = gnn_metrics
    
    # Evaluate MLP
    if mlp_model is not None:
        if not args.quiet:
            print("\nEvaluating MLP Baseline...")
        mlp_model.eval()
        with torch.no_grad():
            X_test_dev = X_test.to(device)
            mlp_logits = mlp_model(X_test_dev)
            mlp_probs = torch.softmax(mlp_logits, dim=1)[:, 1].cpu().numpy()
            mlp_preds = mlp_logits.argmax(dim=1).cpu().numpy()
        
        mlp_metrics = compute_metrics(y_test.numpy(), mlp_preds, mlp_probs)
        if not args.quiet:
            print_metrics(mlp_metrics, "Descriptor MLP")
        all_metrics['MLP Baseline'] = mlp_metrics
        
        from evaluation.metrics import compute_roc_curve
        fpr_mlp, tpr_mlp, _ = compute_roc_curve(y_test.numpy(), mlp_probs)
        roc_data['MLP Baseline'] = (fpr_mlp, tpr_mlp, mlp_metrics.get('roc_auc', 0.5))
        
        results['models']['mlp'] = mlp_metrics
    
    # Evaluate Hybrid
    if hybrid_model is not None:
        if not args.quiet:
            print("\nEvaluating Hybrid QMolNet...")
        hybrid_evaluator = ModelEvaluator(
            hybrid_model, device=device, model_name="Hybrid QMolNet"
        )
        hybrid_metrics = hybrid_evaluator.evaluate(test_loader)
        if not args.quiet:
            hybrid_evaluator.print_results()
        all_metrics['Hybrid QMolNet'] = hybrid_metrics
        
        fpr_h, tpr_h, _ = hybrid_evaluator.get_roc_curve()
        roc_data['Hybrid QMolNet'] = (fpr_h, tpr_h, hybrid_metrics.get('roc_auc', 0.5))
        
        if not args.fast_mode:
            cm_hybrid = hybrid_evaluator.get_confusion_matrix()
            plot_confusion_matrix(
                cm_hybrid,
                save_path=os.path.join(args.output_dir, 'figures', 'hybrid_confusion.png'),
                title='Hybrid QMolNet Confusion Matrix'
            )
            
            # Extract and visualize embeddings (skip in fast mode)
            embeddings, emb_labels = hybrid_evaluator.get_embeddings(test_loader, layer='gnn')
            plot_embedding_comparison(
                embeddings, emb_labels,
                title='Hybrid QMolNet Embeddings',
                save_path=os.path.join(args.output_dir, 'figures', 'embeddings.png')
            )
        
        results['models']['hybrid'] = hybrid_metrics
    
    # =========================================================================
    # Step 7: Generate Comparison Plots
    # =========================================================================
    print("\n[7/7] Generating Comparison Plots...")
    
    # ROC comparison
    if len(roc_data) > 1 and not args.fast_mode:
        plot_multiple_roc_curves(
            roc_data,
            save_path=os.path.join(args.output_dir, 'figures', 'roc_comparison.png')
        )
    
    # Metrics bar chart
    plot_metrics_comparison(
        all_metrics,
        save_path=os.path.join(args.output_dir, 'figures', 'metrics_comparison.png'),
        title='Model Performance Comparison'
    )
    
    # =========================================================================
    # Save Results
    # =========================================================================
    end_time = time.time()
    duration = end_time - start_time
    
    results['duration_seconds'] = duration
    results['timestamp'] = datetime.now().isoformat()
    
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=convert)
    
    # =========================================================================
    # Print Summary
    # =========================================================================
    print("\n" + "="*70)
    print("   EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\n⏱ Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - results.json")
    print(f"  - figures/")
    print(f"  - checkpoints/")
    
    print("\n" + "-"*70)
    print("Model Performance Summary:")
    print("-"*70)
    
    for model_name, metrics in all_metrics.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"  ROC-AUC:  {metrics.get('roc_auc', 0):.4f}")
        print(f"  F1 Score: {metrics.get('f1', 0):.4f}")
    
    # Determine winner
    if all_metrics:
        best_model = max(all_metrics.items(), key=lambda x: x[1].get('accuracy', 0))
        print(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
    
    print("\n" + "="*70)
    print("   Thank you for using Hybrid QMolNet!")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    main()
