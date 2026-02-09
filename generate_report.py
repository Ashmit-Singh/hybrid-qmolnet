#!/usr/bin/env python
"""
Automatic Evaluation Report Generator

Generates comprehensive evaluation reports for trained models including:
- Markdown summary report
- CSV metrics export
- Comparison tables
- Saved plots

Usage:
    python generate_report.py --output_dir outputs_bbbp

Author: Hybrid QMolNet Team
"""

import os
import sys
import argparse
import json
import glob
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation reports for Hybrid QMolNet"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory containing training outputs (default: outputs)",
    )
    parser.add_argument(
        "--report_name",
        type=str,
        default=None,
        help="Name for the report file (default: auto-generated)",
    )
    parser.add_argument(
        "--include_plots",
        action="store_true",
        default=True,
        help="Include plot generation in report (default: True)",
    )
    return parser.parse_args()


def load_metrics(output_dir: str) -> Dict[str, Any]:
    """
    Load saved metrics from output directory.
    
    Args:
        output_dir: Directory containing training outputs
    
    Returns:
        Dictionary with loaded metrics
    """
    metrics = {}
    
    # Look for metrics JSON files
    metrics_files = glob.glob(os.path.join(output_dir, "**", "*metrics*.json"), recursive=True)
    for f in metrics_files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                name = os.path.basename(f).replace('.json', '').replace('_metrics', '')
                metrics[name] = data
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    # Look for results JSON
    results_files = glob.glob(os.path.join(output_dir, "**", "*results*.json"), recursive=True)
    for f in results_files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                if 'metrics' in data:
                    for model_name, model_metrics in data.get('metrics', {}).items():
                        metrics[model_name] = model_metrics
                elif isinstance(data, dict) and any(k in str(data.keys()).lower() for k in ['accuracy', 'auc', 'f1']):
                    name = os.path.basename(f).replace('.json', '')
                    metrics[name] = data
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    # Look for training history
    history_files = glob.glob(os.path.join(output_dir, "**", "*history*.json"), recursive=True)
    for f in history_files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                name = os.path.basename(f).replace('.json', '').replace('_history', '')
                if 'history' not in metrics:
                    metrics['history'] = {}
                metrics['history'][name] = data
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    return metrics


def load_checkpoint_info(output_dir: str) -> Dict[str, Any]:
    """
    Load information from saved checkpoints.
    
    Args:
        output_dir: Directory containing checkpoints
    
    Returns:
        Dictionary with checkpoint information
    """
    import torch
    
    checkpoint_info = {}
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        return checkpoint_info
    
    # Find best checkpoint
    best_checkpoint = os.path.join(checkpoint_dir, "best.pt")
    if os.path.exists(best_checkpoint):
        try:
            checkpoint = torch.load(best_checkpoint, map_location='cpu')
            checkpoint_info['best'] = {
                'epoch': checkpoint.get('epoch', 'N/A'),
                'val_loss': checkpoint.get('val_loss', 'N/A'),
                'val_accuracy': checkpoint.get('val_accuracy', 'N/A'),
                'model_config': checkpoint.get('model_config', {}),
            }
        except Exception as e:
            print(f"Warning: Could not load best checkpoint: {e}")
    
    # Count checkpoints
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    checkpoint_info['num_checkpoints'] = len(checkpoint_files)
    
    return checkpoint_info


def create_comparison_table(metrics: Dict[str, Any]) -> str:
    """
    Create markdown comparison table from metrics.
    
    Args:
        metrics: Dictionary of model metrics
    
    Returns:
        Markdown table string
    """
    if not metrics:
        return "*No metrics available for comparison.*\n"
    
    # Extract model-level metrics (exclude history)
    model_metrics = {k: v for k, v in metrics.items() 
                     if k != 'history' and isinstance(v, dict)}
    
    if not model_metrics:
        return "*No model metrics available for comparison.*\n"
    
    # Get all unique metric names
    all_metric_names = set()
    for model_data in model_metrics.values():
        if isinstance(model_data, dict):
            all_metric_names.update(model_data.keys())
    
    # Filter to common metrics
    common_metrics = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall', 
                      'test_accuracy', 'test_auc', 'val_accuracy']
    metric_names = [m for m in common_metrics if m in all_metric_names]
    
    if not metric_names:
        metric_names = list(all_metric_names)[:6]  # Take first 6 metrics
    
    # Build table header
    header = "| Model | " + " | ".join(m.replace('_', ' ').title() for m in metric_names) + " |"
    separator = "|" + "---|" * (len(metric_names) + 1)
    
    # Build table rows
    rows = []
    for model_name, model_data in model_metrics.items():
        if not isinstance(model_data, dict):
            continue
        values = []
        for metric in metric_names:
            value = model_data.get(metric, "N/A")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        display_name = model_name.replace('_', ' ').title()
        rows.append(f"| {display_name} | " + " | ".join(values) + " |")
    
    if not rows:
        return "*No model metrics available for comparison.*\n"
    
    return header + "\n" + separator + "\n" + "\n".join(rows) + "\n"


def create_csv_export(metrics: Dict[str, Any], output_path: str):
    """
    Export metrics to CSV file.
    
    Args:
        metrics: Dictionary of model metrics
        output_path: Path to save CSV file
    """
    import csv
    
    model_metrics = {k: v for k, v in metrics.items() 
                     if k != 'history' and isinstance(v, dict)}
    
    if not model_metrics:
        print("No metrics to export to CSV.")
        return
    
    # Get all metric names
    all_metric_names = set()
    for model_data in model_metrics.values():
        if isinstance(model_data, dict):
            all_metric_names.update(model_data.keys())
    
    metric_names = sorted(all_metric_names)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model'] + metric_names)
        
        for model_name, model_data in model_metrics.items():
            if not isinstance(model_data, dict):
                continue
            row = [model_name]
            for metric in metric_names:
                value = model_data.get(metric, '')
                row.append(value)
            writer.writerow(row)
    
    print(f"Exported metrics to: {output_path}")


def generate_plots(output_dir: str, report_dir: str):
    """
    Generate and save evaluation plots.
    
    Args:
        output_dir: Directory containing training outputs
        report_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        from visualization.plots import (
            plot_training_curves,
            plot_confusion_matrix,
        )
    except ImportError as e:
        print(f"Warning: Could not import visualization modules: {e}")
        return
    
    # Check for existing figures
    figures_dir = os.path.join(output_dir, "figures")
    if os.path.exists(figures_dir):
        # Copy existing figures
        import shutil
        for fig_file in glob.glob(os.path.join(figures_dir, "*.png")):
            shutil.copy(fig_file, report_dir)
            print(f"Copied figure: {os.path.basename(fig_file)}")
    
    print(f"Plots saved to: {report_dir}")


def generate_markdown_report(
    metrics: Dict[str, Any],
    checkpoint_info: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Generate comprehensive markdown report.
    
    Args:
        metrics: Loaded metrics
        checkpoint_info: Checkpoint information
        output_dir: Output directory name
    
    Returns:
        Markdown report string
    """
    report = []
    
    # Header
    report.append("# Hybrid QMolNet Evaluation Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Output Directory:** `{output_dir}`")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append("This report presents the evaluation results for the Hybrid QMolNet model, ")
    report.append("which combines Graph Neural Networks (GNN) with Variational Quantum Circuits (VQC) ")
    report.append("for molecular property prediction.")
    report.append("")
    
    # Model Architecture
    report.append("## Model Architecture")
    report.append("")
    report.append("```")
    report.append("SMILES → Graph → GNN Encoder (32-dim) → Compression (8-dim) → VQC → Classifier")
    report.append("```")
    report.append("")
    report.append("- **GNN Encoder:** 3-layer Graph Convolutional Network")
    report.append("- **Quantum Layer:** 8-qubit Variational Quantum Circuit with 3 layers")
    report.append("- **Training:** End-to-end with parameter-shift gradients")
    report.append("")
    
    # Training Summary
    if checkpoint_info.get('best'):
        report.append("## Training Summary")
        report.append("")
        best = checkpoint_info['best']
        report.append(f"- **Best Epoch:** {best.get('epoch', 'N/A')}")
        if isinstance(best.get('val_loss'), float):
            report.append(f"- **Validation Loss:** {best['val_loss']:.4f}")
        if isinstance(best.get('val_accuracy'), float):
            report.append(f"- **Validation Accuracy:** {best['val_accuracy']:.4f}")
        report.append(f"- **Total Checkpoints:** {checkpoint_info.get('num_checkpoints', 'N/A')}")
        report.append("")
    
    # Model Comparison
    report.append("## Model Comparison")
    report.append("")
    report.append(create_comparison_table(metrics))
    report.append("")
    
    # Interpretation Notes
    report.append("## Interpretation Notes")
    report.append("")
    report.append("> **Note:** These results represent model performance on the test set. ")
    report.append("> Performance may vary on different molecular datasets or property prediction tasks.")
    report.append("")
    report.append("**Key Metrics:**")
    report.append("- **Accuracy:** Overall classification correctness")
    report.append("- **ROC-AUC:** Area under the receiver operating characteristic curve")
    report.append("- **F1 Score:** Harmonic mean of precision and recall")
    report.append("")
    
    # Scientific Disclaimer
    report.append("## Scientific Disclaimer")
    report.append("")
    report.append("This model provides computational predictions based on molecular structure analysis. ")
    report.append("Results should be interpreted as estimates and should not replace experimental validation. ")
    report.append("The hybrid quantum-classical approach is a research methodology; no claims of ")
    report.append("quantum advantage are made without rigorous benchmarking.")
    report.append("")
    
    return "\n".join(report)


def main():
    """Main report generation pipeline."""
    args = parse_args()
    
    print("=" * 60)
    print("Hybrid QMolNet Evaluation Report Generator")
    print("=" * 60)
    print()
    
    # Validate output directory
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' does not exist.")
        print("Please specify a valid output directory with --output_dir")
        sys.exit(1)
    
    print(f"Loading results from: {args.output_dir}")
    
    # Load metrics and checkpoint info
    metrics = load_metrics(args.output_dir)
    print(f"Loaded metrics for {len(metrics)} items")
    
    checkpoint_info = load_checkpoint_info(args.output_dir)
    if checkpoint_info.get('best'):
        print(f"Found best checkpoint from epoch {checkpoint_info['best'].get('epoch', 'N/A')}")
    
    # Create report directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_name = args.report_name or f"evaluation_report_{timestamp}"
    report_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate markdown report
    report_content = generate_markdown_report(metrics, checkpoint_info, args.output_dir)
    report_path = os.path.join(report_dir, f"{report_name}.md")
    with open(report_path, 'w') as f:
        f.write(report_content)
    print(f"Generated report: {report_path}")
    
    # Export CSV
    csv_path = os.path.join(report_dir, f"{report_name}.csv")
    create_csv_export(metrics, csv_path)
    
    # Generate plots
    if args.include_plots:
        generate_plots(args.output_dir, report_dir)
    
    print()
    print("=" * 60)
    print("Report generation complete!")
    print(f"Reports saved to: {report_dir}")
    print("=" * 60)
    
    # Print summary to console
    print()
    print("--- SUMMARY ---")
    print(create_comparison_table(metrics))


if __name__ == "__main__":
    main()
