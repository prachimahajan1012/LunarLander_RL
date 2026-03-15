#!/usr/bin/env python3
"""
extract_learning_metrics.py

Extract detailed metrics from evaluation data at specific training steps.
Useful for creating comparison tables and detailed statistics.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime


def load_evaluations(npz_path):
    """Load evaluation data from npz file."""
    data = np.load(npz_path)
    return {
        'timesteps': data['timesteps'],
        'results': data['results'],
        'ep_lengths': data['ep_lengths'],
    }


def get_stats_at_timestep(eval_data, timestep_idx):
    """Get all statistics for a specific checkpoint."""
    results = eval_data['results'][timestep_idx]
    ep_lengths = eval_data['ep_lengths'][timestep_idx]
    timestep = int(eval_data['timesteps'][timestep_idx])
    
    return {
        'timestep': timestep,
        'n_episodes': len(results),
        'mean_return': float(np.mean(results)),
        'median_return': float(np.median(results)),
        'std_return': float(np.std(results)),
        'min_return': float(np.min(results)),
        'max_return': float(np.max(results)),
        'q25_return': float(np.percentile(results, 25)),
        'q75_return': float(np.percentile(results, 75)),
        'mean_length': float(np.mean(ep_lengths)),
        'median_length': float(np.median(ep_lengths)),
        'std_length': float(np.std(ep_lengths)),
        'min_length': int(np.min(ep_lengths)),
        'max_length': int(np.max(ep_lengths)),
        'success_rate': float(np.sum(results > 100) / len(results)),
    }


def create_comparison_table(sparse_stats, dense_stats):
    """Create a formatted comparison table."""
    
    # Find key timesteps for comparison
    key_steps = [50000, 100000, 250000, 500000, 750000, 1000000, 1500000, 2000000]
    
    lines = []
    lines.append("\n" + "=" * 110)
    lines.append("DETAILED METRICS COMPARISON")
    lines.append("=" * 110)
    lines.append("")
    
    for step in key_steps:
        lines.append(f"\n--- Training Step: {step:,} ({step/1e6:.1f}M) ---\n")
        
        # Find nearest timestep for sparse
        sparse_idx = None
        if sparse_stats:
            sparse_ts = sparse_stats['timesteps']
            idx = np.argmin(np.abs(sparse_ts - step))
            if np.abs(sparse_ts[idx] - step) < 50000:  # Within 50k tolerance
                sparse_idx = idx
        
        # Find nearest timestep for dense
        dense_idx = None
        if dense_stats:
            dense_ts = dense_stats['timesteps']
            idx = np.argmin(np.abs(dense_ts - step))
            if np.abs(dense_ts[idx] - step) < 50000:
                dense_idx = idx
        
        if sparse_idx is not None and dense_idx is not None:
            sparse_data = get_stats_at_timestep(sparse_stats, sparse_idx)
            dense_data = get_stats_at_timestep(dense_stats, dense_idx)
            
            lines.append(f"{'Metric':<25} {'Sparse':<20} {'Dense':<20} {'Difference':<20}")
            lines.append("-" * 110)
            
            metrics = [
                ('Mean Return', 'mean_return'),
                ('Median Return', 'median_return'),
                ('Std Dev Return', 'std_return'),
                ('Min Return', 'min_return'),
                ('Max Return', 'max_return'),
                ('Q25-Q75 Range', 'q25_return', 'q75_return'),
                ('Success Rate', 'success_rate'),
                ('Mean Episode Length', 'mean_length'),
                ('Median Episode Length', 'median_length'),
            ]
            
            for metric in metrics:
                if len(metric) == 2:
                    label, key = metric
                    sparse_val = sparse_data[key]
                    dense_val = dense_data[key]
                    
                    if 'success' in key or 'rate' in key:
                        fmt_s = f"{sparse_val*100:>6.1f}%"
                        fmt_d = f"{dense_val*100:>6.1f}%"
                        fmt_diff = f"{(dense_val-sparse_val)*100:+6.1f}%"
                    else:
                        fmt_s = f"{sparse_val:>6.1f}"
                        fmt_d = f"{dense_val:>6.1f}"
                        fmt_diff = f"{(dense_val-sparse_val):+6.1f}"
                    
                    lines.append(f"{label:<25} {fmt_s:<20} {fmt_d:<20} {fmt_diff:<20}")
                else:
                    # Range metric
                    label = metric[0]
                    q25_key, q75_key = metric[1], metric[2]
                    sparse_range = f"{sparse_data[q25_key]:.1f}-{sparse_data[q75_key]:.1f}"
                    dense_range = f"{dense_data[q25_key]:.1f}-{dense_data[q75_key]:.1f}"
                    lines.append(f"{label:<25} {sparse_range:<20} {dense_range:<20} {'':20}")
    
    lines.append("\n" + "=" * 110)
    
    return "\n".join(lines)


def extract_convergence_analysis(sparse_stats, dense_stats):
    """Analyze convergence behavior."""
    
    lines = []
    lines.append("\n" + "=" * 110)
    lines.append("CONVERGENCE ANALYSIS")
    lines.append("=" * 110)
    lines.append("")
    
    thresholds = [100, 150, 200, 250]
    
    for threshold in thresholds:
        lines.append(f"\nConvergence to Return >= {threshold}:")
        lines.append("-" * 50)
        
        # Find convergence for sparse
        if sparse_stats:
            sparse_returns = sparse_stats['results'].mean(axis=1)
            sparse_steps = sparse_stats['timesteps']
            sparse_conv_idx = np.where(sparse_returns >= threshold)[0]
            if len(sparse_conv_idx) > 0:
                sparse_conv_step = sparse_steps[sparse_conv_idx[0]]
                lines.append(f"  Sparse:  {sparse_conv_step:,} steps ({sparse_conv_step/1e6:.1f}M)")
            else:
                best_idx = np.argmax(sparse_returns)
                lines.append(f"  Sparse:  Did not converge (best: {sparse_returns[best_idx]:.1f} at {sparse_steps[best_idx]:,} steps)")
        
        # Find convergence for dense
        if dense_stats:
            dense_returns = dense_stats['results'].mean(axis=1)
            dense_steps = dense_stats['timesteps']
            dense_conv_idx = np.where(dense_returns >= threshold)[0]
            if len(dense_conv_idx) > 0:
                dense_conv_step = dense_steps[dense_conv_idx[0]]
                lines.append(f"  Dense:   {dense_conv_step:,} steps ({dense_conv_step/1e6:.1f}M)")
            else:
                best_idx = np.argmax(dense_returns)
                lines.append(f"  Dense:   Did not converge (best: {dense_returns[best_idx]:.1f} at {dense_steps[best_idx]:,} steps)")
    
    lines.append("\n" + "=" * 110)
    
    return "\n".join(lines)


def main():
    """Main function."""
    
    experiments_dir = Path(__file__).parent / "experiments"
    output_dir = experiments_dir / "learning_curves_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 110)
    print("EXTRACTING DETAILED LEARNING METRICS")
    print("=" * 110)
    
    # Load data
    sparse_stats = None
    dense_stats = None
    
    sparse_path = experiments_dir / "sparse" / "evaluations.npz"
    if sparse_path.exists():
        print(f"\nLoading sparse evaluations: {sparse_path}")
        sparse_stats = load_evaluations(sparse_path)
        print(f"  Loaded {len(sparse_stats['timesteps'])} checkpoints")
    
    dense_path = experiments_dir / "dense" / "evaluations.npz"
    if dense_path.exists():
        print(f"Loading dense evaluations: {dense_path}")
        dense_stats = load_evaluations(dense_path)
        print(f"  Loaded {len(dense_stats['timesteps'])} checkpoints")
    
    if not sparse_stats and not dense_stats:
        print("\nERROR: No evaluation data found!")
        return
    
    # Generate comparison table
    print("\nGenerating detailed metrics comparison...")
    comparison = create_comparison_table(sparse_stats, dense_stats)
    print(comparison)
    
    # Generate convergence analysis
    print("\nGenerating convergence analysis...")
    convergence = extract_convergence_analysis(sparse_stats, dense_stats)
    print(convergence)
    
    # Save to file
    report_path = output_dir / 'detailed_metrics_extraction.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(comparison)
        f.write(convergence)
    
    print(f"\nReport saved to: {report_path}")
    print("=" * 110)


if __name__ == "__main__":
    main()
