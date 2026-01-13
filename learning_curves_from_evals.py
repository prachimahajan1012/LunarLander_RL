#!/usr/bin/env python3
"""
learning_curves_from_evals.py

Learning curve analysis from evaluation data using evaluations.npz files.

evaluations.npz files contain:
- timesteps: array of training steps where checkpoints were evaluated
- results: array of episode returns 
- ep_lengths: array of episode lengths 
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_evaluations(npz_path):
    """Load evaluation data from npz file."""
    data = np.load(npz_path)
    return {
        'timesteps': data['timesteps'],
        'results': data['results'],
        'ep_lengths': data['ep_lengths'],
    }


def analyze_evaluations(eval_data):
    """
    Analyze evaluation data and compute statistics.
    
    Returns:
        dict mapping timestep -> stats (mean_return, std_return, success_rate, etc)
    """
    timesteps = eval_data['timesteps']
    results = eval_data['results']
    
    stats_by_timestep = {}
    
    for i, t in enumerate(timesteps):
        episode_returns = results[i]
        
        # Get mean return
        mean_return = np.mean(episode_returns)
        
        # Success rate estimation (for sparse: close to 100 means success, 
        # for dense: need reward > 200 threshold)
        success_count = np.sum(episode_returns >=100)  
        success_rate = success_count / len(episode_returns)
        
        stats_by_timestep[int(t)] = {
            'mean_return': mean_return,
            'min_return': np.min(episode_returns),
            'max_return': np.max(episode_returns),
            'success_rate': success_rate,
            'n_episodes': len(episode_returns),
        }
    
    return stats_by_timestep


def plot_learning_curves(sparse_stats, dense_stats, output_dir):
    """Create comprehensive learning curve plots."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    sns.set_style("whitegrid")
    
    # Extract sorted timesteps
    sparse_steps = sorted(sparse_stats.keys()) if sparse_stats else []
    dense_steps = sorted(dense_stats.keys()) if dense_stats else []
    
    # Extract statistics
    sparse_means = [sparse_stats[t]['mean_return'] for t in sparse_steps] if sparse_stats else []
    sparse_success = [sparse_stats[t]['success_rate'] for t in sparse_steps] if sparse_stats else []
    
    dense_means = [dense_stats[t]['mean_return'] for t in dense_steps] if dense_stats else []
    dense_success = [dense_stats[t]['success_rate'] for t in dense_steps] if dense_stats else []
    
    # Convert steps to millions
    sparse_steps_m = [s/1e6 for s in sparse_steps]
    dense_steps_m = [s/1e6 for s in dense_steps]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1,2, figsize=(15, 5))
    fig.suptitle('Learning Curves: Sparse vs Dense Reward Shaping', fontsize=16, fontweight='bold', y=1.00)
    
    # Plot 1: Mean Return 
    ax = axes[0]
    if sparse_means:
        ax.plot(sparse_steps_m, sparse_means, 'o-', label='Sparse', linewidth=2.5, markersize=8, color='#1f77b4')

    if dense_means:
        ax.plot(dense_steps_m, dense_means, 's-', label='Dense', linewidth=2.5, markersize=8, color='#ff7f0e')

    # Add convergence threshold line
    ax.axhline(y=200, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Solved (200)')
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Solved (100)')
    
    ax.set_xlabel('Training Steps (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Episode Return', fontsize=12, fontweight='bold')
    ax.set_title('Mean Episode Return', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Success Rate
    ax = axes[1]
    if sparse_success:
        ax.plot(sparse_steps_m, np.array(sparse_success)*100, 'o-', label='Sparse', 
               linewidth=2.5, markersize=8, color='#1f77b4')
    if dense_success:
        ax.plot(dense_steps_m, np.array(dense_success)*100, 's-', label='Dense', 
               linewidth=2.5, markersize=8, color='#ff7f0e')
    
    ax.set_xlabel('Training Steps (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Landing Success Rate After Each Checkpoint', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plot_path = output_dir / 'learning_curves_from_evaluations.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function."""
    
    experiments_dir = Path(__file__).parent / "experiments"
    output_dir = experiments_dir / "learning_curves_analysis_base"
    
    # Load sparse evaluations
    sparse_eval_path = experiments_dir / "sparse_base" / "evaluations.npz"
    sparse_stats = None
    
    if sparse_eval_path.exists():
        print(f"\nLoading sparse evaluations: {sparse_eval_path}")
        sparse_eval = load_evaluations(sparse_eval_path)
        sparse_stats = analyze_evaluations(sparse_eval)
    else:
        print(f" Not found: {sparse_eval_path}")
    
    # Load dense evaluations
    dense_eval_path = experiments_dir / "dense_base" / "evaluations.npz"
    dense_stats = None
    
    if dense_eval_path.exists():
        print(f"\nLoading dense evaluations: {dense_eval_path}")
        dense_eval = load_evaluations(dense_eval_path)
        dense_stats = analyze_evaluations(dense_eval)
    else:
        print(f" Not found: {dense_eval_path}")
    
    if not sparse_stats and not dense_stats:
        print("\n ERROR: No evaluation data found!")
        return
    
    # Generate visualizations
    print("\nGenerating learning curve plots...")
    plot_learning_curves(sparse_stats, dense_stats, output_dir)
    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
