#!/usr/bin/env python3
"""
learning_curves_from_evals.py

Fast learning curve analysis using existing evaluations.npz files.
This script extracts data directly from evaluation checkpoints without re-evaluating.

The evaluations.npz files contain:
- timesteps: array of training steps where checkpoints were evaluated
- results: array of episode returns (episodes × evaluations)
- ep_lengths: array of episode lengths (episodes × evaluations)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path


def load_evaluations(npz_path):
    """Load evaluation data from npz file."""
    data = np.load(npz_path)
    return {
        'timesteps': data['timesteps'],
        'results': data['results'],
        'ep_lengths': data['ep_lengths'],
    }


def analyze_evaluations(eval_data, reward_type):
    """
    Analyze evaluation data and compute statistics.
    
    Returns:
        dict mapping timestep -> stats (mean_return, std_return, success_rate, etc)
    """
    timesteps = eval_data['timesteps']
    results = eval_data['results']
    ep_lengths = eval_data['ep_lengths']
    
    stats_by_timestep = {}
    
    for i, t in enumerate(timesteps):
        episode_returns = results[i]
        episode_lengths = ep_lengths[i]
        
        # Calculate statistics
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        median_return = np.median(episode_returns)
        
        # Success rate estimation (for sparse: close to 100 means success, 
        # for dense: need reward > 200 threshold)
        success_count = np.sum(episode_returns >=100)  # Simple heuristic
        success_rate = success_count / len(episode_returns)
        
        stats_by_timestep[int(t)] = {
            'mean_return': mean_return,
            'std_return': std_return,
            'median_return': median_return,
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
    sparse_stds = [sparse_stats[t]['std_return'] for t in sparse_steps] if sparse_stats else []
    sparse_medians = [sparse_stats[t]['median_return'] for t in sparse_steps] if sparse_stats else []
    sparse_success = [sparse_stats[t]['success_rate'] for t in sparse_steps] if sparse_stats else []
    
    dense_means = [dense_stats[t]['mean_return'] for t in dense_steps] if dense_stats else []
    dense_stds = [dense_stats[t]['std_return'] for t in dense_steps] if dense_stats else []
    dense_medians = [dense_stats[t]['median_return'] for t in dense_steps] if dense_stats else []
    dense_success = [dense_stats[t]['success_rate'] for t in dense_steps] if dense_stats else []
    
    # Convert steps to millions
    sparse_steps_m = [s/1e6 for s in sparse_steps]
    dense_steps_m = [s/1e6 for s in dense_steps]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Learning Curves: Sparse vs Dense Reward Shaping', fontsize=16, fontweight='bold', y=1.00)
    
    # Plot 1: Mean Return with Error Bands
    ax = axes[0, 0]
    if sparse_means:
        ax.plot(sparse_steps_m, sparse_means, 'o-', label='Sparse', linewidth=2.5, markersize=8, color='#1f77b4')

    
    if dense_means:
        ax.plot(dense_steps_m, dense_means, 's-', label='Dense', linewidth=2.5, markersize=8, color='#ff7f0e')

    
    # Add convergence threshold line
    ax.axhline(y=200, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Solved (200)')
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Solved (100)')

    
    ax.set_xlabel('Training Steps (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Episode Return', fontsize=12, fontweight='bold')
    ax.set_title('Mean Episode Return ± Std Dev', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Median Return (more robust)
    ax = axes[0, 1]
    if sparse_medians:
        ax.plot(sparse_steps_m, sparse_medians, 'o-', label='Sparse', linewidth=2.5, markersize=8, color='#1f77b4')
    if dense_medians:
        ax.plot(dense_steps_m, dense_medians, 's-', label='Dense', linewidth=2.5, markersize=8, color='#ff7f0e')
    
    ax.axhline(y=200, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Solved (200)')
    ax.set_xlabel('Training Steps (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Median Episode Return', fontsize=12, fontweight='bold')
    ax.set_title('Median Episode Return', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Success Rate
    ax = axes[1, 0]
    if sparse_success:
        ax.plot(sparse_steps_m, np.array(sparse_success)*100, 'o-', label='Sparse', 
               linewidth=2.5, markersize=8, color='#1f77b4')
    if dense_success:
        ax.plot(dense_steps_m, np.array(dense_success)*100, 's-', label='Dense', 
               linewidth=2.5, markersize=8, color='#ff7f0e')
    
    ax.set_xlabel('Training Steps (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Landing Success Rate (Return >= 100)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Plot 4: Return Distribution at Final Step
    ax = axes[1, 1]
    
    final_returns = []
    final_labels = []
    
    if sparse_stats:
        final_step_sparse = max(sparse_steps)
        final_returns.append(sparse_stats[final_step_sparse]['mean_return'])
        final_labels.append(f"Sparse\n({final_step_sparse/1e6:.1f}M steps)")
    
    if dense_stats:
        final_step_dense = max(dense_steps)
        final_returns.append(dense_stats[final_step_dense]['mean_return'])
        final_labels.append(f"Dense\n({final_step_dense/1e6:.1f}M steps)")
    
    if final_returns:
        colors = ['#1f77b4', '#ff7f0e'][:len(final_returns)]
        bars = ax.bar(final_labels, final_returns, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.axhline(y=200, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Solved')
        ax.set_ylabel('Final Mean Return', fontsize=12, fontweight='bold')
        ax.set_title('Final Performance (Last Evaluation)', fontsize=13, fontweight='bold')
        ax.set_ylim([0, max(final_returns) * 1.15])
    
    plt.tight_layout()
    plot_path = output_dir / 'learning_curves_from_evaluations.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved learning curves: {plot_path}")
    plt.close()


def generate_detailed_report(sparse_stats, dense_stats, output_dir):
    """Generate a detailed text report."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = []
    report.append("=" * 70)
    report.append("LEARNING CURVE ANALYSIS REPORT")
    report.append("Training Progress: Sparse vs Dense Reward Shaping")
    report.append("=" * 70)
    report.append("")
    
    SOLVED_THRESHOLD = 200
    
    # Analyze sparse
    if sparse_stats:
        report.append("=" * 70)
        report.append("SPARSE REWARD TRAINING ANALYSIS")
        report.append("=" * 70)
        report.append("")
        
        steps_list = sorted(sparse_stats.keys())
        
        report.append(f"Number of evaluation checkpoints: {len(steps_list)}")
        report.append(f"Training range: {steps_list[0]:,} to {steps_list[-1]:,} steps")
        report.append("")
        
        first_stats = sparse_stats[steps_list[0]]
        final_stats = sparse_stats[steps_list[-1]]
        
        report.append("Early Training (1st checkpoint):")
        report.append(f"  Mean Return:     {first_stats['mean_return']:7.1f}")
        report.append(f"  Median Return:   {first_stats['median_return']:7.1f}")
        report.append(f"  Std Dev:         {first_stats['std_return']:7.1f}")
        report.append(f"  Success Rate:    {first_stats['success_rate']:7.1%}")
        report.append("")
        
        report.append("Late Training (last checkpoint):")
        report.append(f"  Mean Return:     {final_stats['mean_return']:7.1f}")
        report.append(f"  Median Return:   {final_stats['median_return']:7.1f}")
        report.append(f"  Std Dev:         {final_stats['std_return']:7.1f}")
        report.append(f"  Success Rate:    {final_stats['success_rate']:7.1%}")
        report.append("")
        
        # Find convergence point
        convergence_step = None
        for step in steps_list:
            if sparse_stats[step]['mean_return'] >= SOLVED_THRESHOLD:
                convergence_step = step
                break
        
        if convergence_step:
            report.append(f"✓ Converged to >={SOLVED_THRESHOLD} at: {convergence_step:,} steps ({convergence_step/1e6:.1f}M)")
        else:
            best_return = max(sparse_stats[s]['mean_return'] for s in steps_list)
            best_step = max(steps_list, key=lambda s: sparse_stats[s]['mean_return'])
            report.append(f"✗ Did not converge to {SOLVED_THRESHOLD}")
            report.append(f"  Best performance: {best_return:.1f} at {best_step:,} steps")
        
        report.append("")
        report.append("")
    
    # Analyze dense
    if dense_stats:
        report.append("=" * 70)
        report.append("DENSE REWARD TRAINING ANALYSIS")
        report.append("=" * 70)
        report.append("")
        
        steps_list = sorted(dense_stats.keys())
        
        report.append(f"Number of evaluation checkpoints: {len(steps_list)}")
        report.append(f"Training range: {steps_list[0]:,} to {steps_list[-1]:,} steps")
        report.append("")
        
        first_stats = dense_stats[steps_list[0]]
        final_stats = dense_stats[steps_list[-1]]
        
        report.append("Early Training (1st checkpoint):")
        report.append(f"  Mean Return:     {first_stats['mean_return']:7.1f}")
        report.append(f"  Median Return:   {first_stats['median_return']:7.1f}")
        report.append(f"  Std Dev:         {first_stats['std_return']:7.1f}")
        report.append(f"  Success Rate:    {first_stats['success_rate']:7.1%}")
        report.append("")
        
        report.append("Late Training (last checkpoint):")
        report.append(f"  Mean Return:     {final_stats['mean_return']:7.1f}")
        report.append(f"  Median Return:   {final_stats['median_return']:7.1f}")
        report.append(f"  Std Dev:         {final_stats['std_return']:7.1f}")
        report.append(f"  Success Rate:    {final_stats['success_rate']:7.1%}")
        report.append("")
        
        # Find convergence point
        convergence_step = None
        for step in steps_list:
            if dense_stats[step]['mean_return'] >= SOLVED_THRESHOLD:
                convergence_step = step
                break
        
        if convergence_step:
            report.append(f"✓ Converged to >={SOLVED_THRESHOLD} at: {convergence_step:,} steps ({convergence_step/1e6:.1f}M)")
        else:
            best_return = max(dense_stats[s]['mean_return'] for s in steps_list)
            best_step = max(steps_list, key=lambda s: dense_stats[s]['mean_return'])
            report.append(f"✗ Did not converge to {SOLVED_THRESHOLD}")
            report.append(f"  Best performance: {best_return:.1f} at {best_step:,} steps")
        
        report.append("")
        report.append("")
    
    # Comparison
    if sparse_stats and dense_stats:
        report.append("=" * 70)
        report.append("COMPARISON: SPARSE vs DENSE")
        report.append("=" * 70)
        report.append("")
        
        # Find convergence points
        sparse_steps = sorted(sparse_stats.keys())
        dense_steps = sorted(dense_stats.keys())
        
        sparse_conv = None
        for step in sparse_steps:
            if sparse_stats[step]['mean_return'] >= SOLVED_THRESHOLD:
                sparse_conv = step
                break
        
        dense_conv = None
        for step in dense_steps:
            if dense_stats[step]['mean_return'] >= SOLVED_THRESHOLD:
                dense_conv = step
                break
        
        if sparse_conv and dense_conv:
            if dense_conv < sparse_conv:
                speedup = (sparse_conv - dense_conv) / dense_conv * 100
                report.append(f"FASTER: Dense converges {speedup:.1f}% faster than Sparse")
                report.append(f"   Dense: {dense_conv:,} steps ({dense_conv/1e6:.1f}M)")
                report.append(f"   Sparse: {sparse_conv:,} steps ({sparse_conv/1e6:.1f}M)")
            else:
                speedup = (dense_conv - sparse_conv) / sparse_conv * 100
                report.append(f"FASTER: Sparse converges {speedup:.1f}% faster than Dense")
                report.append(f"   Sparse: {sparse_conv:,} steps ({sparse_conv/1e6:.1f}M)")
                report.append(f"   Dense: {dense_conv:,} steps ({dense_conv/1e6:.1f}M)")
        elif sparse_conv:
            report.append(f"YES: Sparse converged at {sparse_conv:,} steps")
            report.append(f"NO: Dense did not converge to {SOLVED_THRESHOLD}")
        elif dense_conv:
            report.append(f"NO: Sparse did not converge to {SOLVED_THRESHOLD}")
            report.append(f"YES: Dense converged at {dense_conv:,} steps")
        
        report.append("")
        
        # Final performance
        final_sparse_return = sparse_stats[max(sparse_steps)]['mean_return']
        final_dense_return = dense_stats[max(dense_steps)]['mean_return']
        
        report.append("Final Performance (at end of training):")
        report.append(f"  Sparse Return:  {final_sparse_return:7.1f}")
        report.append(f"  Dense Return:   {final_dense_return:7.1f}")
        report.append(f"  Difference:     {final_dense_return - final_sparse_return:+7.1f}")
        report.append("")
    
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    # Save report
    report_path = output_dir / 'detailed_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n✓ Saved detailed report: {report_path}")


def main():
    """Main function."""
    
    experiments_dir = Path(__file__).parent / "experiments"
    output_dir = experiments_dir / "learning_curves_analysis_base"
    
    print("=" * 70)
    print("LEARNING CURVE ANALYSIS FROM EVALUATION CHECKPOINTS")
    print("=" * 70)
    
    # Load sparse evaluations
    sparse_eval_path = experiments_dir / "sparse_base" / "evaluations.npz"
    sparse_stats = None
    
    if sparse_eval_path.exists():
        print(f"\nLoading sparse evaluations: {sparse_eval_path}")
        sparse_eval = load_evaluations(sparse_eval_path)
        sparse_stats = analyze_evaluations(sparse_eval, "sparse")
        print(f"  ✓ Loaded {len(sparse_stats)} evaluation checkpoints")
    else:
        print(f"  ✗ Not found: {sparse_eval_path}")
    
    # Load dense evaluations
    dense_eval_path = experiments_dir / "dense_base" / "evaluations.npz"
    dense_stats = None
    
    if dense_eval_path.exists():
        print(f"\nLoading dense evaluations: {dense_eval_path}")
        dense_eval = load_evaluations(dense_eval_path)
        dense_stats = analyze_evaluations(dense_eval, "dense")
        print(f"  ✓ Loaded {len(dense_stats)} evaluation checkpoints")
    else:
        print(f"  ✗ Not found: {dense_eval_path}")
    
    if not sparse_stats and not dense_stats:
        print("\n✗ ERROR: No evaluation data found!")
        return
    
    # Generate visualizations
    print("\nGenerating learning curve plots...")
    plot_learning_curves(sparse_stats, dense_stats, output_dir)
    
    # Generate report
    print("Generating detailed analysis report...")
    generate_detailed_report(sparse_stats, dense_stats, output_dir)
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete!")
    print(f"  Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
