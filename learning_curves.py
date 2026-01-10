#!/usr/bin/env python3
"""
learning_curves.py

Generate learning curves comparing sparse vs dense reward training.
Analyzes checkpoints to show:
- Episode return progression during training
- Success rate during training
- Time to convergence (episodes to solve)
- Return at different training timesteps
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import gymnasium as gym
from stable_baselines3 import PPO

# ---- Reward Wrappers (matching experiment setup) ----

class SparseTerminalRewardEnv(gym.Wrapper):
    """Sparse reward: only terminal + step penalty."""
    def __init__(self, env, success_reward=100.0, fail_reward=-100.0):
        super().__init__(env)
        self.success_reward = success_reward
        self.fail_reward = fail_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = 0.0
        done = terminated or truncated
        if done:
            try:
                leg1 = bool(obs[6])
                leg2 = bool(obs[7])
            except:
                leg1 = leg2 = False
            
            landed = leg1 and leg2
            reward = self.success_reward if landed else self.fail_reward
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class DenseRewardEnv(gym.Wrapper):
    """Dense reward with fuel penalty shaped reward."""
    def __init__(self, env, success_reward=100.0, fail_reward=-100.0, fuel_penalty=-0.05):
        super().__init__(env)
        self.success_reward = success_reward
        self.fail_reward = fail_reward
        self.fuel_penalty = fuel_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Add fuel penalty
        reward += self.fuel_penalty
        
        if done:
            try:
                leg1 = bool(obs[6])
                leg2 = bool(obs[7])
            except:
                leg1 = leg2 = False
            
            landed = leg1 and leg2
            terminal_reward = self.success_reward if landed else self.fail_reward
            reward += terminal_reward
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def get_wrapper(reward_type):
    """Return the appropriate wrapper class."""
    if reward_type == "sparse":
        return SparseTerminalRewardEnv
    elif reward_type == "dense":
        return DenseRewardEnv
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def make_env(reward_type='dense', seed=0):
    """Create environment with reward wrapper."""
    env = gym.make("LunarLander-v3")
    wrapper = get_wrapper(reward_type)
    env = wrapper(env)
    env.reset(seed=seed)
    return env


def evaluate_checkpoint(model_path, reward_type, n_episodes=50, seed=0):
    """
    Evaluate a checkpoint and return episode statistics.
    
    Returns:
        dict with keys: 'returns', 'lengths', 'success_rate', 'mean_return', 'std_return'
    """
    try:
        # Load model
        model = PPO.load(model_path)
        
        # Create environment
        env = make_env(reward_type, seed=seed)
        
        returns = []
        lengths = []
        successes = 0
        
        for _ in range(n_episodes):
            obs, info = env.reset(seed=seed + _)
            done = False
            episode_return = 0.0
            episode_length = 0
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                episode_length += 1
                done = terminated or truncated
            
            returns.append(episode_return)
            lengths.append(episode_length)
            
            # Check if successful (landed on both legs)
            try:
                leg1 = bool(obs[6])
                leg2 = bool(obs[7])
                if leg1 and leg2:
                    successes += 1
            except:
                pass
        
        env.close()
        
        return {
            'returns': np.array(returns),
            'lengths': np.array(lengths),
            'success_rate': successes / n_episodes,
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
        }
    except Exception as e:
        print(f"Error evaluating {model_path}: {e}")
        return None


def extract_checkpoint_step(filename):
    """Extract the step number from checkpoint filename."""
    # Format: ckpt_sparse_run0_1000000_steps.zip
    parts = filename.split('_')
    if 'steps' in parts[-1]:
        try:
            return int(parts[-2])
        except:
            return None
    return None


def collect_learning_data(experiments_dir, reward_type):
    """
    Collect learning data from checkpoints.
    
    Returns:
        dict mapping timestep -> evaluation stats
    """
    if reward_type == "sparse":
        reward_dir = experiments_dir / "sparse_base"
    elif reward_type == "dense":
        reward_dir = experiments_dir / "dense_base"
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
    
    if not reward_dir.exists():
        print(f"Directory not found: {reward_dir}")
        return None
    
    learning_data = {}
    
    # Find all checkpoint files
    checkpoint_files = sorted(reward_dir.glob("ckpt_*_steps.zip"))
    
    print(f"\nProcessing {reward_type} checkpoints...")
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    for i, checkpoint_file in enumerate(checkpoint_files):
        step = extract_checkpoint_step(checkpoint_file.name)
        if step is None:
            continue
        
        print(f"  [{i+1}/{len(checkpoint_files)}] Evaluating {checkpoint_file.name}...", end=" ")
        
        result = evaluate_checkpoint(str(checkpoint_file), reward_type, n_episodes=10)
        
        if result is not None:
            learning_data[step] = result
            print(f"Success rate: {result['success_rate']:.1%}, Mean return: {result['mean_return']:.1f}")
        else:
            print("FAILED")
    
    return learning_data


def load_training_stats(experiments_dir, reward_type):
    """Load pre-computed training stats from JSON files."""
    if reward_type == "sparse":
        reward_dir = experiments_dir / "sparse_base"
    elif reward_type == "dense":
        reward_dir = experiments_dir / "dense_base"
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
    
    stats_by_run = {}
    stats_files = sorted(reward_dir.glob("stats_*_seed*.json"))
    
    for stats_file in stats_files:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            run_name = stats_file.stem
            stats_by_run[run_name] = stats
    
    return stats_by_run


def plot_learning_curves(learning_data_sparse, learning_data_dense, output_dir):
    """Create comparison plots between sparse and dense rewards."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Extract data for plotting
    steps_sparse = sorted(learning_data_sparse.keys()) if learning_data_sparse else []
    steps_dense = sorted(learning_data_dense.keys()) if learning_data_dense else []
    
    returns_sparse = [learning_data_sparse[s]['mean_return'] for s in steps_sparse] if learning_data_sparse else []
    returns_dense = [learning_data_dense[s]['mean_return'] for s in steps_dense] if learning_data_dense else []
    
    success_sparse = [learning_data_sparse[s]['success_rate'] for s in steps_sparse] if learning_data_sparse else []
    success_dense = [learning_data_dense[s]['success_rate'] for s in steps_dense] if learning_data_dense else []
    
    std_sparse = [learning_data_sparse[s]['std_return'] for s in steps_sparse] if learning_data_sparse else []
    std_dense = [learning_data_dense[s]['std_return'] for s in steps_dense] if learning_data_dense else []
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean Episode Return
    ax = axes[0, 0]
    if steps_sparse:
        ax.plot([s/1e6 for s in steps_sparse], returns_sparse, 'o-', label='Sparse', linewidth=2, markersize=6)
    if steps_dense:
        ax.plot([s/1e6 for s in steps_dense], returns_dense, 's-', label='Dense', linewidth=2, markersize=6)
    ax.set_xlabel('Training Steps (Millions)', fontsize=11)
    ax.set_ylabel('Mean Episode Return', fontsize=11)
    ax.set_title('Episode Return during Training', fontsize=12, fontweight='bold')
    # Add convergence threshold line
    ax.axhline(y=200, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Solved (200)')
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Solved (100)')

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Success Rate
    ax = axes[0, 1]
    if steps_sparse:
        ax.plot([s/1e6 for s in steps_sparse], success_sparse, 'o-', label='Sparse', linewidth=2, markersize=6)
    if steps_dense:
        ax.plot([s/1e6 for s in steps_dense], success_dense, 's-', label='Dense', linewidth=2, markersize=6)
    ax.set_xlabel('Training Steps (Millions)', fontsize=11)
    ax.set_ylabel('Success Rate', fontsize=11)
    ax.set_title('Landing Success Rate during Training', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Plot 3: Return with Error Bars
    ax = axes[1, 0]
    if steps_sparse:
        ax.errorbar([s/1e6 for s in steps_sparse], returns_sparse, yerr=std_sparse, 
                   fmt='o-', label='Sparse', linewidth=2, markersize=6, capsize=5, capthick=2)
    if steps_dense:
        ax.errorbar([s/1e6 for s in steps_dense], returns_dense, yerr=std_dense, 
                   fmt='s-', label='Dense', linewidth=2, markersize=6, capsize=5, capthick=2)
    ax.set_xlabel('Training Steps (Millions)', fontsize=11)
    ax.set_ylabel('Episode Return ± Std Dev', fontsize=11)
    ax.set_title('Episode Return (with Variability)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Convergence Speed (episodes to solve)
    ax = axes[1, 1]
    solve_threshold = 200  # LunarLander is "solved" at 200 avg return
    
    sparse_convergence = None
    for step, ret in zip(steps_sparse, returns_sparse):
        if ret >= solve_threshold:
            sparse_convergence = step
            break
    
    dense_convergence = None
    for step, ret in zip(steps_dense, returns_dense):
        if ret >= solve_threshold:
            dense_convergence = step
            break
    
    if sparse_convergence or dense_convergence:
        labels = []
        values = []
        colors = []
        if sparse_convergence:
            labels.append("Sparse")
            values.append(sparse_convergence / 1e6)
            colors.append('C0')
        else:
            labels.append("Sparse (Not Converged)")
            values.append(2.0)
            colors.append('C0')
        
        if dense_convergence:
            labels.append("Dense")
            values.append(dense_convergence / 1e6)
            colors.append('C1')
        else:
            labels.append("Dense (Not Converged)")
            values.append(2.0)
            colors.append('C1')
        
        bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Steps to Convergence (Millions)', fontsize=11)
        ax.set_title(f'Convergence Speed (threshold={solve_threshold})', fontsize=12, fontweight='bold')
        ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Max evaluated')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}M' if val < 2.0 else 'N/A',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved learning curves to: {output_dir / 'learning_curves_comparison.png'}")
    plt.close()


def generate_convergence_report(learning_data_sparse, learning_data_dense, output_dir):
    """Generate a text report with key metrics."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    report = []
    report.append("=" * 60)
    report.append("LEARNING CURVE ANALYSIS REPORT")
    report.append("Sparse vs Dense Reward Training")
    report.append("=" * 60)
    report.append("")
    
    solve_threshold = 200
    
    # Analyze sparse
    if learning_data_sparse:
        report.append("SPARSE REWARD TRAINING")
        report.append("-" * 60)
        steps_sparse = sorted(learning_data_sparse.keys())
        returns_sparse = [learning_data_sparse[s]['mean_return'] for s in steps_sparse]
        success_sparse = [learning_data_sparse[s]['success_rate'] for s in steps_sparse]
        
        report.append(f"Training points evaluated: {len(steps_sparse)}")
        report.append(f"Final mean return: {returns_sparse[-1]:.1f}")
        report.append(f"Final success rate: {success_sparse[-1]:.1%}")
        report.append(f"Best mean return: {max(returns_sparse):.1f}")
        report.append(f"Best success rate: {max(success_sparse):.1%}")
        
        # Find convergence
        convergence_step = None
        for step, ret in zip(steps_sparse, returns_sparse):
            if ret >= solve_threshold:
                convergence_step = step
                break
        
        if convergence_step:
            report.append(f"Converged at: {convergence_step:,} steps ({convergence_step/1e6:.1f}M steps)")
        else:
            report.append(f"Did not converge to {solve_threshold} threshold")
        
        report.append("")
    
    # Analyze dense
    if learning_data_dense:
        report.append("DENSE REWARD TRAINING")
        report.append("-" * 60)
        steps_dense = sorted(learning_data_dense.keys())
        returns_dense = [learning_data_dense[s]['mean_return'] for s in steps_dense]
        success_dense = [learning_data_dense[s]['success_rate'] for s in steps_dense]
        
        report.append(f"Training points evaluated: {len(steps_dense)}")
        report.append(f"Final mean return: {returns_dense[-1]:.1f}")
        report.append(f"Final success rate: {success_dense[-1]:.1%}")
        report.append(f"Best mean return: {max(returns_dense):.1f}")
        report.append(f"Best success rate: {max(success_dense):.1%}")
        
        # Find convergence
        convergence_step = None
        for step, ret in zip(steps_dense, returns_dense):
            if ret >= solve_threshold:
                convergence_step = step
                break
        
        if convergence_step:
            report.append(f"Converged at: {convergence_step:,} steps ({convergence_step/1e6:.1f}M steps)")
        else:
            report.append(f"Did not converge to {solve_threshold} threshold")
        
        report.append("")
    
    # Comparison
    if learning_data_sparse and learning_data_dense:
        report.append("COMPARISON")
        report.append("-" * 60)
        
        steps_sparse = sorted(learning_data_sparse.keys())
        steps_dense = sorted(learning_data_dense.keys())
        returns_sparse = [learning_data_sparse[s]['mean_return'] for s in steps_sparse]
        returns_dense = [learning_data_dense[s]['mean_return'] for s in steps_dense]
        
        # Convergence speed comparison
        conv_sparse = None
        for step, ret in zip(steps_sparse, returns_sparse):
            if ret >= solve_threshold:
                conv_sparse = step
                break
        
        conv_dense = None
        for step, ret in zip(steps_dense, returns_dense):
            if ret >= solve_threshold:
                conv_dense = step
                break
        
        if conv_sparse and conv_dense:
            speedup = conv_sparse / conv_dense
            faster = "Dense" if speedup > 1 else "Sparse"
            report.append(f"Convergence speedup: {faster} is {abs(speedup - 1) * 100:.1f}% faster")
        
        # Final performance
        final_sparse = returns_sparse[-1]
        final_dense = returns_dense[-1]
        report.append(f"Final return difference: Dense - Sparse = {final_dense - final_sparse:+.1f}")
        
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    # Save report
    report_path = output_dir / 'convergence_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n✓ Saved report to: {report_path}")


def main():
    """Main function."""
    
    experiments_dir = Path(__file__).parent / "experiments"
    output_dir = experiments_dir / "learning_curves_analysis_base"
    
    print("Collecting learning curve data...")
    
    # Collect data from checkpoints
    learning_data_sparse = collect_learning_data(experiments_dir, "sparse")
    learning_data_dense = collect_learning_data(experiments_dir, "dense")
    
    if not learning_data_sparse and not learning_data_dense:
        print("ERROR: No checkpoint data found!")
        return
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_learning_curves(learning_data_sparse, learning_data_dense, output_dir)
    
    # Generate report
    print("Generating analysis report...")
    generate_convergence_report(learning_data_sparse, learning_data_dense, output_dir)
    
    print("\n✓ Learning curve analysis complete!")
    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
