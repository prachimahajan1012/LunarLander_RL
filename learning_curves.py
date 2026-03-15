#!/usr/bin/env python3
"""
learning_curves.py

Generate learning curves comparing sparse vs dense reward training.

Uses:
- PPO checkpoints saved during training
- Reward wrappers and env creation imported from
  lunarlander_reward_shaping_experiment.py

Produces:
- Learning curve plots
- Convergence comparison
- Text summary report
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO

# ---- IMPORT FROM MAIN EXPERIMENT FILE ----
from lunarlander_reward_shaping_experiment import make_env


# ---------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------

def evaluate_checkpoint(
    model_path: str,
    reward_type: str,
    n_episodes: int = 20,
    seed: int = 0,
):
    """
    Evaluate a saved PPO checkpoint.

    Returns:
        dict with:
            mean_return
            std_return
            success_rate
    """
    model = PPO.load(model_path)

    env = make_env(reward_type=reward_type, seed=seed)()

    returns = []
    successes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_r = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_r += r

        returns.append(total_r)

        # success = landed on both legs
        try:
            if bool(obs[6]) and bool(obs[7]):
                successes += 1
        except Exception:
            pass

    env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": successes / n_episodes,
    }


def extract_step_from_ckpt(filename: str):
    """
    Extract training steps from checkpoint filename.

    Expected format:
        ckpt_dense_run0_500000_steps.zip
    """
    parts = filename.split("_")
    try:
        return int(parts[-2])
    except Exception:
        return None


def collect_learning_data(experiments_dir: Path, reward_type: str):
    """
    Evaluate all checkpoints for a given reward type.

    Returns:
        dict: step -> evaluation stats
    """
    reward_dir = experiments_dir / reward_type
    if not reward_dir.exists():
        print(f"[WARN] Directory not found: {reward_dir}")
        return {}

    checkpoints = sorted(reward_dir.glob("ckpt_*_steps.zip"))
    print(f"\n{reward_type.upper()} | Found {len(checkpoints)} checkpoints")

    data = {}

    for ckpt in checkpoints:
        step = extract_step_from_ckpt(ckpt.name)
        if step is None:
            continue

        print(f"Evaluating {ckpt.name} ...", end=" ")
        stats = evaluate_checkpoint(
            model_path=str(ckpt),
            reward_type=reward_type,
            n_episodes=10,
        )

        data[step] = stats
        print(
            f"return={stats['mean_return']:.1f}, "
            f"success={stats['success_rate']:.0%}"
        )

    return data


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_learning_curves(sparse_data, dense_data, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    def unpack(data):
        steps = sorted(data.keys())
        return (
            [s / 1e6 for s in steps],
            [data[s]["mean_return"] for s in steps],
            [data[s]["std_return"] for s in steps],
            [data[s]["success_rate"] for s in steps],
        )

    if sparse_data:
        xs, ys, ys_std, _ = unpack(sparse_data)
        plt.errorbar(xs, ys, yerr=ys_std, label="Sparse", marker="o")

    if dense_data:
        xd, yd, yd_std, _ = unpack(dense_data)
        plt.errorbar(xd, yd, yerr=yd_std, label="Dense", marker="s")

    plt.axhline(200, linestyle="--", color="red", alpha=0.6, label="Solved (200)")
    plt.xlabel("Training Steps (Millions)")
    plt.ylabel("Mean Episode Return")
    plt.title("Learning Curves: Sparse vs Dense Rewards")
    plt.legend()
    plt.tight_layout()

    outpath = output_dir / "learning_curves.png"
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"✓ Saved plot: {outpath}")


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def convergence_step(data, threshold=200):
    for step in sorted(data.keys()):
        if data[step]["mean_return"] >= threshold:
            return step
    return None


def generate_report(sparse_data, dense_data, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("=" * 60)
    report.append("LEARNING CURVE ANALYSIS")
    report.append("=" * 60)
    report.append("")

    for name, data in [("Sparse", sparse_data), ("Dense", dense_data)]:
        report.append(name.upper())
        report.append("-" * 40)

        if not data:
            report.append("No data available\n")
            continue

        steps = sorted(data.keys())
        final = data[steps[-1]]

        report.append(f"Final mean return: {final['mean_return']:.1f}")
        report.append(f"Final success rate: {final['success_rate']:.0%}")

        conv = convergence_step(data)
        if conv:
            report.append(f"Converged at: {conv/1e6:.2f}M steps")
        else:
            report.append("Did not converge (>=200)")

        report.append("")

    text = "\n".join(report)
    print("\n" + text)

    path = output_dir / "convergence_report.txt"
    path.write_text(text)
    print(f"✓ Saved report: {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    experiments_dir = Path(__file__).parent / "experiments"
    output_dir = experiments_dir / "learning_curves_analysis"

    print("Collecting checkpoint evaluations...")
    sparse_data = collect_learning_data(experiments_dir, "sparse")
    dense_data = collect_learning_data(experiments_dir, "dense")

    if not sparse_data and not dense_data:
        print("ERROR: No checkpoint data found.")
        return

    print("\nPlotting learning curves...")
    plot_learning_curves(sparse_data, dense_data, output_dir)

    print("Generating report...")
    generate_report(sparse_data, dense_data, output_dir)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
