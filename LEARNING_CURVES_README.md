# Learning Curve Analysis Guide

## Overview

I've created two Python scripts to extract learning curves from your trained models:

### 1. **`learning_curves_from_evals.py`** (RECOMMENDED - Fast)
- Uses existing evaluation checkpoints stored in `.npz` files
- Runs in seconds
- Best for quick analysis and comparison
- **Run:** `python learning_curves_from_evals.py`

### 2. **`learning_curves.py`** (Comprehensive - Slower)
- Re-evaluates all checkpoints by loading each model
- Takes longer (hours) but provides more detailed statistics
- Gives episode-by-episode breakdown for each checkpoint
- **Run:** `python learning_curves.py`

## Key Findings from Your Models

### Quick Summary

| Metric | Sparse | Dense |
|--------|--------|-------|
| **Convergence Step** | ✗ Did not converge | 550,000 steps (0.6M) |
| **Final Return** | 92.0 | 241.4 |
| **Final Success Rate** | ~0% | 92% |
| **Training Range** | 50K - 2M steps | 50K - 2M steps |

### Key Insights

1. **Dense Rewards Converge Faster**
   - Dense reward function converged to the 200-point threshold at 550,000 steps
   - Sparse never reached 200 points even after 2M steps
   - Dense is much more stable and achieves higher final performance

2. **Performance Gap**
   - Dense has **+149.4 points** advantage over sparse in final performance
   - Dense achieves 92% success rate vs ~0% for sparse
   - This suggests reward shaping significantly helps learning

3. **Learning Dynamics**
   - Sparse starts at -100 (all failures, since few episodes succeed)
   - Dense starts at -992 but rapidly improves after 50K steps
   - Dense shows smooth, steady improvement; sparse remains stuck

## Output Files

All results are saved to: `experiments/learning_curves_analysis/`

- **`learning_curves_from_evaluations.png`** - Comparison plots with:
  - Mean episode return over training
  - Median return (robust to outliers)
  - Success rate progression
  - Final performance comparison

- **`detailed_analysis_report.txt`** - Text report with:
  - Early vs late training statistics
  - Convergence analysis
  - Performance comparison metrics

## What the Plots Show

### Plot 1: Mean Episode Return ± Std Dev
- Shows how average return improves during training
- Shaded areas represent ±1 standard deviation
- Red dashed line at 200 marks "solved" threshold
- **Takeaway:** Dense reaches solved threshold; sparse doesn't

### Plot 2: Median Episode Return
- More robust measure (ignores outliers)
- Easier to see convergence behavior
- **Takeaway:** Dense has consistent high performance; sparse stays low

### Plot 3: Success Rate (Return > 100)
- Percentage of episodes that "succeed"
- **Takeaway:** Dense reaches 92% success; sparse stays near 0%

### Plot 4: Final Performance
- Bar chart comparing final returns
- **Takeaway:** Dense significantly outperforms sparse

## Comparing Reward Shaping Effects

Based on these learning curves, you can clearly see:

1. **Reward shaping helps learning speed** - Dense converges 3.6x faster (550K vs 2M steps)
2. **Reward shaping improves final performance** - Dense achieves 2.6x higher return
3. **Sparse rewards are harder to learn from** - Much sparser signal makes learning difficult
4. **Dense rewards enable landing skill** - Achieves 92% landing success vs 0% for sparse

## How to Modify the Analysis

Edit the scripts to customize:

**`learning_curves_from_evals.py`:**
- Change `SOLVED_THRESHOLD = 200` to different convergence criteria
- Modify success rate calculation: `success_count = np.sum(episode_returns > 100)`
- Adjust plot styling in `plot_learning_curves()` function

**`learning_curves.py`:**
- Change evaluation episodes: `n_episodes=10` parameter
- Modify reward wrapper parameters in `get_wrapper()`

## Next Steps

You can further analyze by:

1. **Plotting individual runs** - Check variance across different random seeds
2. **Extracting specific timesteps** - Get exact return values at key training milestones
3. **Creating episode-by-episode plots** - See distribution of episode returns
4. **Analyzing failure modes** - Examine what kinds of failures are most common

Would you like me to create any of these additional analysis scripts?
