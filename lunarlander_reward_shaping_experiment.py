#!/usr/bin/env python3
"""
lunarlander_reward_shaping_experiment.py

Run PPO on LunarLander-v2 with two reward regimes:
 - sparse: reward only at episode end for landing success/failure
 - dense: shaped reward (original + fuel penalty)

Outputs:
 - saved models
 - CSV logs per run
 - matplotlib learning curves (avg across seeds)
"""

import os
import numpy as np
import gymnasium as gym  # Gymnasium replacement
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
import argparse
from scipy import stats
import json
import time

# -------------------------
# Reward wrappers
# -------------------------
class SparseTerminalRewardEnv(gym.Wrapper):
    """Sparse reward: only terminal + step penalty."""
    def __init__(self, env, success_reward=100.0, fail_reward=-100.0, step_penalty=-0.01):
        super().__init__(env)
        self.success_reward = success_reward
        self.fail_reward = fail_reward
        self.step_penalty = step_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self.step_penalty
        done = terminated or truncated
        if done:
            # LunarLander-v2 state: [pos.x pos.y vel.x vel.y angle angularVel leg1 contact leg2 contact]
            try:
                leg1 = bool(obs[6])
                leg2 = bool(obs[7])
            except Exception:
                leg1 = False
                leg2 = False
            if leg1 and leg2:
                reward += self.success_reward
            else:
                reward += self.fail_reward
        return obs, reward, terminated, truncated, info

class DenseFuelTerminalWrapper(gym.Wrapper):
    """Dense reward: penalize fuel usage per action."""
    def __init__(self, env, fuel_cost=0.05):
        super().__init__(env)
        self.fuel_cost = fuel_cost

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if action != 0:
            reward -= self.fuel_cost
        return obs, reward, terminated, truncated, info

# -------------------------
# Utilities: train, eval, logs
# -------------------------
def make_env(reward_type='dense', seed=0, render=False):
    """Create env with reward wrapper."""
    def _init():
        env = gym.make('LunarLander-v3')
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        if reward_type == 'sparse':
            env = SparseTerminalRewardEnv(env, success_reward=100.0, fail_reward=-100.0, step_penalty=-0.01)
        elif reward_type == 'dense':
            env = DenseFuelTerminalWrapper(env, fuel_cost=0.05)
        else:
            raise ValueError("unknown reward_type")
        # Monitor removed — not needed for Gymnasium
        return env
    return _init

def evaluate_model(model, env, n_eval_episodes=100, deterministic=True):
    returns = []
    successes = 0
    fuel_used = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        fuel = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += r
            if action != 0:
                fuel += 1
        returns.append(total_r)
        fuel_used.append(fuel)
        try:
            leg1 = bool(obs[6])
            leg2 = bool(obs[7])
        except:
            leg1 = False
            leg2 = False
        if leg1 and leg2:
            successes += 1
    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'success_rate': successes / n_eval_episodes,
        'mean_fuel': np.mean(fuel_used)
    }

# -------------------------
# Main experiment runner
# -------------------------
def run_experiment(output_dir='experiments', reward_type='dense', n_timesteps=int(2e6),
                   seed=0, run_id=0, eval_freq=50000, n_eval_episodes=50):
    os.makedirs(output_dir, exist_ok=True)
    set_random_seed(seed)
    env_fn = make_env(reward_type=reward_type, seed=seed)
    env = DummyVecEnv([env_fn])
    model = PPO('MlpPolicy', env, verbose=0, seed=seed,
                n_steps=2048, batch_size=64, n_epochs=10, learning_rate=3e-4, clip_range=0.2)
    # Eval callback
    eval_env_fn = make_env(reward_type=reward_type, seed=seed+1234)
    eval_env = DummyVecEnv([eval_env_fn])
    eval_callback = EvalCallback(eval_env, best_model_save_path=output_dir,
                                 log_path=output_dir, eval_freq=eval_freq // env.num_envs,
                                 n_eval_episodes=n_eval_episodes, deterministic=True, render=False)
    chk_callback = CheckpointCallback(save_freq=eval_freq // env.num_envs, save_path=output_dir,
                                      name_prefix=f'ckpt_{reward_type}_run{run_id}')
    start_time = time.time()
    model.learn(total_timesteps=n_timesteps, callback=[eval_callback, chk_callback])
    elapsed = time.time() - start_time

    final_eval_env = make_env(reward_type=reward_type, seed=seed+999)()
    stats = evaluate_model(model, final_eval_env, n_eval_episodes=100)
    # save model and stats
    model_file = os.path.join(output_dir, f'ppo_{reward_type}_run{run_id}_seed{seed}.zip')
    model.save(model_file)
    stats_file = os.path.join(output_dir, f'stats_{reward_type}_run{run_id}_seed{seed}.json')
    with open(stats_file, 'w') as f:
        json.dump({'seed': seed, 'reward_type': reward_type, 'n_timesteps': n_timesteps,
                   'elapsed_seconds': elapsed, **stats}, f, indent=2)
    return {'model_file': model_file, **stats}

# -------------------------
# Batch runner
# -------------------------
def batch_run(reward_type, seeds, outdir, n_timesteps):
    results = []
    for i, seed in enumerate(seeds):
        print(f"Running reward={reward_type} seed={seed} ({i+1}/{len(seeds)})")
        res = run_experiment(output_dir=outdir, reward_type=reward_type,
                             n_timesteps=n_timesteps, seed=seed, run_id=i)
        results.append(res)
    return results

def aggregate_results(experiment_dirs):
    import glob
    rows = []
    for d in experiment_dirs:
        files = glob.glob(os.path.join(d, 'stats_*.json'))
        for fpath in files:
            try:
                with open(fpath,'r') as f:
                    data = json.load(f)
                rows.append(data)
            except Exception as e:
                print("skip", fpath, e)
    return pd.DataFrame(rows)

def plot_metric(df, metric='mean_return', groupby='reward_type', outpath='plot.png'):
    import seaborn as sns
    sns.set(style='whitegrid')
    plt.figure(figsize=(8,6))
    sns.barplot(x=groupby, y=metric, data=df)
    plt.title(f'{metric} by {groupby}')
    plt.savefig(outpath)
    plt.close()

# -------------------------
# CLI entrypoint
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='experiments')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--n_timesteps', type=float, default=2e6)
    parser.add_argument('--seeds_start', type=int, default=0)
    args = parser.parse_args()

    seeds = [args.seeds_start + i for i in range(args.n_seeds)]
    dense_dir = os.path.join(args.outdir, 'dense')
    sparse_dir = os.path.join(args.outdir, 'sparse')

    print("=== Running dense experiments ===")
    dense_results = batch_run('dense', seeds, dense_dir, int(args.n_timesteps))

    print("=== Running sparse experiments ===")
    sparse_results = batch_run('sparse', seeds, sparse_dir, int(args.n_timesteps))

    # aggregate
    df = aggregate_results([dense_dir, sparse_dir])
    df.to_csv(os.path.join(args.outdir, 'aggregate_results.csv'), index=False)
    print("Saved aggregate_results.csv")

    # basic stats & t-test
    dense_vals = df[df['reward_type']=='dense']['mean_return'].values
    sparse_vals = df[df['reward_type']=='sparse']['mean_return'].values
    print("Dense mean_return:", np.mean(dense_vals), np.std(dense_vals))
    print("Sparse mean_return:", np.mean(sparse_vals), np.std(sparse_vals))
    try:
        tstat, pval = stats.ttest_ind(dense_vals, sparse_vals, equal_var=False)
        print("t-test (dense vs sparse) t=", tstat, "p=", pval)
    except Exception as e:
        print("t-test failed:", e)

    # plot
    try:
        import seaborn as sns
        plt.figure(figsize=(6,4))
        sns.boxplot(x='reward_type', y='mean_return', data=df)
        plt.title('Distribution of evaluation returns (across seeds)')
        plt.savefig(os.path.join(args.outdir, 'returns_boxplot.png'))
        plt.close()
        print("Saved returns_boxplot.png")
    except Exception as e:
        print("plotting failed:", e)
