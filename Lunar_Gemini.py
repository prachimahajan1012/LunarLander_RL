#!/usr/bin/env python3
"""
lunar_lander_rl_trainer.py

An interactive command-line application to train, watch, and analyze
a PPO agent learning to land in LunarLander-v2 using different reward regimes.
"""

import os
import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
import argparse
from scipy import stats
import json
import time
import glob

# --- Configuration ---
ENV_ID = 'LunarLander-v3' # FIXED: Changed from -v2 to -v3 to resolve DeprecatedEnv error
MODEL_SAVE_DIR = 'lunarlander_models'
LOG_DIR = 'lunarlander_logs'

# -------------------------
# Reward wrappers
# -------------------------
class SparseTerminalRewardEnv(gym.Wrapper):
    """Sparse reward: reward only at terminal state + small step penalty."""
    def __init__(self, env, success_threshold=100.0, step_penalty=-0.01):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.success_threshold = success_threshold
        
    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        reward = self.step_penalty
        done = terminated or truncated
        
        # The base environment gives the terminal reward. We check the final cumulative
        # reward to determine success/failure, though this is approximate for simplicity.
        # In this sparse regime, we only care about the step penalty and the final outcome.
        if done:
            # Check for success: if the final y-position is near zero and velocity is low.
            # A simpler check is to see if the base reward was high (LunarLander-v2 gives
            # +100 for landing, -100 for crashing).
            final_reward = 0
            if obs[6] and obs[7]: # Both legs contact
                 final_reward += 100.0
            if terminated and final_reward > 0:
                 reward += final_reward # Success reward
            else:
                 reward -= 100.0 # Failure/crash penalty
                 
        return obs, reward, terminated, truncated, info

class DenseFuelTerminalWrapper(gym.Wrapper):
    """Dense reward: penalize fuel usage per action (in addition to standard LL-v2 rewards)."""
    def __init__(self, env, fuel_cost=0.05):
        super().__init__(env)
        self.fuel_cost = fuel_cost

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Penalize for using the main engine (action 1) or side jets (actions 2, 3)
        if action != 0:
            reward -= self.fuel_cost
        return obs, reward, terminated, truncated, info

# -------------------------
# Utilities: train, eval, logs
# -------------------------
def make_env(reward_type='dense', seed=0, render_mode=None):
    """Create env with reward wrapper."""
    def _init():
        # Set render_mode for Gymnasium
        env = gym.make(ENV_ID, render_mode=render_mode)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
        # Apply wrappers based on reward type
        if reward_type == 'sparse':
            # Sparse environment: base reward is mostly zeroed out until terminal state
            env = SparseTerminalRewardEnv(env)
        elif reward_type == 'dense':
            # Dense environment: adds a fuel penalty to the original reward
            env = DenseFuelTerminalWrapper(env)
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")
        return env
    return _init

def evaluate_model(model, reward_type, n_eval_episodes=100, deterministic=True):
    """Evaluate a trained model."""
    env_fn = make_env(reward_type=reward_type, seed=9999)()
    returns = []
    successes = 0
    fuel_used = []

    for _ in range(n_eval_episodes):
        obs, _ = env_fn.reset()
        done = False
        total_r = 0.0
        fuel = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env_fn.step(action)
            done = terminated or truncated
            total_r += r
            if action != 0:
                fuel += 1
        returns.append(total_r)
        fuel_used.append(fuel)

        # Success check: check if both legs are in contact (index 6 and 7)
        if obs[6] and obs[7]:
            successes += 1
            
    env_fn.close()
    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'success_rate': successes / n_eval_episodes,
        'mean_fuel': np.mean(fuel_used)
    }

# -------------------------
# Training mode
# -------------------------
def train_model(reward_type, n_timesteps, seed, run_id):
    """Runs a single training session."""
    
    out_path = os.path.join(MODEL_SAVE_DIR, reward_type, f'run_{run_id}')
    os.makedirs(out_path, exist_ok=True)
    
    print(f"\n--- Starting Training: {reward_type.upper()} Reward (Run {run_id}, Seed {seed}) ---")
    print(f"Outputting files to: {out_path}")

    set_random_seed(seed)
    env_fn = make_env(reward_type=reward_type, seed=seed)
    env = DummyVecEnv([env_fn])

    # PPO Agent Setup
    model = PPO('MlpPolicy', env, verbose=1, seed=seed,
                n_steps=2048, batch_size=64, n_epochs=10, learning_rate=3e-4, clip_range=0.2)
    
    # Callbacks for Evaluation and Checkpoints
    eval_env_fn = make_env(reward_type=reward_type, seed=seed + 1234)
    eval_env = DummyVecEnv([eval_env_fn])
    
    eval_freq = int(n_timesteps / 20) # Evaluate 20 times during training
    
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=out_path,
                                 log_path=out_path, 
                                 eval_freq=eval_freq // env.num_envs,
                                 n_eval_episodes=50, 
                                 deterministic=True, 
                                 render=False)
    
    chk_callback = CheckpointCallback(save_freq=eval_freq // env.num_envs, 
                                      save_path=out_path,
                                      name_prefix='ckpt')
                                      
    start_time = time.time()
    
    # Start Learning!
    model.learn(total_timesteps=n_timesteps, callback=[eval_callback, chk_callback])
    
    elapsed = time.time() - start_time

    # Final evaluation and save
    stats = evaluate_model(model, reward_type, n_eval_episodes=100)
    
    model_file = os.path.join(out_path, f'final_ppo_model.zip')
    model.save(model_file)
    
    stats_file = os.path.join(out_path, f'final_stats.json')
    with open(stats_file, 'w') as f:
        json.dump({'seed': seed, 'reward_type': reward_type, 'n_timesteps': n_timesteps,
                   'elapsed_seconds': elapsed, 'final_model_path': model_file, **stats}, f, indent=2)
                   
    print(f"\n--- Training Complete in {elapsed:.2f} seconds ---")
    print(f"Final Success Rate: {stats['success_rate']:.2%}")
    print(f"Final Mean Return: {stats['mean_return']:.2f}")

# -------------------------
# Watch mode
# -------------------------
def watch_agent(model_path, reward_type, n_episodes):
    """Loads a trained model and watches it play in a GUI window."""
    print(f"\n--- Watching Agent from: {model_path} ---")
    print(f"Using Environment Reward Type: {reward_type.upper()}")
    
    # Create the environment with render_mode='human'
    env_fn = make_env(reward_type=reward_type, seed=42, render_mode='human')
    env = env_fn()
    
    # Load the trained model
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model path is correct and the file is a stable_baselines3 PPO model.")
        return
        
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        
        print(f"Episode {episode + 1}/{n_episodes} started...")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += r
            steps += 1
            
            # This is crucial for rendering the window
            env.render()
            time.sleep(0.01) # Small delay for better visualization
            
        success = "SUCCESS" if obs[6] and obs[7] else "FAILURE"
        print(f"Episode {episode + 1} finished in {steps} steps. Total Reward: {total_r:.2f}. Outcome: {success}")
        
    env.close()

# -------------------------
# Plot mode
# -------------------------
def plot_results():
    """Aggregates all results and plots learning curves and final stats."""
    
    print("\n--- Generating Plots and Statistics ---")
    
    all_stats_files = glob.glob(os.path.join(MODEL_SAVE_DIR, '**', 'final_stats.json'), recursive=True)
    
    if not all_stats_files:
        print("No final_stats.json files found. Please run 'train' mode first.")
        return
        
    rows = []
    for fpath in all_stats_files:
        try:
            with open(fpath,'r') as f:
                data = json.load(f)
            # Infer run_id from path structure
            reward_type = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
            run_id = os.path.basename(os.path.dirname(fpath)).replace('run_', '')
            data['run_id'] = int(run_id)
            data['reward_type'] = reward_type
            rows.append(data)
        except Exception as e:
            print(f"Skipping corrupt file {fpath}: {e}")
            
    if not rows:
        print("Could not load any valid statistics.")
        return
        
    df_final = pd.DataFrame(rows)
    df_final.to_csv(os.path.join(LOG_DIR, 'aggregate_final_results.csv'), index=False)
    print(f"Saved final aggregate results to {os.path.join(LOG_DIR, 'aggregate_final_results.csv')}")

    # Plotting Final Success Rate
    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    import seaborn as sns
    
    sns.barplot(x='reward_type', y='success_rate', data=df_final, palette=['#1f77b4', '#ff7f0e'])
    plt.title('Final Success Rate Comparison (across Seeds)')
    plt.ylabel('Success Rate')
    plt.xlabel('Reward Shaping Regime')
    plt.savefig(os.path.join(LOG_DIR, 'final_success_rate.png'))
    plt.close()
    print(f"Saved final_success_rate.png to {LOG_DIR}")
    
    # --- Learning Curve Plot (Optional, requires eval_logs) ---
    all_eval_logs = glob.glob(os.path.join(MODEL_SAVE_DIR, '**', 'evaluations.npz'), recursive=True)
    if all_eval_logs:
        print("Generating learning curves...")
        data_for_plot = []
        for fpath in all_eval_logs:
            data = np.load(fpath)
            timesteps = data['timesteps']
            mean_rewards = data['results'].mean(axis=1)
            
            # Infer reward_type and run_id
            path_parts = fpath.split(os.sep)
            # Expecting structure: lunarlander_models/sparse/run_0/evaluations.npz
            reward_type = path_parts[-3] 
            run_id = path_parts[-2].replace('run_', '')

            df_log = pd.DataFrame({
                'timesteps': timesteps,
                'mean_reward': mean_rewards,
                'reward_type': reward_type,
                'run_id': run_id
            })
            data_for_plot.append(df_log)

        if data_for_plot:
            df_logs = pd.concat(data_for_plot)
            plt.figure(figsize=(10, 6))
            sns.lineplot(x='timesteps', y='mean_reward', hue='reward_type', data=df_logs, errorbar='sd')
            plt.title('Learning Curves (Mean Evaluation Reward)')
            plt.xlabel('Timesteps')
            plt.ylabel('Mean Reward')
            plt.legend(title='Reward Type')
            plt.savefig(os.path.join(LOG_DIR, 'learning_curve.png'))
            plt.close()
            print(f"Saved learning_curve.png to {LOG_DIR}")
        
    # T-test for fun
    dense_vals = df_final[df_final['reward_type']=='dense']['mean_return'].values
    sparse_vals = df_final[df_final['reward_type']=='sparse']['mean_return'].values
    print("\n--- Summary Statistics ---")
    print(f"Dense Mean Return: {np.mean(dense_vals):.2f} (Std: {np.std(dense_vals):.2f})")
    print(f"Sparse Mean Return: {np.mean(sparse_vals):.2f} (Std: {np.std(sparse_vals):.2f})")
    
    if len(dense_vals) > 1 and len(sparse_vals) > 1:
        try:
            tstat, pval = stats.ttest_ind(dense_vals, sparse_vals, equal_var=False)
            print(f"T-test (dense vs sparse) T-statistic: {tstat:.2f}, P-value: {pval:.4f}")
        except Exception as e:
            print("T-test failed (need more seeds):", e)
    
    os.makedirs(LOG_DIR, exist_ok=True)


# -------------------------
# CLI entrypoint
# -------------------------
if __name__ == '__main__':
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="Interactive LunarLander RL Trainer.")
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # --- Training Subcommand ---
    train_parser = subparsers.add_parser('train', help='Train one or more agents.')
    train_parser.add_argument('reward_type', type=str, choices=['dense', 'sparse'], help='Reward shaping regime to use.')
    train_parser.add_argument('--n_timesteps', type=float, default=2e5, 
                              help='Total timesteps for training (e.g., 2e5).')
    train_parser.add_argument('--n_runs', type=int, default=3, help='Number of independent training runs (seeds).')
    train_parser.add_argument('--seeds_start', type=int, default=0, help='Starting seed for runs.')

    # --- Watch Subcommand ---
    watch_parser = subparsers.add_parser('watch', help='Watch a trained agent play.')
    watch_parser.add_argument('model_path', type=str, help='Path to the .zip file of the trained PPO model.')
    watch_parser.add_argument('reward_type', type=str, choices=['dense', 'sparse'], 
                              help='Reward type the agent was trained on (important for environment recreation).')
    watch_parser.add_argument('--n_episodes', type=int, default=5, help='Number of episodes to watch.')

    # --- Plot Subcommand ---
    plot_parser = subparsers.add_parser('plot', help='Aggregate results and generate plots.')

    args = parser.parse_args()

    if args.mode == 'train':
        seeds = [args.seeds_start + i for i in range(args.n_runs)]
        n_timesteps = int(args.n_timesteps)
        
        print(f"Preparing to train {args.n_runs} agents for {n_timesteps} timesteps each with {args.reward_type} reward.")
        
        for i, seed in enumerate(seeds):
            run_id = args.seeds_start + i
            train_model(args.reward_type, n_timesteps, seed, run_id)
        
        print("\nAll training runs complete. Use 'plot' to visualize results.")

    elif args.mode == 'watch':
        watch_agent(args.model_path, args.reward_type, args.n_episodes)
        
    elif args.mode == 'plot':
        plot_results()
        
    else:
        parser.print_help()