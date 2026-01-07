#!/usr/bin/env python3
"""
evaluate_model.py

Evaluate a trained PPO model on LunarLander with custom reward parameters.
Supports both sparse and dense rewards.
"""

import os
import sys
import argparse
import json
import numpy as np
import gymnasium as gym
from pathlib import Path
from datetime import datetime
import torch

# Suppress TensorFlow warnings if using stable-baselines3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ---- Reward Wrappers (from main experiment) ----

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


def get_model_path(reward_type, sparse_success=None, sparse_fail=None, 
                  dense_success=None, dense_fail=None, fuel_penalty=None):
    """Get or create path to model directory."""
    
    experiments_dir = Path(__file__).parent / "experiments"
    experiments_dir.mkdir(exist_ok=True)
    
    if reward_type == "sparse":
        model_dir = experiments_dir / f"sparse_s{sparse_success}_f{sparse_fail}"
    else:
        model_dir = experiments_dir / f"dense_s{dense_success}_f{dense_fail}_f{fuel_penalty}"
    
    model_dir.mkdir(exist_ok=True)
    return model_dir


def evaluate_model(model_path, env, num_episodes=50, record_videos=False, video_dir=None, reward_type=""):
    """
    Evaluate a model and optionally record videos.
    
    Args:
        model_path: Path to saved model
        env: Environment to evaluate on
        num_episodes: Number of episodes to evaluate
        record_videos: Whether to record videos
        video_dir: Directory to save videos
        reward_type: Type of reward (sparse/dense) for video subdirectory
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    # Load model
    try:
        model = PPO.load(str(model_path))
        model.set_env(env)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        # For video recording
        frames = []
        if record_videos and video_dir:
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            except:
                pass
        
        while not done:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            # Record frame
            if record_videos and video_dir:
                try:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                except:
                    pass
        
        # Check if successful (both legs on ground)
        try:
            leg1 = bool(obs[6])
            leg2 = bool(obs[7])
            if leg1 and leg2:
                success_count += 1
        except:
            pass
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Save video
        if record_videos and video_dir and frames:
            try:
                save_video_frames(frames, video_dir, f"episode_{episode:04d}", reward_type)
            except Exception as e:
                print(f"Error saving video for episode {episode}: {e}")
        
        # Progress
        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{num_episodes}")
    
    # Calculate statistics
    stats = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "success_rate": float(success_count / num_episodes),
        "num_episodes": num_episodes,
        "timestamp": datetime.now().isoformat()
    }
    
    return stats


def save_video_frames(frames, video_dir, name, reward_type=""):
    """Save frames as MP4 video using imageio."""
    try:
        import imageio
        
        video_dir = Path(video_dir)
        if reward_type:
            video_dir = video_dir / reward_type.lower()
        video_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = video_dir / f"{name}.mp4"
        
        # Convert frames to uint8 if needed
        frames_array = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
                frames_array.append(frame)
        
        if frames_array:
            imageio.mimsave(str(output_path), frames_array, fps=30)
            print(f"Saved video: {output_path}")
    except ImportError:
        print("Warning: imageio not installed. Skipping video save.")
    except Exception as e:
        print(f"Error saving video: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LunarLander model with custom rewards")
    
    parser.add_argument("--eval_episodes", type=int, default=50,
                       help="Number of episodes to evaluate")
    parser.add_argument("--sparse_success", type=float, default=100.0,
                       help="Success reward for sparse")
    parser.add_argument("--sparse_fail", type=float, default=-100.0,
                       help="Failure reward for sparse")
    parser.add_argument("--dense_success", type=float, default=100.0,
                       help="Success reward for dense")
    parser.add_argument("--dense_fail", type=float, default=-100.0,
                       help="Failure reward for dense")
    parser.add_argument("--fuel_penalty", type=float, default=-0.05,
                       help="Fuel penalty for dense")
    parser.add_argument("--reward_type", choices=["sparse", "dense"], default="sparse",
                       help="Which reward type to evaluate")
    parser.add_argument("--checkpoint", type=str, default="best",
                       help="Checkpoint to load: 'best' for best_model.zip, or step count for checkpoint (e.g., '1000000')")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to saved model. If not provided, uses default location")
    parser.add_argument("--record_videos", action="store_true",
                       help="Record evaluation videos")
    parser.add_argument("--video_dir", type=str, default=None,
                       help="Directory to save videos")
    parser.add_argument("--render_mode", choices=["human", "rgb_array"], default="rgb_array",
                       help="Environment render mode")
    
    args = parser.parse_args()
    
    # Setup environment
    print(f"Setting up {args.reward_type} reward environment...")
    base_env = gym.make("LunarLander-v3", render_mode=args.render_mode)
    
    if args.reward_type == "sparse":
        env = SparseTerminalRewardEnv(
            base_env,
            success_reward=args.sparse_success,
            fail_reward=args.sparse_fail
        )
    else:
        env = DenseRewardEnv(
            base_env,
            success_reward=args.dense_success,
            fail_reward=args.dense_fail,
            fuel_penalty=args.fuel_penalty
        )
    
    # Determine model path
    if args.model_path is None:
        experiments_dir = Path(__file__).parent / "experiments"
        reward_type_lower = args.reward_type.lower()
        
        # Determine which directory (sparse, sparse2, dense)
        if reward_type_lower == "sparse":
            # Check sparse2 first, then sparse
            for model_dir_name in ["sparse2", "sparse"]:
                model_dir = experiments_dir / model_dir_name
                if model_dir.exists():
                    break
            else:
                model_dir = experiments_dir / "sparse"
        else:
            model_dir = experiments_dir / "dense"
        
        # Determine which checkpoint file
        if args.checkpoint == "best":
            model_path = model_dir / "best_model.zip"
        else:
            # args.checkpoint is a step count
            checkpoint_pattern = f"ckpt_{reward_type_lower}_run0_{args.checkpoint}_steps.zip"
            model_path = model_dir / checkpoint_pattern
    else:
        model_path = Path(args.model_path)
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Available models:")
        experiments_dir = Path(__file__).parent / "experiments"
        for d in experiments_dir.glob("*"):
            if d.is_dir():
                models = list(d.glob("*.zip"))
                if models:
                    print(f"  {d.name}:")
                    for m in sorted(models)[:10]:
                        print(f"    - {m.name}")
        sys.exit(1)
    
    # Run evaluation
    print(f"Evaluating {args.reward_type} model from {model_path}")
    print(f"Episodes: {args.eval_episodes}")
    
    stats = evaluate_model(
        str(model_path),
        env,
        num_episodes=args.eval_episodes,
        record_videos=args.record_videos,
        video_dir=args.video_dir,
        reward_type=args.reward_type
    )
    
    if stats:
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        # Save stats
        if args.video_dir:
            video_dir_path = Path(args.video_dir)
            reward_type_dir = video_dir_path / args.reward_type.lower()
            reward_type_dir.mkdir(parents=True, exist_ok=True)
            
            stats_path = reward_type_dir / f"stats_{args.reward_type}_{args.eval_episodes}_episodes.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\nStats saved to: {stats_path}")
    
    env.close()


if __name__ == "__main__":
    main()