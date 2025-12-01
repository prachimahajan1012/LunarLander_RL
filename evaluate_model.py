from stable_baselines3 import PPO
from lunarlander_reward_shaping_experiment import make_env, evaluate_model

def main():
    model_path = "experiments/dense/best_model.zip"   # change as needed
    model = PPO.load(model_path, device="cpu")
    # If you want videos, set record_video=True and video_dir to a folder
    eval_env = make_env(reward_type='dense', seed=999, record_video=True, video_dir="experiments/videos_eval_dense2")()
    stats = evaluate_model(model, eval_env, n_eval_episodes=100, deterministic=True)
    print(stats)

if __name__ == "__main__":
    main()