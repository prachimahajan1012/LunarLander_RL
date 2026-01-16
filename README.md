# Getting Started with LunarLander Reward Shaper Web Interface
This code trains a RL agent to solve the LunarLander game under two reward settings: Sparse and Dense.

## Quick Start

### 1. Install Dependencies
Create a virtual environment and install all packages in it.
```bash
pip install -r requirements.txt
```

## Run Model Training

Train models with custom rewards, adjust seeds and timesteps, (file runs training for both sparse and dense one after the other by default):

```bash
python lunarlander_reward_shaping_experiment.py \
    --n_seeds 5 \   
    --n_timesteps 2000000 \
    --record_videos \               # will slow down training, better to check videos later
    --video_dir experiments/videos_custom
```

## Evaluation

To evaluate a trained model simply launch our UI with:
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

## How to Use the UI

1. Go to **Evaluation** tab
2. Select the reward type you want to run evaluation for.
3. Choose model checkpoint that you want to evaluate on.
4. Choose the number of evaluation episodes you want to see.
5. Click **Evaluate**
6. View videos. 
7. Go to **Results** tab to view the rest of the videos and some stats.


To run evaluation without using the UI run:

```bash
python evaluate_model.py \
    --reward_type sparse \
    --eval_episodes 10 \
    --record_videos \ 
    --video_dir experiments/videos_custom
```

Replace `sparse` with `dense` for dense reward evaluation. Uses the best model from your trained directory by default. 







