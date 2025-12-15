# Getting Started with LunarLander Reward Shaper Web Interface

## Quick Start (Local Development)

### 1. Install Dependencies
```bash
pip install streamlit imageio matplotlib pandas
```

### 2. Run the Streamlit App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Docker Deployment

### 1. Build the Docker Image
```bash
docker build -t lunarlander-web:latest .
```

### 2. Run with Docker
```bash
docker run -p 8501:8501 -v $(pwd)/experiments:/app/experiments lunarlander-web:latest
```

### 3. Or use Docker Compose (Recommended)
```bash
docker-compose up -d
```

Access the app at `http://localhost:8501`

To stop:
```bash
docker-compose down
```

## Features

### Evaluation Tab
- **Reward Configuration**: Adjust sparse and dense reward parameters with sliders
- **Training Configuration**: Set training steps and evaluation episodes
- **Run Evaluation**: Execute evaluations with custom parameters
- **Real-time Progress**: Monitor evaluation progress

### Results Tab
- **Video Gallery**: View recorded evaluation videos
- **Statistics**: Display evaluation metrics (success rate, avg reward, etc.)
- **Directory Selection**: Browse results from different experimental runs

### Data Tab
- **Model Management**: View available trained models
- **Video Management**: Browse recorded videos
- **Storage Info**: See file sizes and organization

## How to Use

### Workflow 1: Evaluate Existing Models
1. Go to **Evaluation** tab
2. Set reward parameters using sliders
3. Configure evaluation settings
4. Click **Start Evaluation**
5. View results in **Results** tab with videos and stats

### Workflow 2: Compare Reward Settings
1. Select "Compare Both" for reward type
2. Adjust both sparse and dense parameters
3. Run evaluation
4. Compare videos side-by-side in Results tab

## Configuration

### Reward Parameters

**Sparse Rewards**:
- `Success Reward`: Points for successful landing (50-500, default 100)
- `Failure Reward`: Points for crash/failure (-500 to -10, default -100)

**Dense Rewards**:
- `Success Reward`: Points for successful landing (50-500, default 100)
- `Failure Reward`: Points for crash/failure (-500 to -10, default -100)
- `Fuel Penalty`: Penalty per timestep (-1.0 to 0.0, default -0.05)


### Video playback not working
Make sure `imageio` is installed:
```bash
pip install imageio
```

### Docker memory issues
Adjust memory limits in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 8G  # Increase as needed
```

## Advanced: Custom Model Training

To train models with custom rewards before using the web interface:

```bash
python lunarlander_reward_shaping_experiment.py \
    --n_seeds 1 \
    --n_timesteps 100000 \
    --record_videos \
    --video_dir experiments/videos_custom
```

## System Requirements

- **CPU**: 2+ cores (4+ recommended for training)
- **RAM**: 2GB minimum (4GB+ recommended)
- **Storage**: 2GB+ for models and videos
- **GPU**: Optional (will use if available)

## Deployment to Cloud

### AWS EC2
```bash
# Launch t3.large instance (2 vCPU, 8GB RAM)
# Install Docker and run:
docker-compose up -d
# Access via: http://<instance-ip>:8501
```

### Google Cloud Run
```bash
gcloud run deploy lunarlander-web \
    --source . \
    --platform managed \
    --port 8501 \
    --memory 4Gi
```

