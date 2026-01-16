"""
Streamlit web interface for viewing LunarLander evaluation results (videos).
Allows model checkpoint selection.
"""

import streamlit as st
import sys
import json
import subprocess
from pathlib import Path
import shutil
import re

st.set_page_config(
    page_title="LunarLander Reward Shaper",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 LunarLander Reward Shaper")
st.markdown("Choose reward type and visualize model evaluation results.")

# Project paths
PROJECT_ROOT = Path(__file__).parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
VIDEOS_DIR = EXPERIMENTS_DIR / "videos_custom"

# Create videos directory if it doesn't exist
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# Helper functions for checkpoint management
def get_checkpoint_steps(reward_dir):
    """Extract checkpoint step counts from model files."""
    checkpoints = {}
    for model_file in reward_dir.glob("ckpt_*_*_steps.zip"):
        # Parse filename: ckpt_<type>_seed<N>_<steps>_steps.zip
        match = re.search(r'_(\d+)_steps\.zip$', model_file.name)
        if match:
            steps = int(match.group(1))
            checkpoints[steps] = model_file
    return checkpoints

def get_every_fifth_checkpoint(checkpoints):
    """Get every 5th checkpoint from the sorted checkpoints."""
    sorted_steps = sorted(checkpoints.keys())
    # Get every 5th checkpoint (cause there are too many)
    every_fifth = [sorted_steps[i] for i in range(0, len(sorted_steps), 5)]
    return every_fifth

def get_checkpoint_options(reward_type):
    """Get list of checkpoint options"""
    reward_dir = EXPERIMENTS_DIR / reward_type.lower()
    
    if not reward_dir.exists():
        return []
    
    options = []
    checkpoints = get_checkpoint_steps(reward_dir)
    
    if not checkpoints:
        return []

    every_fifth = get_every_fifth_checkpoint(checkpoints)
    
    for steps in every_fifth:
        options.append((f"{steps:,} steps", steps, "checkpoint"))
    
    # Add best model if it exists
    best_model = reward_dir / "best_model.zip"
    if best_model.exists():
        options.append(("Best Model", "best", "best"))
    
    return options

st.sidebar.header("⚙️ Model Selection")

reward_type = st.sidebar.radio(
    "Select Reward Type",
    options=["Sparse", "Dense", "Compare Both"],
    help="Choose which reward configuration to evaluate"
)

# Checkpoint selector - gets options based on selected reward type
if reward_type == "Compare Both":
    checkpoint_options = get_checkpoint_options("sparse")
else:
    checkpoint_options = get_checkpoint_options("dense")

if checkpoint_options:
    checkpoint_labels = [opt[0] for opt in checkpoint_options]
    selected_checkpoint_label = st.sidebar.selectbox(
        "Select Checkpoint",
        options=checkpoint_labels,
        help="Choose a checkpoint to evaluate"
    )
    
    # Find the selected checkpoint
    selected_checkpoint = None
    for opt in checkpoint_options:
        if opt[0] == selected_checkpoint_label:
            selected_checkpoint = opt[1]
            break
else:
    st.sidebar.warning(f"No checkpoints found")
    selected_checkpoint = None

st.sidebar.divider()
st.sidebar.header("⚙️ Evaluation Configuration")

col1 = st.sidebar.columns(1)[0]

with col1:
    eval_steps = st.number_input(
        "Evaluation Episodes",
        min_value=10,
        max_value=100,
        value=50,
        step=5
    )

# Main content area
tab1, tab2 = st.tabs(["🎮 Evaluation", "📊 Results"])

with tab1:
    st.header("Run Evaluation")
    
    if selected_checkpoint is None:
        st.error("No checkpoint selected. Please select a checkpoint from the sidebar.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"""
            **Evaluation Parameters:**
            - Checkpoint: {selected_checkpoint_label}
            - Eval Episodes: {eval_steps}
            - Type: {reward_type}
            """)
        
        with col2:
            if st.button("▶️ Evaluate", key="run_eval", use_container_width=True):
                st.session_state.running = True
                st.session_state.eval_type = "single"
        
        if st.session_state.get("running"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            output_area = st.empty()
            
            eval_type = st.session_state.get("eval_type", "single")
            
            if reward_type == "Compare Both":
                status_text.info("⏳ Setting up comparison evaluation for both sparse and dense...")
                rewards_to_eval = ["Sparse", "Dense"]
            else:
                status_text.info(f"⏳ Setting up evaluation for {reward_type}...")
                rewards_to_eval = [reward_type]
            
            try:
                # Clear previous videos
                try:
                    for p in VIDEOS_DIR.iterdir():
                        try:
                            if p.is_file():
                                p.unlink()
                            elif p.is_dir():
                                shutil.rmtree(p)
                        except Exception:
                            pass
                    status_text.info("Cleared previous videos")
                except Exception as e:
                    status_text.warning(f"Could not clear videos directory: {e}")
                
                all_videos = {}
                
                for idx, reward in enumerate(rewards_to_eval):
                    status_text.info(f"🚀 Running evaluation for {reward} model...")
                    progress_bar.progress(int((idx / len(rewards_to_eval)) * 100))
                    
                    python_executable = sys.executable
                    cmd_args = [
                        python_executable,
                        str(PROJECT_ROOT / "evaluate_model.py"),
                        "--eval_episodes", str(eval_steps),
                        "--reward_type", reward.lower(),
                        "--checkpoint", str(selected_checkpoint),
                        "--video_dir", str(VIDEOS_DIR),
                        "--record_videos",
                    ]
                    
                    result = subprocess.run(
                        cmd_args,
                        cwd=PROJECT_ROOT,
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    
                    print(f"Return code: {result.returncode}")
                    if result.stdout:
                        print("STDOUT:", result.stdout)
                    if result.stderr:
                        print("STDERR:", result.stderr)
                    
                    if result.returncode == 0:
                        # Get videos for this reward type
                        reward_video_dir = VIDEOS_DIR / reward.lower()
                        if reward_video_dir.exists():
                            videos = sorted(reward_video_dir.glob("*.mp4"))
                            all_videos[reward] = videos
                    else:
                        status_text.error(f"❌ Evaluation failed for {reward}")
                        with output_area.expander("📝 Error Details"):
                            error_output = result.stderr if result.stderr else result.stdout
                            if error_output:
                                st.code(error_output, language="text")
                
                progress_bar.progress(100)
                status_text.success("✅ Evaluation completed!")
                
                # Display videos
                if all_videos:
                    if reward_type == "Compare Both" and "Sparse" in all_videos and "Dense" in all_videos:
                        # Side-by-side comparison
                        st.subheader("📹 Sparse vs Dense Comparison")
                        sparse_videos = all_videos["Sparse"]
                        dense_videos = all_videos["Dense"]
                        
                        max_episodes = min(len(sparse_videos), len(dense_videos), 5)
                        
                        for i in range(max_episodes):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Sparse - Episode {i}**")
                                st.video(str(sparse_videos[i]))
                            with col2:
                                st.write(f"**Dense - Episode {i}**")
                                st.video(str(dense_videos[i]))
                    else:
                        # Single reward type videos
                        for reward, videos in all_videos.items():
                            st.subheader(f"📹 {reward} Videos")
                            cols = st.columns(2)
                            for idx, video_file in enumerate(sorted(videos)[:10]):
                                with cols[idx % 2]:
                                    st.video(str(video_file))
                                    st.caption(f"Episode {video_file.stem}")
                
            except subprocess.TimeoutExpired:
                status_text.error("⏱️ Evaluation timed out")
            except Exception as e:
                status_text.error(f"❌ Error: {str(e)}")
            
            st.session_state.running = False

with tab2:
    st.header("Results & Visualization")
    
    result_path = EXPERIMENTS_DIR / "videos_custom"
    
    # Check for reward subdirectories
    reward_subdirs = {}
    if result_path.exists():
        for subdir in result_path.iterdir():
            if subdir.is_dir():
                videos = list(subdir.glob("*.mp4"))
                if videos:
                    reward_subdirs[subdir.name] = sorted(videos)
    
    # Also check for videos directly in videos_custom
    direct_videos = []
    if result_path.exists():
        direct_videos = [v for v in result_path.glob("*.mp4") if v.is_file()]
    
    if reward_subdirs:
        st.subheader("📹 Recent Evaluation Videos")
        for reward_type, videos in reward_subdirs.items():
            with st.expander(f"**{reward_type.capitalize()} Videos** ({len(videos)} videos)", expanded=True):
                cols = st.columns(2)
                for idx, video_file in enumerate(sorted(videos)[:10]):
                    with cols[idx % 2]:
                        st.video(str(video_file))
                        st.caption(f"Episode {video_file.stem}")
    elif direct_videos:
        st.subheader(f"📹 Videos")
        cols = st.columns(2)
        for idx, video_file in enumerate(sorted(direct_videos)[:10]):
            with cols[idx % 2]:
                st.video(str(video_file))
                st.caption(f"Episode {video_file.stem}")
    else:
        st.info("No evaluation videos found. Run an evaluation first!")
    
    # Try to load and display stats
    stats_files = []
    if result_path.exists():
        for subdir in result_path.iterdir():
            if subdir.is_dir():
                stats = list(subdir.glob("*.json"))
                stats_files.extend(stats)
        stats_files.extend(list(result_path.glob("*.json")))
    
    if stats_files:
        st.subheader("📊 Evaluation Statistics")
        
        for stats_file in sorted(stats_files):
            try:
                with open(stats_file) as f:
                    stats_data = json.load(f)
                    with st.expander(f"📈 {stats_file.stem}"):
                        cols = st.columns(3)
                        for col_idx, (stat_name, stat_value) in enumerate(stats_data.items()):
                            with cols[col_idx % 3]:
                                if isinstance(stat_value, (int, float)):
                                    st.metric(stat_name, f"{stat_value:.4f}" if isinstance(stat_value, float) else str(stat_value))
                                else:
                                    st.write(f"**{stat_name}**: {stat_value}")
            except Exception as e:
                st.warning(f"Could not load stats from {stats_file.name}: {e}")


st.divider()

# Initialize session state
if "running" not in st.session_state:
    st.session_state.running = False
if "eval_type" not in st.session_state:
    st.session_state.eval_type = "single"
