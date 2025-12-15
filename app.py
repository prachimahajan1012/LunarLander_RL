"""
Streamlit web interface for LunarLander reward shaping experimentation.
Allows interactive tweaking of reward parameters and visualization of results.
"""

import streamlit as st
import os
import sys
import json
import subprocess
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
import tempfile
import shutil

# Page config
st.set_page_config(
    page_title="LunarLander Reward Shaper",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🚀 LunarLander Reward Shaper")
st.markdown("Tweak reward parameters and visualize training results in real-time")

# Project paths
PROJECT_ROOT = Path(__file__).parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
VIDEOS_DIR = EXPERIMENTS_DIR / "videos_custom"

# Create videos directory if it doesn't exist
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# Sidebar for reward configuration
st.sidebar.header("⚙️ Reward Configuration")

reward_type = st.sidebar.radio(
    "Select Reward Type",
    options=["Sparse", "Dense", "Compare Both"],
    help="Choose which reward configuration to test"
)

col1, col2 = st.sidebar.columns(2)

if reward_type in ["Sparse", "Compare Both"]:
    with col1:
        st.subheader("Sparse Rewards")
        sparse_success = st.slider(
            "Success Reward",
            min_value=50.0,
            max_value=500.0,
            value=100.0,
            step=10.0,
            key="sparse_success"
        )
        sparse_fail = st.slider(
            "Failure Reward",
            min_value=-500.0,
            max_value=-10.0,
            value=-100.0,
            step=10.0,
            key="sparse_fail"
        )

if reward_type in ["Dense", "Compare Both"]:
    with col2:
        st.subheader("Dense Rewards")
        dense_success = st.slider(
            "Success Reward",
            min_value=50.0,
            max_value=500.0,
            value=100.0,
            step=10.0,
            key="dense_success"
        )
        dense_fail = st.slider(
            "Failure Reward",
            min_value=-500.0,
            max_value=-10.0,
            value=-100.0,
            step=10.0,
            key="dense_fail"
        )
        fuel_penalty = st.slider(
            "Fuel Penalty",
            min_value=-1.0,
            max_value=0.0,
            value=-0.05,
            step=0.01,
            key="fuel_penalty"
        )

# Training parameters
st.sidebar.header("🎯 Training Configuration")

col1, col2 = st.sidebar.columns(2)
with col1:
    n_timesteps = st.number_input(
        "Training Steps",
        min_value=5000,
        max_value=500000,
        value=50000,
        step=5000
    )

with col2:
    eval_steps = st.number_input(
        "Evaluation Episodes",
        min_value=10,
        max_value=100,
        value=50,
        step=5
    )

use_existing_model = st.sidebar.checkbox(
    "Use Existing Model",
    value=False,
    help="Use pre-trained model instead of training from scratch"
)

# Main content area
tab1, tab2, tab3 = st.tabs(["🎮 Evaluation", "📊 Results", "📁 Data"])

with tab1:
    st.header("Run Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Evaluation Parameters:**
        - Timesteps: {n_timesteps:,}
        - Eval Episodes: {eval_steps}
        - Type: {reward_type}
        """)
    
    with col2:
        if st.button("▶️ Start Evaluation", key="run_eval", use_container_width=True):
            st.session_state.running = True
    
    if st.session_state.get("running"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        output_area = st.empty()
        
        status_text.info("⏳ Setting up evaluation environment...")

        # Clear previous videos so each evaluation run overwrites the directory
        try:
            for p in VIDEOS_DIR.iterdir():
                try:
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p)
                except Exception:
                    # ignore individual file removal errors
                    pass
            status_text.info("🗑️ Cleared previous videos from the output directory")
        except Exception as e:
            status_text.warning(f"Could not clear videos directory: {e}")

        try:
            # Prepare evaluation command - use the same Python executable as Streamlit
            python_executable = sys.executable
            cmd_args = [
                python_executable,
                str(PROJECT_ROOT / "evaluate_model.py"),
                "--eval_episodes", str(eval_steps),
                "--reward_type", reward_type.lower(),
            ]
            
            if reward_type in ["Sparse", "Compare Both"]:
                cmd_args.extend([
                    "--sparse_success", str(sparse_success),
                    "--sparse_fail", str(sparse_fail),
                ])
            
            if reward_type in ["Dense", "Compare Both"]:
                cmd_args.extend([
                    "--dense_success", str(dense_success),
                    "--dense_fail", str(dense_fail),
                    "--fuel_penalty", str(fuel_penalty),
                ])
            
            cmd_args.extend([
                "--video_dir", str(VIDEOS_DIR),
                "--record_videos",
            ])
            
            status_text.info("🚀 Running evaluation...")
            progress_bar.progress(50)
            
            # Debug: Print the command being run
            print(f"Running command: {' '.join(cmd_args)}")
            print(f"Working directory: {PROJECT_ROOT}")
            
            # Run evaluation
            result = subprocess.run(
                cmd_args,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            # Print output to terminal for debugging
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            
            progress_bar.progress(100)
            
            if result.returncode == 0:
                status_text.success("✅ Evaluation completed successfully!")
                
                # Display output
                if result.stdout:
                    with output_area.expander("📝 Output Details"):
                        st.code(result.stdout, language="text")
            else:
                status_text.error(f"❌ Evaluation failed with return code {result.returncode}")
                with output_area.expander("📝 Error Details"):
                    error_output = result.stderr if result.stderr else result.stdout
                    if error_output:
                        st.code(error_output, language="text")
                    else:
                        st.warning("No error output captured. Check the Streamlit terminal for details.")
        
        except subprocess.TimeoutExpired:
            status_text.error("⏱️ Evaluation timed out")
        except Exception as e:
            status_text.error(f"❌ Error: {str(e)}")
        
        st.session_state.running = False

with tab2:
    st.header("Results & Visualization")
    
    # # List available video directories
    # subdirs = [d for d in EXPERIMENTS_DIR.iterdir() 
    #            if d.is_dir() and d.name.startswith("videos_")]
    # subdirs = sorted(subdirs, key=lambda x: x.stat().st_mtime, reverse=True)
    
    # if not subdirs:
    #     st.warning("No evaluation results found yet. Run an evaluation first!")
    # else:
    #     selected_dir = st.selectbox(
    #         "Select Results Directory",
    #         options=[d.name for d in subdirs],
    #         index=0
    #     )
        
    result_path = EXPERIMENTS_DIR / "videos_custom"
        
        # Get videos in this directory
    videos = list(result_path.glob("*.mp4"))
        
    if videos:
        st.subheader(f"📹 Videos in videos custom")
        
        # Create columns for video grid
        cols = st.columns(2)
        
        for idx, video_file in enumerate(sorted(videos)[:10]):  # Show first 10
            with cols[idx % 2]:
                st.video(str(video_file))
                st.caption(f"Episode {video_file.stem}")
    else:
        st.info("No videos found in this directory")
        
    # Try to load stats
    json_files = list(result_path.glob("*.json"))
    if json_files:
        st.subheader("📊 Statistics")
        
        stats_data = {}
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    stats_data[json_file.stem] = json.load(f)
            except:
                pass
        
        if stats_data:
            # Display stats in expandable sections
            for key, stats in stats_data.items():
                with st.expander(f"📈 {key}"):
                    cols = st.columns(3)
                    for col, (stat_name, stat_value) in enumerate(stats.items()):
                        with cols[col % 3]:
                            if isinstance(stat_value, (int, float)):
                                st.metric(stat_name, f"{stat_value:.2f}")

with tab3:
    st.header("📁 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Available Models")
        
        model_dirs = [d for d in EXPERIMENTS_DIR.iterdir() 
                     if d.is_dir() and not d.name.startswith("videos_")]
        
        if model_dirs:
            for model_dir in sorted(model_dirs):
                model_files = list(model_dir.glob("*.zip"))
                st.write(f"**{model_dir.name}**")
                if model_files:
                    for mf in model_files:
                        st.caption(f"  • {mf.name} ({mf.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            st.info("No models found")
    
    with col2:
        st.subheader("Video Directories")
        
        video_dirs = [d for d in EXPERIMENTS_DIR.iterdir() 
                     if d.is_dir() and d.name.startswith("videos_")]
        
        if video_dirs:
            for vd in sorted(video_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
                videos_count = len(list(vd.glob("*.mp4")))
                st.write(f"**{vd.name}** - {videos_count} videos")
        else:
            st.info("No video directories found")

# Footer
st.divider()
st.markdown("""
---
**LunarLander Reward Shaper** | Made with Streamlit 🎈
""")

# Initialize session state
if "running" not in st.session_state:
    st.session_state.running = False
