import os
import sys
import tempfile
import time
import subprocess
import pandas as pd
import gradio as gr

def get_file_path(file_input):
    if file_input is None:
        return None
    if isinstance(file_input, str):
        return file_input
    if isinstance(file_input, list):
        if len(file_input) == 0:
            return None
        return get_file_path(file_input[0])
    if isinstance(file_input, dict):
        return file_input.get("path")
    if hasattr(file_input, "path"):
        return file_input.path
    if hasattr(file_input, "name"):
        return file_input.name
    return None

# Ensure src is in python path to resolve internal imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src.preprocess import run_preprocessing
from src.rank import run_ranking

def run_ranker(custom_file, use_sample, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Initializing pipeline...")
    
    # Set candidate path
    if use_sample:
        candidates_file = os.path.join("challange_dataset", "sample_candidates.json")
    else:
        candidates_file = get_file_path(custom_file)
        if not candidates_file:
            return None, None

    # Setup output CSV path in a temporary directory
    temp_dir = tempfile.mkdtemp()
    output_csv = os.path.join(temp_dir, "submission.csv")
    cache_dir = os.path.join(temp_dir, "cache")

    try:
        progress(0.05, desc="Preprocessing candidates (honeypot & blacklist checking)...")
        run_preprocessing(candidates_file, cache_dir)
        
        progress(0.40, desc="Running hybrid ranking (BM25 + ONNX CrossEncoder + RRF)...")
        run_ranking(candidates_file, output_csv, cache_dir)
        
        progress(0.95, desc="Generating output...")
        
        if os.path.exists(output_csv):
            df = pd.read_csv(output_csv)
            progress(1.0, desc="Ranking completed!")
            return df, output_csv
        else:
            return None, None
    except Exception as e:
        print(f"Error during execution: {e}")
        return None, None

# Define Gradio blocks interface
with gr.Blocks(title="Redrob AI — Candidate Ranker Sandbox") as demo:
    gr.Markdown("# 🎯 Redrob AI — Candidate Discovery & Ranking Sandbox")
    gr.Markdown("This hosted sandbox runs the candidate discovery and ranking pipeline end-to-end on CPU.")
    
    with gr.Row():
        with gr.Column(scale=1):
            use_sample = gr.Checkbox(label="Use preloaded 50-candidate sample (sample_candidates.json)", value=True)
            custom_file = gr.File(label="Or upload custom candidate JSON/JSONL sample", file_types=[".json", ".jsonl"])
            run_btn = gr.Button("🚀 Run Candidate Ranker", variant="primary")
            
        with gr.Column(scale=2):
            output_df = gr.Dataframe(label="🏆 Top Ranked Candidates")
            output_file = gr.File(label="📥 Download Submission CSV")

    run_btn.click(
        fn=run_ranker,
        inputs=[custom_file, use_sample],
        outputs=[output_df, output_file]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
