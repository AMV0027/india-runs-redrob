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

def run_ranker(custom_file, use_sample):
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

    # Run subprocess rank.py
    cmd = [
        sys.executable,
        "-X", "utf8",
        "rank.py",
        "--candidates", candidates_file,
        "--out", output_csv
    ]

    print(f"Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    
    if res.returncode != 0:
        print(f"Pipeline failed:\n{res.stderr}\n{res.stdout}")
        return None, None

    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        return df, output_csv
    else:
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
