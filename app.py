import streamlit as st
import os
import sys
import tempfile
import pandas as pd
import time
import subprocess

st.set_page_config(
    page_title="Redrob AI — Candidate Ranker Sandbox",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Redrob AI — Candidate Discovery & Ranking Sandbox")
st.markdown("""
This hosted sandbox environment runs the **The Open Dev** team's candidate discovery pipeline end-to-end on CPU.
Upload a sample candidate JSONL/JSON file (up to 100 candidates) or use our preloaded 50-candidate sample to verify results.
""")

# Setup Sidebar Info
st.sidebar.header("Pipeline Specifications")
st.sidebar.markdown("""
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (ONNX optimized)
- **Execution**: 100% Offline (locally packaged weights/tokenizer)
- **Features**: BM25 Lexical + Cross-Encoder Semantic + RRF Fusion + 11 Behavioral Multipliers
- **Limit**: CPU only, ≤ 5 minutes execution
""")

# File source selection
option = st.radio(
    "Choose candidate data source:",
    ("Use preloaded 50-candidate sample (sample_candidates.json)", "Upload a custom candidate sample file")
)

candidates_file = None
temp_dir = tempfile.TemporaryDirectory()

if option == "Use preloaded 50-candidate sample (sample_candidates.json)":
    preloaded_path = os.path.join("challange_dataset", "sample_candidates.json")
    if os.path.exists(preloaded_path):
        candidates_file = preloaded_path
        st.success("Preloaded 50-candidate sample selected.")
    else:
        st.error(f"Preloaded sample file not found at {preloaded_path}. Please upload a file instead.")
else:
    uploaded_file = st.file_uploader("Upload candidates file (.json or .jsonl)", type=["json", "jsonl"])
    if uploaded_file is not None:
        file_ext = ".jsonl" if uploaded_file.name.endswith(".jsonl") else ".json"
        candidates_file = os.path.join(temp_dir.name, f"uploaded_candidates{file_ext}")
        with open(candidates_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Successfully uploaded: {uploaded_file.name}")

if candidates_file:
    if st.button("🚀 Run Candidate Ranker", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Define output CSV path
        output_csv = os.path.join(temp_dir.name, "submission_output.csv")
        
        status_text.text("Initializing pipeline and model weights...")
        progress_bar.progress(10)
        
        # Run wrapper rank.py as a subprocess to capture logs
        # We pass -X utf8 to handle Windows unicode prints
        cmd = [
            sys.executable,
            "-X", "utf8",
            "rank.py",
            "--candidates", candidates_file,
            "--out", output_csv
        ]
        
        log_area = st.empty()
        log_content = []
        
        t_start = time.time()
        
        # Execute process and stream logs
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        
        progress_bar.progress(30)
        status_text.text("Running Preprocessing & ONNX Re-ranking...")
        
        for line in process.stdout:
            log_content.append(line)
            # Display running logs in a code block
            log_area.code("".join(log_content[-15:])) # show last 15 lines of logs
            
            # Update progress based on log outputs
            if "Preprocessing completed" in line:
                progress_bar.progress(50)
                status_text.text("Preprocessing done. Scoring candidates with Cross-Encoder...")
            elif "gated candidates on Current Role" in line:
                progress_bar.progress(70)
                status_text.text("Scoring current & past roles...")
            elif "Writing results" in line:
                progress_bar.progress(90)
                status_text.text("Generating reasoning & writing CSV...")
                
        process.wait()
        duration = time.time() - t_start
        
        if process.returncode == 0 and os.path.exists(output_csv):
            progress_bar.progress(100)
            status_text.text(f"Pipeline executed successfully in {duration:.2f} seconds!")
            
            # Load and display results
            df = pd.read_csv(output_csv)
            st.subheader(f"🏆 Top Ranked Candidates (Found {len(df)} candidates)")
            st.dataframe(df, use_container_width=True)
            
            # Download Button
            with open(output_csv, "rb") as file:
                btn = st.download_button(
                    label="📥 Download Submission CSV",
                    data=file,
                    file_name="submission.csv",
                    mime="text/csv"
                )
        else:
            st.error(f"Pipeline execution failed with exit code {process.returncode}.")
            st.code("".join(log_content))

temp_dir.cleanup()
