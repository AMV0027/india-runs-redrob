"""
run_pipeline.py — Master Pipeline Orchestrator
===============================================
Runs the entire candidate discovery pipeline sequentially and logs
consolidated execution times for each stage and the total end-to-end run.

Usage:
  python run_pipeline.py
"""

import subprocess
import time
import sys

def run_command(command, description):
    print(f"\n>>> Running: {description}...")
    t0 = time.time()

    # Use sys.executable to ensure we use the same python interpreter
    process = subprocess.Popen([sys.executable] + command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')

    # Stream the output to the console in real-time
    for line in process.stdout:
        print(f"  {line}", end='')

    process.wait()
    duration = time.time() - t0

    if process.returncode != 0:
        print(f"\n[ERROR] {description} failed with exit code {process.returncode}")
        sys.exit(process.returncode)

    return duration

if __name__ == "__main__":
    print("=" * 60)
    print("  REDROB INTELLECTUAL CANDIDATE DISCOVERY PIPELINE RUNNER")
    print("=" * 60)

    t_start = time.time()

    # Stage 1: Preprocessing
    time_preprocess = run_command(
        ["src/preprocess.py", "--candidates", "challange_dataset/candidates.jsonl", "--output_dir", "data/cache"],
        "Stage 1: Preprocessing (is_honeypot, is_blacklisted & Down-selection to 1800)"
    )

    # Stage 2: Ranking
    time_rank = run_command(
        ["src/rank.py", "--candidates", "challange_dataset/candidates.jsonl", "--out", "submission.csv", "--cache_dir", "data/cache"],
        "Stage 2: Ranking (BM25 + CrossEncoder + RRF + Multipliers)"
    )

    # Stage 3: Local Evaluation
    time_eval = run_command(
        ["src/local_eval.py", "--submission", "submission.csv", "--candidates", "challange_dataset/candidates.jsonl"],
        "Stage 3: Local Evaluation (Sanity checks & local NDCG/MAP report)"
    )

    total_duration = time.time() - t_start

    print("\n" + "=" * 60)
    print("  CONSOLIDATED PERFORMANCE METRICS")
    print("=" * 60)
    print(f"  - Preprocessing   : {time_preprocess:.2f} seconds")
    print(f"  - Ranking         : {time_rank:.2f} seconds")
    print(f"  - Local Evaluation: {time_eval:.2f} seconds")
    print(f"  - Total Pipeline  : {total_duration:.2f} seconds")
    print("=" * 60)
