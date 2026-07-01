#!/usr/bin/env python
"""
rank.py — Entry Point Wrapper for Redrob Candidate Ranker
=========================================================
Runs the end-to-end preprocessing and ranking pipeline using the single command:
  python rank.py --candidates <path_to_jsonl> --out <path_to_csv>
"""

import os
import sys
import argparse
import subprocess
import time

def main():
    parser = argparse.ArgumentParser(description="Redrob Founding Team Senior AI Engineer Candidate Ranker")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates.jsonl file.")
    parser.add_argument("--out", type=str, required=True, help="Path to output submission.csv file.")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 60)
    print("  REDROB INTELLECTUAL CANDIDATE DISCOVERY & RANKING SYSTEM")
    print("=" * 60)
    print(f"Candidates Input : {args.candidates}")
    print(f"Submission Output: {args.out}")

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Use a workspace-local data/cache directory for intermediate pickle files
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Get absolute paths to scripts
    base_dir = os.path.dirname(os.path.abspath(__file__))
    preprocess_script = os.path.join(base_dir, "src", "preprocess.py")
    rank_script = os.path.join(base_dir, "src", "rank.py")

    # Step 1: Preprocessing
    print("\n[Stage 1/2] Running Preprocessing (Honeypot filters & fast down-selection)...")
    t0 = time.time()
    preprocess_cmd = [
        sys.executable,
        preprocess_script,
        "--candidates", args.candidates,
        "--output_dir", cache_dir
    ]
    
    # Run the subprocess
    res_preprocess = subprocess.run(preprocess_cmd)
    if res_preprocess.returncode != 0:
        print(f"\n[ERROR] Preprocessing stage failed with exit code {res_preprocess.returncode}", file=sys.stderr)
        sys.exit(res_preprocess.returncode)
    
    t_preprocess = time.time() - t0
    print(f"[SUCCESS] Preprocessing completed in {t_preprocess:.2f} seconds.")

    # Step 2: Ranking & Re-ranking
    print("\n[Stage 2/2] Running Hybrid Ranking (BM25 + CrossEncoder + RRF + Multipliers)...")
    t0 = time.time()
    rank_cmd = [
        sys.executable,
        rank_script,
        "--candidates", args.candidates,
        "--out", args.out,
        "--cache_dir", cache_dir
    ]
    
    res_rank = subprocess.run(rank_cmd)
    if res_rank.returncode != 0:
        print(f"\n[ERROR] Ranking stage failed with exit code {res_rank.returncode}", file=sys.stderr)
        sys.exit(res_rank.returncode)

    t_rank = time.time() - t0
    print(f"[SUCCESS] Ranking completed in {t_rank:.2f} seconds.")

    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"  Execution completed successfully in {total_time:.2f} seconds.")
    print("=" * 60)

if __name__ == "__main__":
    main()
