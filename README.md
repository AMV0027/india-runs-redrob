# Redrob Candidate Discovery & Ranking System

This repository contains the submission for the Redrob Candidate Discovery & Ranking Challenge by team **Antigravity AI**.

Our solution is a high-performance, two-stage, CPU-only candidate ranking pipeline designed to process a large pool of candidate profiles (100,000+) under 5 minutes without any external network or GPU dependencies.

---

## 🏗️ Architecture Overview

The pipeline splits the workload into offline precomputation and real-time ranking stages:

1. **Offline Precomputation (`precompute.py`)**: 
   - Parallel feature extraction via `ProcessPoolExecutor`.
   - Generates structured candidate signals and saves them as a Snappy-compressed columnar Parquet file.
   - Builds L2-normalized TF-IDF sparse matrices for candidate skills and career experience.
   - Fits a `BM25Okapi` index over case-folded combined candidate profile texts.
   
2. **Real-time Ranking (`rank.py`)**:
   - **Hybrid Retrieval**: Extracts candidate sub-pools (top-2,000 each) using TF-IDF cosine similarity and BM25 queries, merging them to form a high-recall candidate pool.
   - **Additive Base Scoring**: Evaluates candidate suitability based on experience alignment, skill overlap, BM25 exact matches, years of experience curve, and platform engagement signals.
   - **Disqualification Multipliers**: Applies hard multiplicative filters (`0.0` to `0.7`) to penalize logic trap profiles (honeypots, ghosts, pure research, API wrappers, consulting-only, title chasers).
   - **Cross-Encoder Reranking**: Reranks the top 300 candidates using a local transformer model (`ms-marco-MiniLM-L-6-v2`) on CPU.
   - **Reasoning Generation**: Generates 1-2 sentence factual, non-hallucinated explanations for every candidate in the top 100.

---

## ⚡ Performance Summary

| Metric | Limit | Achieved (100k pool) | Margin |
|---|---|---|---|
| **Precomputation Time** | None (Offline) | **1 min 16 sec** | — |
| **Ranking Time (`rank.py`)** | ≤ 5 minutes | **49.2 seconds** | **83.6% Headroom** (5.9× faster than limit) |
| **Memory usage** | ≤ 16 GB RAM | **~4 GB RAM** | Safe within sandbox constraints |
| **Network & GPU** | Disabled | **CPU only, 100% Offline** | Verified |

---

## 📂 Repository Structure

```
├── docs/                      # Development documentation, slides, and answers
│   ├── edge_strategy.md       # Strategy for logic traps and edge cases
│   ├── hackathon_understanding.md
│   ├── implementation_plan.md
│   ├── research_notes.md
│   ├── slide_answers.md
│   ├── slides.md
│   └── walkthrough.md
├── challange_dataset/         # Organizer bundle containing data and validator
│   ├── candidates.jsonl       # Full candidate dataset (100,000 records)
│   ├── sample_candidates.json # 50 sample candidate profiles for testing
│   └── validate_submission.py # Official validator script
├── precompute.py              # Offline feature extraction & indexing script
├── rank.py                    # Runtime candidate scoring & reranking script
├── requirements.txt           # Dependency specifications
├── submission_metadata.yaml   # Portal metadata for Stage 3 validation
└── submission.csv             # Generated ranked top 100 output file
```

---

## 🚀 Getting Started & Reproduction

Follow these steps to run the pipeline and generate the submission:

### 1. Environment Setup

It is recommended to use a virtual environment:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Precomputation (Run once)

Run the precompute script to extract features and construct indexes from the dataset:

```bash
python precompute.py --input challange_dataset/candidates.jsonl --out_dir .
```

### 3. Generate Submission CSV

Run the ranking script to evaluate candidates and output the top 100:

```bash
python rank.py --candidates challange_dataset/candidates.jsonl --out ./submission.csv
```

### 4. Validate Format

Run the organizer's validation script to verify formatting compliance:

```bash
python challange_dataset/validate_submission.py submission.csv
```
