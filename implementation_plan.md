# Implementation Plan: Candidate Discovery & Ranking System

## Goal Description

The objective is to build an intelligent candidate ranking system for the "Senior AI Engineer — Founding Team" role. The system must process 100,000 candidate profiles and output the top 100 ranked candidates with a 1-2 sentence reasoning for each. 

Based on design feedback, we have adopted a robust **Hybrid Retrieval and Additive Scoring** architecture that balances vector semantics, exact keyword matching, and structured skill overlaps to maximize recall and ranking quality, while strictly adhering to the 5-minute CPU runtime constraint.

## Proposed Architecture

We will adopt a **Two-Stage Pipeline** with a multi-vector and hybrid retrieval strategy.

### Stage 1: Pre-Computation (Offline)
Since the 5-minute constraint applies only to the `reproduce_command`, we will heavily pre-compute features, vectors, and sparse indexes offline.
- **Multi-Vector Embeddings**: Instead of one monolithic text blob, we will generate multiple embeddings per candidate using `sentence-transformers`:
  - `skills_vector`: Embeddings of their explicit skills list.
  - `experience_vector`: Embeddings of their career history descriptions.
  - `profile_vector`: Embeddings of their headline and summary.
- **Sparse Indexing (BM25)**: Create a BM25 or TF-IDF index over the combined candidate text to capture exact keyword matches (e.g., "AWS", "RAG").
- **Structured Feature Extraction**:
  - `years_of_experience`: Total numeric years.
  - `behavioral_score`: A normalized composite of `redrob_signals` (response rate, recent login, etc.).
  - `consulting_penalty`: Soft penalty (e.g., `-0.1`) rather than a disqualifier.
  - `title_chaser_penalty`: Soft penalty for short tenures.
  - `honeypot_flag`: Boolean flag for mathematically impossible profiles.
- **Output**: FAISS indexes for each vector type, a serialized BM25 index, and a structured `features.parquet` file.

### Stage 2: Runtime Retrieval & Re-ranking (`rank.py`)
This script must run in under 5 minutes on a CPU.
1. **Query Processing**: 
   - Parse the JD into structured requirements (Skills: Python, NLP, LLMs, AWS; Exp: 5-9 years).
   - Compute JD vectors (`jd_skills_vector`, `jd_exp_vector`, `jd_profile_vector`).
2. **Hybrid Retrieval**: 
   - Retrieve top 2,000 via `skills_vector` FAISS.
   - Retrieve top 2,000 via `experience_vector` FAISS.
   - Retrieve top 2,000 via BM25 exact match.
   - Union the candidate IDs to form a robust candidate pool (avoids vector retrieval missing critical exact matches).
3. **Additive Scoring Re-ranking**: Score the candidate pool using an additive formula:
   ```python
   score = (0.35 * semantic_exp_match) +
           (0.25 * semantic_skills_match) +
           (0.15 * bm25_match) +
           (0.15 * structured_exp_match) +
           (0.10 * behavioral_score) -
           (0.05 * consulting_penalty) -
           (0.05 * title_chaser_penalty)
   ```
   *Note: Honeypots will have their final score severely penalized to drop them out of the top 100.*
4. **Component-Based Explainability**: Store the numeric contributions of each feature. Generate the reasoning string dynamically based on the highest contributing factors.
   ```text
   Ranked highly due to: 92% semantic experience alignment, strong behavioral signals, and 6 years of product engineering background.
   ```
5. **Output**: Write the top 100 to `submission.csv`.

## The "Hackathon Edge" Strategy (Optimizations to Win)

While the hybrid pipeline is solid, the top teams will go further to capture the *implicit* requirements of the JD. We will implement these three advanced optimizations to gain a competitive edge:

### 1. Offline NLP Classification for JD "Traps"
Instead of simple regex, we will use a lightweight zero-shot classifier (e.g., `facebook/bart-large-mnli` or `DeBERTa`) **offline** during pre-computation to evaluate the `description` fields of the candidate's career history and generate boolean flags:
- `is_pure_research`: "Did this person primarily publish papers without deploying to real users?"
- `is_langchain_wrapper`: "Is their AI experience limited to just using LangChain in the last 12 months?"
- `is_consulting_only`: "Has this person only worked at consulting/services firms?"
These flags will be used as **hard multipliers** (e.g., `score *= 0.1`) during runtime, perfectly capturing the JD's explicit disqualifiers.

### 2. Lightweight Cross-Encoder Reranking
Standard vector embeddings (like Sentence-BERT) often struggle with nuanced relationships (e.g., differentiating between a "Marketing Manager with AI skills" and an "AI Engineer"). 
- **Optimization**: After the FAISS/BM25 union retrieves the top ~500 candidates, we will run a fast **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over the candidate's combined text and the JD.
- **Why it wins**: Cross-encoders attend to the JD and candidate text simultaneously, yielding state-of-the-art semantic matching. Reranking 500 candidates on a CPU takes only ~10-15 seconds, comfortably fitting our 5-minute constraint while drastically improving precision.

### 3. Hard Behavioral & Honeypot Multipliers
Instead of a simple additive behavioral score, we will use a multiplicative approach for red flags:
- **The "Ghost" Filter**: If `redrob_signals.last_active_date` is > 6 months ago AND `recruiter_response_rate` < 0.1, their score multiplier is `0.0` (they are unavailable).
- **Honeypot Determinism**: If `skills[].duration_months == 0` while `proficiency == "expert"`, multiplier is `0.0`.
- **Tie-Breaker Hierarchy**: We will sort ties deterministically using `notice_period_days` (sub-30 prioritized) and `willing_to_relocate` (Noida/Pune prioritized), exactly as requested by the JD.

## Proposed Changes

### [NEW] `precompute.py`
Script to parse `candidates.jsonl`, extract multi-dimensional text features, compute multiple embeddings, build FAISS and BM25 indexes, and save structured metadata to `features.parquet`.

### [NEW] `rank.py`
The fast runtime script that performs hybrid retrieval (Union of FAISS and BM25), computes the additive score, generates traceable explanations, and writes `submission.csv`.

### [NEW] `submission_metadata.yaml`
Filled out template with team details and commands.

## Verification Plan

### Automated Tests
- Run `precompute.py` on `sample_candidates.json` to ensure features, multiple vectors, and BM25 indexes are generated correctly.
- Run `rank.py` on the sample indexes to ensure it completes well within the 5-minute budget and produces exactly 100 rows in the correct format.
- Run `python validate_submission.py submission.csv` (provided by organizers) to guarantee the format is perfect.

### Local Quality Check
- Verify that the explanation string correctly highlights the top scoring components (e.g., if BM25 match is high, it mentions exact keyword matches).
- Verify that honeypots in the sample data are successfully filtered.

## Why it Wins (Speed & Problem Solving)

**1. Is it fast? (The 5-Minute CPU Constraint)**
Our architecture is designed precisely for the 5-minute limit:
- **Offline Heavy Lifting**: We process all 100,000 candidates offline. We generate embeddings, parse resumes, and run NLP classifiers locally without time limits.
- **Lightning Fast Online Ranking**:
  - FAISS + BM25 retrieve the top 500 candidates in **~1 to 2 seconds**.
  - The Cross-Encoder reranks those 500 candidates on a CPU in **~10 to 15 seconds**.
  - Final score calculation and saving to CSV takes **~1 second**.
- **Total Runtime**: **~15-20 seconds**, well within the 300-second limit.

**2. Does it solve the problem? (The Logic Traps)**
The organizers stated keyword matching is a "trap" and test for the JD's *true intent*:
- **Trap 1 (Honeypot)**: Instantly dropped due to impossible stats checks (0 months used but expert proficiency).
- **Trap 2 (The "Ghost" Candidate)**: Dropped if inactive for 6 months and ignoring recruiter messages.
- **Trap 3 ("Pure Research" vs "Production")**: Cross-Encoder understands the semantic difference; offline classifier adds hard penalty to pure research backgrounds.
- **Trap 4 (Title Chasers / Non-Engineers)**: Separation of `title` ensures "Marketing Managers" with AI keywords don't rank highly.
