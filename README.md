# Redrob AI — Senior AI Engineer Candidate Ranking System

![Banner_Image](https://i.ibb.co/gZPsps3n/github-banner.png)

This document explains **exactly** how the ranking system works, what every decision means, why each number was chosen, and how the final output is produced.

---

## 🎯 The Goal

We have **~100,000 synthetic candidate profiles** stored in `candidates.jsonl`. Each profile describes a person — their job history, skills, education, and behavioural signals on the Redrob platform. Our task is to find and output the **top 100 candidates** who are the best fit for this specific role:

> **Senior AI Engineer (Founding Team)** at Redrob AI
> Location: Pune / Noida, Hybrid
> Experience: 5–9 years (flexible)

The submission must be a CSV file with exactly 100 rows, each with a `candidate_id`, a `rank` (1 to 100), a `score` (floating-point, must be non-increasing), and a `reasoning` sentence.

There is one strict hardware constraint: **the ranking script must complete in under 5 minutes on a CPU, with no internet access.**

---

## 🏗️ Pipeline Architecture

```
india-runs-redrob/
├── challange_dataset/
│   ├── candidates.jsonl (100,000 profiles)
│   ├── sample_candidates.json
│   └── validate_submission.py
├── data/
│   └── cache/
│       └── preprocessed_data.pkl
├── models/
│   └── cross-encoder/
│       ├── model.onnx
│       ├── model.onnx.data
│       └── [tokenizer configs/vocab files]
├── src/
│   ├── preprocess.py
│   ├── rank.py
│   ├── rerank.py
│   ├── utils.py
│   ├── local_eval.py
│   └── extract_profiles.py
├── rank.py (Root entry-point wrapper)
├── run_pipeline.py
├── submission_metadata.yaml
└── submission.csv (Generated CSV submission file)

[rank.py] — Root Entry-point Wrapper
  ├── Executes src/preprocess.py (Honeypot/blacklist filters & down-selection → 1800)
  └── Executes src/rank.py (BM25 + CrossEncoder + RRF + Multipliers → submission.csv)

[src/preprocess.py] — Stage 1 Preprocessing
  ├── Stage 1.1: Honeypot Detection
  ├── Stage 1.2: Consulting Blacklist Filter
  ├── Stage 1.3: Heuristic Down-selection → Top 1,800
  └── saves: data/cache/preprocessed_data.pkl
        │
        ▼
[src/rank.py] — Stage 2 Ranking (Runs in < 3 mins offline)
  ├── Stage 2.1: Dynamic JD Parsing
  ├── Stage 2.2: BM25 Lexical Retrieval (3 segments)
  ├── Stage 2.3: Cross-Encoder Semantic Scoring (3 segments) [ONNX Offline-loaded]
  ├── Stage 2.4: RRF Rank Fusion (CE + BM25 + Title + ATS)
  ├── Stage 2.5: Soft Penalties + Behavioral Multipliers (11 signals)
  └── writes: submission.csv
```

**`src/rerank.py`** — imported by `src/rank.py`. Contains the `CrossEncoderReranker` class and the enriched candidate text builder. Re-engineered to run on **ONNX Runtime (ORT)**, featuring dynamic shape exporting and thread-tuned CPU parallelism to minimize ranking latency.

**`src/utils.py`** — shared utilities. Contains all keyword constants, capability groups, honeypot detection, blacklist checks, feature extraction, ATS scoring, JD parsing, and reasoning generation.

---

## 📋 Step-by-Step: What `preprocess.py` Does

### Stage 1 — Honeypot Detection (`utils.py → is_honeypot()`)

The dataset contains profiles with **logically impossible timelines** planted as traps. Any submission that ranks more than 10 honeypots in the top 100 is automatically **disqualified**.

We run five mathematical checks on every profile:

**Check 1 — Total job months vs. stated experience**
Sum of `duration_months` across all jobs must not exceed `(years_of_experience × 12) + 12`. A buffer of 12 months accounts for overlapping jobs and transitions.

**Check 2 — Career span date vs. stated experience**
The calendar span between the earliest `start_date` and latest `end_date` across all jobs must not exceed `years_of_experience + 2.0` years.

**Check 3 — Individual skill duration vs. stated experience**
No single skill's `duration_months` should exceed `(years_of_experience × 12) + 6`.

**Check 4 — Expert skills with no experience**
If a candidate claims ≥ 8 `expert`/`advanced` proficiency skills but total skill duration across all skills is < 12 months, they are flagged.

**Check 5 — Education graduation year vs. years_of_experience**
Using 2026 as the reference year, maximum possible experience = `(2026 - latest_graduation_year) + 1`. If the stated `years_of_experience` exceeds this by more than 2 years, they are flagged.

---

### Stage 2 — Consulting Blacklist (`utils.py → is_blacklisted()`)

A **two-layer check** per job in the candidate's career:

**Layer 1 — Name-based**: Matches against 17 real consulting/IT services firms:

```
TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL,
Tech Mahindra, L&T, Mindtree, Mphasis, NTT Data, UST Global,
EY, PwC, Deloitte, KPMG
```

Plus 16 fictional/honeypot companies: Dunder Mifflin, Pied Piper, Hooli, Stark Industries, Wayne Enterprises, Acme Corp, Initech, etc.

**Layer 2 — Industry-based**: Flags jobs whose `industry` field matches:

```
IT Services, IT Consulting, Consulting, Outsourcing,
Managed Services, Staffing, BPO, KPO
```

**Rule**: A candidate is discarded only if **every single job** in their career was at a blacklisted firm or consulting industry. A candidate who spent 4 years at TCS then joined Swiggy is **not** discarded (but will receive a score penalty at ranking time). Profiles with no career history are also discarded.

---

### Stage 3 — Heuristic Down-Selection to Top 1,800

We score all ~80,000 remaining profiles using `extract_candidate_features()` (described below) and fast rule-based heuristics across these dimensions:

| Dimension | What is scored |
| --------- | -------------- |

### Stage 3 — Down-Selecting to 1,800 (The Heuristic Scorer)

We score each of the 80,000 profiles using fast, rule-based heuristics across 11 dimensions using the centralized Feature Extractor to select the top 1,800. No AI embedding happens here.
The top 1,800 candidates are written to a cache for the re-ranker.

---

## 🧮 How the Final Scores Are Calculated (`rank.py`)

### 1. `preprocess.py` — Runs Once Offline (No Time Limit)

This script does all the heavy lifting before submission time. It filters out bad candidates, selects the most promising 1,800, and saves the preprocessed text segments to `data/cache/preprocessed_data.pkl`.

**This step takes approximately 26 seconds total.**

### 2. `rank.py` — Runs at Submission Time (Must Finish in < 5 Minutes)

This script orchestrates the following stages:

- **Stage 2 (Coarse Retrieval)**: Runs BM25 Okapi indexing across the cached summaries and roles (~1s).
- **Stage 3 (Headline Semantic Filtering)**: Runs all 1,800 candidates jointly with the Job Description through the locally-packaged ONNX `models/cross-encoder` on just the short **Headline/Summary Segment** (~52s).
- **Stage 4 (Dynamic Gating)**: Combines Headline CE, Title Relevance, ATS integrity, and BM25 Headline scores to select the **top 700 candidates** for deep career history evaluation.
- **Stage 5 (Deep Semantic Re-Ranking)**: Evaluates the heavier **Current Role** and **Past Roles Segments** _only_ for the gated top 700 candidates (~51s).
- **Stage 6 (Final Blend)**: Blends the resulting three-segment Cross-Encoder score (60%) with lexical and structural channels (40%), applies behavioral multipliers, and writes `submission.csv` (~5s).

**This step takes approximately 129 seconds total (well under the 300s limit).**

---

### Stage 2 — BM25 Lexical Retrieval

BM25Okapi indexes are built over 3 segmented text fields per candidate:

| Segment            | Content                                | RRF Weight |
| ------------------ | -------------------------------------- | ---------- |
| Headline / Summary | Candidate's self-description           | 0.20       |
| Current Role       | Current job title + description        | 0.50       |
| Past Roles         | All previous job titles + descriptions | 0.30       |

Query: 15 domain keywords (`vector search`, `pinecone`, `qdrant`, `ndcg`, `llm`, etc.).

---

### Stage 3 — Cross-Encoder Semantic Gating & Re-Ranking

Model: **`cross-encoder/ms-marco-MiniLM-L-6-v2`** (66M parameters, CPU-optimized)

The full JD is evaluated against each candidate's enriched profile text. To process 1,800 candidates within the 5-minute CPU constraint, we implement a **two-phase dynamic gating router**:

1. **Headline Phase**: All 1,800 candidates are scored on their short **Headline/Summary Segment**.
2. **Intermediate Rank Gating**: We combine this early Headline CE score with fast-channel scores (BM25, Title relevance, ATS integrity) to construct a gating rank.
3. **Deep Career Phase**: Only the **top 700 gated candidates** proceed to have their heavier **Current Role** and **Past Roles** segments evaluated.

**Enriched Candidate Text** (built by `build_candidate_text()` in `rerank.py`):

```
{title} at {company}. {years_exp} years of experience.
Core Capabilities: {top 2 capability groups by strength}.
Top Skills: {top 8 skills by duration}.
Achievements & Production Experience: {sentences mentioning: led/managed/built/scaled/optimized/deployed}.
```

This gives the Cross-Encoder ownership-level evidence rather than raw keyword lists.

**Soft capability gating**: If a candidate has a `0` strength score in a JD must-have capability group, their CE score is scaled by `× 0.70` (soft confidence adjustment, not hard exclusion).

---

### Stage 4 — Reciprocal Rank Fusion (RRF)

RRF combines the rank positions of 4 scoring channels (standard constant k = 60):

```
RRF_Score = (0.60 / (60 + R_CE))
           + (0.25 / (60 + R_BM25))
           + (0.10 / (60 + R_Title))
           + (0.05 / (60 + R_ATS))
```

| Channel | Weight | Description                 |
| ------- | ------ | --------------------------- |
| R_CE    | 0.60   | Cross-Encoder semantic rank |
| R_BM25  | 0.25   | BM25 lexical keyword rank   |
| R_Title | 0.10   | Tiered title relevance rank |
| R_ATS   | 0.05   | ATS resume-integrity rank   |

The raw RRF score is **min-max normalized** to `[0.0, 1.0]`.

**Title Tiering** (for R_Title):

- Tier 1 (AI, NLP, Search, RAG): Senior → +4.5, Standard → +2.0, Junior → −2.0
- Tier 2 (ML, Machine Learning, Applied Scientist): Senior → +3.0, Standard → +1.0, Junior → −3.0
- CV/Robotics/Unrelated titles: −5.0

**ATS Resume-Integrity Score** (for R_ATS):
| Component | Weight | What it checks |
|---|---|---|
| Skill Coverage | 40% | Relevant skills matched against domain keywords |
| Career Stability | 30% | Average tenure per job (< 15 months = 0.5, > 36 months = 1.2 bonus) |
| Gap Penalty | 15% | Career gaps > 12 months between consecutive jobs (score = 0.6) |
| Career Progression | 15% | Junior → Senior promotion detected = 1.25 bonus |

---

### Stage 5 — Soft Penalties & Behavioral Multipliers

**Soft Multiplicative Penalties** (applied to normalized RRF score):

| Condition                                       | Multiplier |
| ----------------------------------------------- | ---------- |
| Forbidden skills matched (accounting, HR, etc.) | `× 0.50`   |
| CV/Robotics dominance (≥ 3 CV skills)           | `× 0.75`   |
| ≥ 50% career at consulting/blacklisted firms    | `× 0.70`   |

**Search Depth Vocabulary Bonus** (added to relevance score):
Up to +0.20 for matching terms: `semantic search`, `learning to rank`, `hybrid search`, `reranking`, `dense retrieval`, `vector search`, `ndcg`, `mrr`, `information retrieval`, etc.

**Behavioral Multipliers** (multiplied together into `avail_mult`):

| Signal               | Field                        | Multiplier Logic                                                       |
| -------------------- | ---------------------------- | ---------------------------------------------------------------------- |
| Recency              | `last_active_date`           | ≤ 45 days: ×1.15 / > 120 days: ×0.65 / > 180 days: ×0.40               |
| Response Rate        | `recruiter_response_rate`    | ≥ 0.70: ×1.10 / < 0.40: ×0.80 / < 0.20: ×0.30                          |
| Response Time        | `avg_response_time_hours`    | ≤ 12h: ×1.08 / ≤ 48h: ×1.04 / > 200h: ×0.90                            |
| Location             | `location`, `country`        | Pune/Noida: ×1.20 / NCR: ×1.18 / Relocate: ×1.05 / Overseas: ×0.35     |
| Notice Period        | `notice_period_days`         | 0d: ×1.18 / ≤15d: ×1.15 / ≤30d: ×1.10 / ≤90d: ×0.85 / >90d: ×0.70      |
| GitHub               | `github_activity_score`      | > 80: ×1.25 / > 50: ×1.15 / −1 (no GitHub): ×0.75                      |
| Open to Work         | `open_to_work_flag`          | False: ×0.70                                                           |
| Skill Assessment     | `skill_assessment_scores`    | ≥ 75 on relevant skill: ×1.12 / ≥ 65: ×1.06 / < 40: ×0.88              |
| Recruiter Saves      | `saved_by_recruiters_30d`    | ≥ 8: ×1.10 / ≥ 4: ×1.05 / ≥ 2: ×1.02                                   |
| Application Activity | `applications_submitted_30d` | ≥ 5: ×1.05 / ≥ 2: ×1.02 / 0 apps: ×0.97                                |
| Interview Rate       | `interview_completion_rate`  | ≥ 0.85: ×1.06 / < 0.65: ×0.90 / < 0.50: ×0.80                          |
| Certifications       | `certifications[].name`      | Domain-relevant cert (ML, LLM, AWS, GCP, etc.): ×1.08                  |
| Work Mode            | `preferred_work_mode`        | Onsite: ×1.04 / Remote + no relocate: ×0.88 / Remote + relocate: ×0.95 |

**Experience Floor**: Candidates with < 5.0 years experience receive a final `× 0.65` penalty.

**Final Score Formula**:

```
Final_Score = RRF_Normalized × avail_mult × exp_floor_mult
```

---

### Stage 6 — Sigmoid Normalization & Hard Exclusion Gate

The raw composite score is passed through a bounded sigmoid:

```
Normalized_Score = 1.0 / (1.0 + exp(-2.5 × (Final_Score − 0.55)))
```

This ensures all scores are in `[0.0, 1.0]` without arbitrary truncation. Ranking order is **exactly preserved** (sigmoid is monotonically increasing).

Before writing `submission.csv`, a final hard gate removes any remaining honeypots or fully-blacklisted candidates that survived preprocessing (edge cases). Candidates are then sorted descending by rounded score (4 decimal places), with ties broken alphabetically by `candidate_id`.

---

## 🏷️ Capability Groups

Keywords are organized into 6 semantic groups for consistent matching across all pipeline stages:

| Group                  | Example Terms                                                                |
| ---------------------- | ---------------------------------------------------------------------------- |
| Vector Retrieval       | Pinecone, Qdrant, FAISS, Milvus, Weaviate, HNSW, ANN Search, Dense Retrieval |
| Search Infrastructure  | Elasticsearch, OpenSearch, BM25, Hybrid Search, Solr, Inverted Index         |
| Recommendation Systems | Collaborative Filtering, Learning to Rank, Personalization, RRF              |
| Production ML          | Fine Tuning, Docker, Kubernetes, ONNX, Quantization, AWS, GCP                |
| LLM Engineering        | LLM, LoRA, QLoRA, LangChain, RAG, Prompt Engineering                         |
| Evaluation Metrics     | NDCG, MRR, MAP, BLEU, A/B Testing, Benchmarking                              |

---

## 📊 Local Evaluation Results

```
--- LOCAL METRICS REPORT ---
NDCG@10  : 0.9355 (Weight: 50%)
NDCG@50  : 0.7928 (Weight: 30%)
MAP      : 0.7255 (Weight: 15%)
P@10     : 1.0000 (Weight: 5%)
Composite: 0.8644
----------------------------
Total Pipeline: 155.47 seconds (limit: 300s, offline CPU execution)
```

### Consolidated Performance Breakdowns (ONNX Offline-loaded execution):

- **Preprocessing (Stage 1):** 26.44 seconds
- **Ranking (Stage 2):** 129.03 seconds
- **Local Evaluation (Stage 3):** 24.66 seconds

---

## 🔧 How to Run

### Step 1 — Install dependencies

Ensure you have all dependencies installed in your Python environment:
```bash
pip install -r requirements.txt
```

### Step 2 — Run the reproduction command

Organizers can run the end-to-end wrapper script to produce the submission CSV file in under 3 minutes (which automatically performs honeypot/blacklist preprocessing and re-ranking offline):
```bash
python rank.py --candidates challange_dataset/candidates.jsonl --out submission.csv
```

### Step 3 — Local Evaluation & Sanity Checks

To run sanity checks and calculate local evaluation metrics on the generated submission:
```bash
python -X utf8 src/local_eval.py --candidates challange_dataset/candidates.jsonl --submission submission.csv
```

### Step 4 — Extract Top 100 Profiles (Optional)

To extract the details and profiles of the top 100 candidates into JSON and CSV summaries for manual verification:
```bash
python src/extract_profiles.py
```

### Step 5 — Run Pipeline Master Script (Optional)

Alternatively, to run the whole preprocessing, ranking, and evaluation flow with consolidated times in one command:
```bash
python run_pipeline.py
```
