# Implementation Plan: Candidate Discovery & Ranking System

## Goal Description

Build an intelligent candidate ranking system for the **"Senior AI Engineer — Founding Team"** role. The system must process 100,000 candidate profiles and output the top 100 ranked candidates with a 1-2 sentence reasoning for each.

The architecture is a **Two-Stage Pipeline** using **TF-IDF Sparse Retrieval + BM25 Hybrid + Cross-Encoder Re-ranking**. This replaces the original FAISS neural embedding design, cutting offline precomputation from ~56 minutes down to **~1-2 minutes** while preserving retrieval quality and strictly adhering to the 5-minute CPU runtime constraint.

---

## Architecture Overview

### Stage 1: Pre-Computation — `precompute.py` (Offline, No Time Limit)

Processes `candidates.jsonl` in **streaming chunks** with **parallel feature extraction** (`ProcessPoolExecutor`) to produce four artifacts:

| Artifact | Description |
|---|---|
| `features.parquet` | Structured candidate features (PyArrow, Snappy-compressed) |
| `tfidf_skills.pkl` | TF-IDF vectorizer + L2-normalised sparse matrix for skills text |
| `tfidf_exp.pkl` | TF-IDF vectorizer + L2-normalised sparse matrix for experience text |
| `bm25.pkl` | BM25Okapi index + combined corpus + candidate ID list |

#### Feature Extraction per Candidate (`process_candidate`)

Each candidate record yields the following structured features written to `features.parquet`:

| Feature | Type | Logic |
|---|---|---|
| `years_of_experience` | `float32` | From `profile.years_of_experience` |
| `honeypot_flag` | `bool` | `skill.duration_months == 0` AND `proficiency == "expert"` |
| `ghost_flag` | `bool` | `last_active_date > 180 days` ago AND `recruiter_response_rate < 0.10` |
| `is_consulting_only` | `bool` | All career history at firms in a hardcoded consulting blocklist |
| `title_chaser_flag` | `bool` | Average tenure across `career_history` is `< 18 months` (only if `> 2` roles) |
| `is_pure_research` | `bool` | `>= 3` research keyword hits AND `< 3` production keyword hits in career descriptions |
| `is_langchain_wrapper` | `bool` | `>= 2` LangChain/API-wrapper keyword hits AND `< 2` production keyword hits |
| `notice_period_days` | `int32` | From `redrob_signals` |
| `willing_to_relocate` | `bool` | From `redrob_signals` |
| `profile_completeness_score` | `float32` | From `redrob_signals` |
| `open_to_work` | `bool` | From `redrob_signals.open_to_work_flag` |
| `recruiter_response_rate` | `float32` | From `redrob_signals` |
| `github_activity_score` | `float32` | From `redrob_signals` |

#### NLP Classification (Regex-Based, Not Zero-Shot)

> **Note (Design Change):** The original plan called for a zero-shot DeBERTa classifier (`facebook/bart-large-mnli`). The actual implementation uses **fast regex pattern matching** instead — this avoids the heavy model dependency and keeps precompute time under 2 minutes.

Three compiled regex patterns classify experience descriptions:
- **`RESEARCH_KEYWORDS`**: Detects PhD, arXiv, publications, conferences, research labs, IEEE, NeurIPS, ICLR, etc.
- **`LANGCHAIN_KEYWORDS`**: Detects LangChain, LlamaIndex, GPT-wrapper, no-code, prompt-engineering-only, etc.
- **`PRODUCTION_AI_KEYWORDS`**: Detects deployed, serving, inference, vector search, fine-tuning, RLHF, A/B test, scale, etc.

The ratio of hits determines the `is_pure_research` and `is_langchain_wrapper` boolean flags.

#### TF-IDF Sparse Indexes (Replaces FAISS Neural Embeddings)

Two TF-IDF indexes are built — one for skills text, one for experience text:
- `TfidfVectorizer(max_features=60_000, sublinear_tf=True, min_df=2, ngram_range=(1,2), dtype=float32)`
- Matrix is **L2-normalised** so dot-product equals cosine similarity at query time (matching the old FAISS `IndexFlatIP` behaviour)

#### BM25 Index

`BM25Okapi` is built over the `combined_tokens` list (skills + experience + profile, casefolded). The `combined_texts` string list is also stored for the cross-encoder at runtime.

#### Key Optimisations in `precompute.py`

| Tag | Optimisation |
|---|---|
| `[#1]` | Parallel feature extraction via `ProcessPoolExecutor` |
| `[#7]` | Streaming chunks (default 10,000) to bound peak RAM |
| `[#8]` | PyArrow batched `ParquetWriter` — avoids one huge in-memory DataFrame |
| `[#9]` | `orjson` for 2-5x faster JSON parsing (falls back to stdlib) |
| `[#11]` | Direct token-list construction, no intermediate strings |
| `[#15]` | `casefold()` for BM25 tokenisation |
| `[NEW]` | TF-IDF sparse vectors replace neural embeddings (56 min -> ~30 s) |

**Runtime target**: ~1-2 minutes on any CPU.

---

### Stage 2: Runtime Ranking — `rank.py` (<=5 min, CPU-only, no network)

#### Pipeline Steps

**Step 1 — Load Precomputed Artifacts**
- `features.parquet` -> `pd.DataFrame`
- `tfidf_skills.pkl` -> vectorizer + L2-normed sparse matrix
- `tfidf_exp.pkl` -> vectorizer + L2-normed sparse matrix
- `bm25.pkl` -> BM25Okapi, `combined_texts`, `cid_list`

**Step 2 — Load Raw Candidates**
- Streams `candidates.jsonl` into a `cand_lookup` dict `{candidate_id -> raw_dict}` for reasoning generation.

**Step 3 — Hybrid TF-IDF + BM25 Retrieval**

JD is split into two structured query strings:

```python
JD_SKILLS = "Python embeddings vector search retrieval ranking NLP LLMs fine-tuning \
              sentence-transformers FAISS Pinecone Weaviate Qdrant hybrid search \
              NDCG MRR MAP evaluation A/B testing recommendation system"

JD_EXP    = "5 to 9 years applied ML AI engineer production deployment real users \
              end-to-end ranking search recommendation pipeline product company \
              shipped vector search embedding retrieval at scale"
```

- Both queries are L2-normalised and dot-producted against the respective TF-IDF matrices -> `skills_sims`, `exp_sims`
- `np.argpartition` retrieves top-K (default 2,000) per channel — O(n) fast partial sort
- BM25 scores computed over the full corpus with JD tokens, top-K selected
- **Union** of all three index sets forms the candidate pool (avoids missing exact-keyword matches)

**Step 4 — Additive Base Scoring**

```python
# TF-IDF cosine sims shifted from [0,1] to [0.5, 1] to match FAISS (x+1)/2 rescaling
skills_match = 0.5 + tfidf_cosine_skills / 2
exp_match    = 0.5 + tfidf_cosine_exp    / 2

# BM25 normalized to [0, 1] by pool max
bm25_score = bm25_raw / bm25_pool_max

# Structured experience score: peaks at 1.0 for 5-9 yrs, linear decay outside band
struct_exp = structured_exp_score(years_of_experience)

# Behavioral composite
behavioral = (profile_completeness_score / 100.0 * 0.4) +
             (recruiter_response_rate * 0.4) +
             (open_to_work * 0.2)

# Final additive base score — weights sum to 1.0
base_score = (0.35 * exp_match) +
             (0.25 * skills_match) +
             (0.15 * bm25_score) +
             (0.15 * struct_exp) +
             (0.10 * behavioral)
```

**Step 5 — Hard Multipliers (JD Logic Traps)**

| Flag | Multiplier Cap | Rationale |
|---|---|---|
| `honeypot_flag` | `0.0` (eliminated) | Mathematically impossible profile |
| `ghost_flag` | `0.0` (eliminated) | Inactive + non-responsive — effectively unavailable |
| `is_pure_research` | `clip(upper=0.1)` | Pure academia, no production deployment |
| `is_langchain_wrapper` | `clip(upper=0.2)` | AI experience is API-calls only |
| `is_consulting_only` | `clip(upper=0.5)` | No product company experience |
| `title_chaser_flag` | `clip(upper=0.7)` | Short average tenures signal instability |

`score = base_score x multiplier`

> **Design Note:** This replaces the original soft additive penalties (`-0.05 * consulting_penalty`). Hard multiplicative caps make disqualification deterministic — a bad candidate cannot "score around" the penalty.

**Step 6 — Cross-Encoder Re-ranking**

- Top 300 candidates by `score` are sent to `cross-encoder/ms-marco-MiniLM-L-6-v2` on CPU
- CE scores are min-max normalised to `[0, 1]`
- Final score blends CE semantics with structured base:
  ```python
  final_score = (0.55 * ce_score + 0.45 * base_score) * multiplier
  ```
- Gracefully falls back to `base_score` if cross-encoder fails or is unavailable

**Step 7 — Tie-Breaking & Output**

- `notice_period_days <= 30` -> `+0.001` boost (JD states they can buy out 30-day notice)
- Final sort: `[final_score DESC, notice_period_days ASC, candidate_id ASC]`
- Top 100 selected; `rank` column assigned 1-100
- `reasoning` generated per candidate via `generate_reasoning()` using raw profile data
- Written to `submission.csv` with columns: `candidate_id`, `rank`, `score`, `reasoning`

#### Reasoning Generation (`generate_reasoning`)

Builds a factual 1-2 sentence string from the candidate's raw profile:
- **Base sentence**: `current_title`, years of experience, top AI-relevant skills (up to 4), recruiter response rate
- **Nuance sentence**: Flags honeypot, ghost, consulting-only, pure research, LangChain wrapper, title chaser, notice period, open-to-work
- Capped at 280 characters to stay readable

---

## Proposed Changes

### [MODIFIED] `development/precompute.py`
Offline preprocessing script. Replaces original FAISS + zero-shot classifier design with TF-IDF sparse indexes + regex classifiers. Produces `features.parquet`, `tfidf_skills.pkl`, `tfidf_exp.pkl`, `bm25.pkl`.

### [MODIFIED] `development/rank.py`
Runtime ranking script. Replaces original FAISS retrieval with TF-IDF cosine similarity via `linear_kernel`. Implements hard multipliers instead of soft additive penalties. Retains cross-encoder reranking. Writes `submission.csv`.

### [NEW] `submission_metadata.yaml`
Filled-out submission metadata template (team details, reproduce commands).

---

## CLI Usage

```bash
# Offline precomputation (~1-2 min on any CPU)
python development/precompute.py \
  --input challange_dataset/candidates.jsonl \
  --out_dir development/

# Runtime ranking (must complete < 5 min on CPU)
python development/rank.py \
  --candidates challange_dataset/candidates.jsonl \
  --out development/submission.csv \
  --data_dir development/

# Validate submission format
python challange_dataset/validate_submission.py development/submission.csv

# Benchmark helper (measures wall time)
python benchmark.py python development/precompute.py \
  --input challange_dataset/candidates.jsonl \
  --out_dir development
```

---

## Verification Plan

### Automated Tests
1. Run `precompute.py` on `sample_candidates.json` — must finish in `< 5s`
2. Run `rank.py` on sample artifacts — must finish in `< 2 min`
3. Assert `submission.csv` has **exactly 100 rows** with strictly non-increasing scores
4. Run `validate_submission.py submission.csv` (organizer-provided) — must pass

### Quality Checks
- Verify honeypot candidates in sample data are eliminated (multiplier = 0.0)
- Verify ghost candidates (inactive + low response rate) are eliminated
- Verify `reasoning` text is factual and references actual profile data (no hallucination)
- Verify top 100 candidates have `years_of_experience` predominantly in the 5-9 year band

---

## Why It Wins

### Speed (5-Minute CPU Constraint)

| Stage | Method | Estimated Time |
|---|---|---|
| TF-IDF retrieval (2,000 x 3 channels) | `np.argpartition` + `linear_kernel` | ~1-2 s |
| BM25 scoring | `rank_bm25` | ~1-2 s |
| Additive scoring on pool | Vectorised pandas | ~0.5 s |
| Cross-encoder reranking (top 300) | `ms-marco-MiniLM-L-6-v2` CPU | ~10-15 s |
| Output generation | Pandas + CSV write | ~0.5 s |
| **Total** | | **~15-20 s** |

### Logic Traps (JD Disqualifiers)

| Trap | Handled By |
|---|---|
| Honeypot (impossible skill stats) | `get_honeypot_flag()` -> multiplier `0.0` |
| Ghost (inactive + non-responsive) | `get_ghost_flag()` -> multiplier `0.0` |
| Pure Research (no production) | `classify_research_vs_production()` -> `clip(0.1)` |
| LangChain Wrapper (API-only AI) | `classify_research_vs_production()` -> `clip(0.2)` |
| Consulting-Only (no product co.) | `is_consulting_only()` -> `clip(0.5)` |
| Title Chaser (short tenures) | `get_title_chaser_flag()` -> `clip(0.7)` |
