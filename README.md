# Redrob AI — Senior AI Engineer Candidate Ranking System

This document explains **exactly** how the ranking system works, what every decision means, why each number was chosen, and how the final output is produced. No fillers. No abbreviations.

---

## 🎯 The Goal

We have **100,000 synthetic candidate profiles** stored in `candidates.jsonl`. Each profile describes a person — their job history, skills, education, and behavioral signals on the Redrob platform. Our task is to find and output the **top 100 candidates** who are the best fit for this specific role:

> **Senior AI Engineer (Founding Team)** at Redrob AI  
> Location: Pune / Noida, Hybrid  
> Experience: 5–9 years (flexible)

The submission must be a CSV file with exactly 100 rows, each with a `candidate_id`, a `rank` (1 to 100), a `score` (floating-point, must be non-increasing), and a `reasoning` sentence.

There is one strict hardware constraint: **the ranking script must complete in under 5 minutes on a CPU, with no internet access.**

---

## 🏗️ The Four-Stage Pipeline Architecture

Running a full AI model over all 100,000 candidates in under 5 minutes on a CPU is not possible. So we split the work into four stages across two scripts:

### 1. `preprocess.py` — Runs Once Offline (No Time Limit)
This script does all the heavy lifting before submission time. It filters out bad candidates, selects the most promising 3,000, and runs the AI Bi-Encoder embedding model on just those 3,000. The results are saved to `data_cache/preprocessed_data.pkl`.

**This step takes approximately 2–3 minutes total.**

### 2. `rank.py` — Runs at Submission Time (Must Finish in < 5 Minutes)
This script orchestrates three further stages:

- **Stage 2 (Coarse Retrieval)**: Loads pre-computed embeddings, runs BM25 + cosine similarity + rule-based bonuses, and shortlists the **top 400 candidates** (~15s).
- **Stage 3 (Cross-Encoder Re-Ranking)**: Feeds each of the 400 shortlisted candidates jointly with the Job Description through a `cross-encoder/ms-marco-MiniLM-L-6-v2` transformer model for precise relevance scoring (~40–80s).
- **Stage 4 (Final Blend)**: Blends the cross-encoder score (60%) with the rule-based behavioral signals (40%) — response rate, notice period, location, inactivity — and writes the final `submission.csv`.

**This step takes approximately 60–100 seconds total.**

### 3. `rerank.py` — Cross-Encoder Module (imported by `rank.py`)
Contains the `CrossEncoderReranker` class and candidate text builder. Handles batched cross-encoder inference and min-max normalization of raw logit scores to the `[0, 1]` range.

---

## 📋 Step-by-Step: What `preprocess.py` Does

### Stage 1 — Honeypot Detection (`utils.py → is_honeypot()`)

The dataset contains profiles with **logically impossible timelines**. These are called "honeypots" — they are planted traps. Any submission that ranks more than 10 honeypots in the top 100 is automatically **disqualified** by the hackathon organizers.

We detect honeypots by running five mathematical checks on every profile:

**Check 1 — Total job months vs. stated experience**  
We add up the `duration_months` field across every job in the candidate's career history. We then compare this sum to their stated `years_of_experience` field (converted to months), allowing a buffer of 12 extra months to account for overlapping jobs or part-time transitions.

- *Example*: If a candidate says they have 5 years (60 months) of experience, but the sum of all their individual job durations adds up to 85 months — that is more than 60 + 12 = 72, so they are flagged as a honeypot.

**Check 2 — Career span date vs. stated experience**  
We find the earliest `start_date` across all their jobs and the latest `end_date`. We calculate the real calendar span between those two dates and compare it to their stated years of experience, allowing a buffer of 2 years for education gaps or breaks.

- *Example*: If a candidate's first job started in 2010 and their last job ended in 2023, that is a 13-year span. But if they state only 5 years of experience, they are flagged — because no amount of gaps can explain 8 missing years.

**Check 3 — Individual skill duration vs. stated experience**  
Each skill in the profile has a `duration_months` field saying how many months the candidate used that skill. No single skill should be used for longer than the candidate's total experience, plus a 6-month buffer.

- *Example*: If a candidate states 4 years (48 months) of experience, but one skill says they've used Python for 72 months, that is impossible. They are flagged.

**Check 4 — Expert skills with no experience**  
If a candidate claims 8 or more skills at the `expert` or `advanced` proficiency level, but the total combined duration across all skills is less than 12 months, that is an impossible claim. They are flagged.

**Check 5 — Education Graduation Year vs. years_of_experience**  
If a candidate claims 15 years of experience but graduated with their latest degree in 2022, that is a timeline contradiction. Using 2026 as the baseline context year, the maximum possible experience is calculated as `(2026 - latest_graduation_year) + 1` (allowing a 1-year buffer for final-year internships). If the stated `years_of_experience` exceeds this threshold by more than 2 years, they are flagged.

---

### Stage 2 — Consulting Blacklist (`utils.py → is_blacklisted()`)

The job description explicitly says: *"Only consulting firm experience (TCS, Infosys, Wipro, etc.) is a disqualifier."*

We execute a robust two-layer check:

**Layer 1 — Name-based check**  
We maintain a list of 18 consulting/IT-outsourcing firms that are disqualifiers:
```
TCS (Tata Consultancy Services), Infosys, Wipro, Accenture, Cognizant,
Capgemini, HCL, Tech Mahindra, L&T Infotech, Mindtree, Mphasis,
NTT Data, UST Global, EY, PwC, Deloitte, KPMG
```
We check every job in a candidate's career history. If the company name matches any of these firm names (using strict word boundary matching to avoid false positives), it is flagged.

**Layer 2 — Industry-based check**  
To catch smaller consulting firms not in our named list, we check the `industry` field of each job. If the industry belongs to:
```
IT Services, IT Consulting, Consulting, Outsourcing, Managed Services, Staffing, BPO, KPO
```
it is flagged.

**The rule is**: If **every single job** in a candidate's career was at a blacklisted name-matched firm OR belonged to a consulting industry, they are discarded. A candidate who spent 4 years at TCS but then moved to Swiggy for 3 years is **not** discarded (though they will be score-penalized during ranking). Profiles with **no career history at all** are also discarded here.

After both checks, roughly ~80,000 clean profiles remain.

---

### Stage 3 — Down-Selecting to 2,000 (The Heuristic Scorer)

We score each of the 80,000 profiles using fast, rule-based heuristics across 11 dimensions using the centralized Feature Extractor to select the top 2,000. No AI embedding happens here.
The top 2,000 candidates are written to a cache for the re-ranker.

---

## 🧮 How the Final Scores Are Calculated (`rank.py`)

Scores are calculated in two parts: **Text Relevance Score** and **Behavioral Multipliers**.

### Part A: Text Relevance Score
Relevance Score is computed using **Reciprocal Rank Fusion (RRF)** to combine the ranking lists of our 4 core models (using standard constant $k=60$):
```
RRF_Score = (0.60 / (60 + R_CE)) + (0.25 / (60 + R_BM25)) + (0.10 / (60 + R_Title)) + (0.05 / (60 + R_ATS))
```
Where:
* $R_{CE}$ is the candidate's rank in the Cross-Encoder semantic score.
* $R_{BM25}$ is the rank in the segmented BM25 Okapi keyword matches.
* $R_{Title}$ is the rank in the tiered title relevance scorer.
* $R_{ATS}$ is the rank in the ATS resume-integrity parser.

The resulting combined RRF score is min-max normalized to `[0.0, 1.0]` before applying behavioral multipliers.

1. **Semantic Cross-Encoder (0.60 Weight)**: Evaluated jointly by `cross-encoder/ms-marco-MiniLM-L-6-v2` across profile segments. If a candidate has `0` strength in a must-have capability (e.g. `Vector Retrieval`), their CE score is scaled by a **`* 0.70`** soft confidence adjustment.
2. **BM25 Keyword Match (0.25 Weight)**: Counts exact keyword hits across profile segments, normalized.
3. **Title Match (+0.10 Weight)**: Prioritizes AI/NLP/Search roles over general ML. Tier 1 (AI, NLP, Search, RAG) gets up to **+4.5 points** (Senior) or +2.0 points (Standard). Tier 2 (ML, Machine Learning, Applied Scientist) gets **+3.0 points** (Senior) or +1.0 points (Standard). Unrelated titles get -5.0 points.
4. **ATS Resume-Integrity Score (+0.05 Weight)**: Evaluates structural profile features:
   - *Skill Coverage (40%)*: Checks profile skills against JD prerequisite tech stacks.
   - *Career Stability (30%)*: Rewards longer average employment tenure (>3 years) and penalizes job-hopping (<1.2 years).
   - *Anomalous Gap Penalties (15%)*: Penalizes silent career gaps of >12 months between consecutive jobs.
   - *Progression Promotion (15%)*: Awards growth bonuses for candidates who show promotions from junior to senior roles over their history.
5. **Soft Multiplicative Penalties**:
   - If ≥ 50% of their total career duration was spent at blacklisted/consulting firms: **`* 0.70`**
   - CV/Robotics Dominance: **`* 0.75`**
   - Forbidden Skills matched: **`* 0.50`**
6. **Search Depth Bonus**: Up to +0.20 points max for deep search vocabulary (e.g., `semantic search`, `learning to rank`, `hybrid search`).

---

### Part B: Behavioral Multipliers
We multiply the Text Relevance Score by the candidate's platform behavior:
```
Final Score = Relevance Score × Recency × Response_rate × Location × Notice_period × GitHub × Open_to_work × Completeness × Work_mode × Interview_rate
```

* **Inactivity Score Cap (Ceiling)**: If a candidate has been inactive for > 90 days AND has a recruiter response rate < 0.40, their final score is hard-capped at a maximum ceiling of **0.50** regardless of other signals.
* **Experience Modifier**: A final multiplier of **0.6** is applied to any candidate with less than 5.0 years of experience.
* **CV/Speech Domain Count Penalty**: A final multiplier of **0.75** is applied to any candidate with 4 or more CV/Speech skills (e.g. OpenCV, YOLO, ASR).
* **Recency**: Active in last 45 days = **× 1.15**; Inactive > 180 days = **× 0.3** (relevance score also hard-capped at 0.2).
* **Response Rate**: Reply to recruiters ≥ 70% = **× 1.1**; Reply < 20% = **× 0.2** (relevance score also hard-capped at 0.2).
* **Location**: Located in Pune/Noida = **× 1.15**; Located in NCR region (Delhi, Gurgaon, Ghaziabad, Faridabad) = **× 1.12** (NCR semi-local boost); Not local but willing to relocate = **× 1.05**; Not local and not willing to relocate (within India) = **× 0.95**; Overseas and not willing to relocate = **× 0.3**.
* **Notice Period**: notice_period_days == 0 = **× 1.18**; ≤ 15 days = **× 1.15**; ≤ 30 days = **× 1.10**; ≤ 60 days = **× 1.0**; ≤ 90 days = **× 0.85**; > 90 days = **× 0.70**.
* **GitHub**: Score > 50 = **× 1.15**; Score 11 to 50 = **× 1.03**; Score = -1 (no GitHub linked) = **× 0.70** (heavy penalty).
* **Open to work**: `open_to_work_flag` is True = **× 1.0**; False = **× 0.65** (relaxed to **× 0.90** for passive gems who are highly responsive and active on GitHub).
* **Profile Completeness**: Score < 50% = **× 0.85**.
* **Work Mode**: Stated preferred mode is `remote` and `willing_to_relocate` is False = **× 0.80**.
* **Interview Rate**: Stated interview completion rate < 60% = **× 0.85**.

### Part C: Bounded Sigmoid Normalization
To prevent wild score fluctuations and ensure that no score exceeds 1.0 mathematically, the raw composite score is passed through a bounded logistic sigmoid function:
```
Normalized Score = 1.0 / (1.0 + exp(-2.5 * (Final_Score - 0.55)))
```
This forces all raw scores smoothly into a strict `[0.0, 1.0]` range without arbitrary truncation (`max(1, score)`). Because the sigmoid is a monotonically increasing function, the ranking order of the candidates is **exactly preserved**.

Candidates are sorted in descending order of their rounded normalized score (4 decimal places). Ties are broken alphabetically by `candidate_id` ascending.

---

## 🏷️ How Tiers are Evaluated in the Code (`local_eval.py`)

Instead of statically categorizing candidates based on their final ranking position, the system **evaluates the candidate's actual profile features** to assign them to a performance tier. The grading logic enforces the same boundaries:

1. **Experience Check**: Candidates with less than 4.5 years of experience are restricted to Tier 2 or Tier 1.
2. **Inactivity Check**: Candidates inactive > 90 days with response rate < 0.40 are restricted to Tier 2 or Tier 1.
3. **CV/Audio/Robotics Domain Check**: Candidates with 4 or more Computer Vision, Speech/Audio (ASR/TTS), or Robotics skills are restricted to Tier 2 or Tier 1.

* **Tier 0 (Disqualified)**: Honeypot (Timeline checks fail) or spent entire career at consulting/IT services firms (or consulting industries).
* **Tier 4 (Perfect Match)**: 5–12 years of experience; holds a current ML/AI/NLP/Search engineering title; possesses vector database skills (Pinecone, Qdrant, etc.); has validated career history (NLP/search mentioned in past job descriptions); is highly active/responsive (or is a strong responsive passive contributor); and is not CV/Audio-dominated.
* **Tier 3 (Good Match)**: 4–12 years of experience; engineering title or ML job history; possesses vector, search, or core ML skills; has validated career history; is moderately active; and is not CV/Audio-dominated.
* **Tier 2 (Adjacent Match)**: General Software, Data, or systems engineers.
* **Tier 1 (Unrelated)**: Unrelated domains or roles.

---

## 🔧 How to Run

**Step 1 — Install dependencies**:
```bash
pip install -r requirements.txt
```

**Step 2 — Run preprocessing offline**:
```bash
python preprocess.py --candidates ../PS/candidates.jsonl --output_dir data_cache
```

**Step 3 — Run ranking**:
```bash
python rank.py --candidates ../PS/candidates.jsonl --out submission.csv --cache_dir data_cache
```

**Step 4 — Validate and evaluate**:
```bash
python local_eval.py
```
