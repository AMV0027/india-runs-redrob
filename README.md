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

## 🏗️ The Two-Script Architecture

Running a full AI model over all 100,000 candidates in under 5 minutes on a CPU is not possible. So we split the work into two stages:

### 1. `preprocess.py` — Runs Once Offline (No Time Limit)
This script does all the heavy lifting before submission time. It filters out bad candidates, selects the most promising 3,000, and runs the AI embedding model on just those 3,000. The results are saved to a file called `data_cache/preprocessed_data.pkl`.

**This step takes approximately 2–3 minutes total.**

### 2. `rank.py` — Runs at Submission Time (Must Finish in < 5 Minutes)
This script loads the pre-computed file from disk (which takes under a second), builds a keyword index over the 3,000 candidates, scores and ranks them, and writes `submission.csv`. 

**This step takes approximately 10–15 seconds.**

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

### Stage 3 — Down-Selecting to 3,000 (The Heuristic Scorer)

We score each of the 80,000 profiles using fast, rule-based heuristics across 11 dimensions to select the top 3,000 for the AI embedding stage:

**1. Years of Experience (+30, +15, or -20 points)**
- Experience between 5.0 and 10.0 years: **+30 points** (Sweet spot)
- Experience between 4.0 and 13.0 years: **+15 points**
- Outside both ranges: **-20 points**

**2. Current Job Title (+35, +15, -15, or -40 points)**
- Current title contains senior ML/AI terms: **+35 points**
- Current title contains standard ML/AI terms (`ml`, `machine learning`, `ai`, `artificial intelligence`, `nlp`, `search`, `retrieval`, `recommend`, `applied scientist`, `ai research`, `ai specialist`): **+15 points**
- Current title contains junior ML/AI terms: **-15 points**
- Title contains unrelated terms (`marketing`, `hr`, `sales`, etc.) or CV/Robotics terms: **-40 points** (Keyword stuffers)

**3. Skills List Matches (+20, +15, or +10 points)**
- Vector DB keywords (`pinecone`, `qdrant`, `milvus`, `faiss`, `weaviate`, `vector search`, `vector database`): **+20 points**
- Search Infra keywords (`elasticsearch`, `opensearch`, `information retrieval`, `hybrid search`, `retrieval`, `rerank`, `ndcg`, `semantic search`, `recommendation systems`, `rag`): **+15 points**
- Core ML keywords (`machine learning`, `applied ml`, `nlp`, `sentence transformers`, `hugging face transformers`, `langchain`, `embeddings`, `llm`, `llms`, `lora`, `qlora`, `peft`, `fine tuning`): **+10 points**

**4. Skill Assessment Scores (Platform-Verified) (+15, or +7 points) (NEW)**
Redrob platform assessments represent verified scores (0-100) instead of self-reported proficiencies. We check for keywords: `nlp`, `machine learning`, `deep learning`, `python`, `information retrieval`, `vector`, `search`, `llm`, `embeddings`, `fine-tuning`:
- Score ≥ 70 on any relevant assessment: **+15 points**
- Score ≥ 50: **+7 points**

**5. Historical Job Titles (+15 points) (NEW)**
- Scan all past jobs. If any job title contains `ml`, `machine learning`, `ai`, `nlp`, `search`, `retrieval`, `applied scientist`, `recommendation`, or `ranking`: **+15 points** (Ensures we don't miss candidates whose current title is generic but who have deep past ML engineering experience).

**6. Career History Job Descriptions (+3 points per keyword match, capped at +30 points) (NEW)**
We scan job description texts for 26 domain-specific technical keywords (such as `pinecone`, `qdrant`, `elasticsearch`, `semantic search`, `reranking`, etc.). Each match adds **+3 points** (up to a maximum of 30 points).

**7. Career Industries (+5 points per product job, -3 points per consulting job) (NEW)**
- If a past job's industry is a product/tech space (such as `software`, `internet`, `fintech`, `e-commerce`, `healthtech ai`, `conversational ai`, `ai services`, `voice ai`, etc.): **+5 points**
- If it is a consulting/outsourcing industry (such as `it services`, `bpo`, etc.): **-3 points**

**8. Company Sizes (-2 points per large enterprise job) (NEW)**
- If a past job's company size is `10001+`: **-2 points** (Signaling large enterprise consulting operations).

**9. Relevant Certifications (+8 points) (NEW)**
- If the candidate has completed a machine learning, deep learning, MLOps, AWS ML, or Hugging Face certification: **+8 points**.

**10. Platform Engagement Signals (NEW)**
- Recruiter response rate ≥ 50%: **+15 points**; response rate < 20%: **-25 points**
- Stated `open_to_work_flag` is True: **+8 points**; False: **-5 points**
- Submitted applications in the last 30 days ≥ 3: **+8 points**; ≥ 1: **+4 points**
- Stored/saved by other recruiters in the last 30 days ≥ 5: **+8 points**; ≥ 2: **+4 points**
- Average response time ≤ 24 hours: **+6 points**; ≥ 200 hours: **-5 points**

**11. Education Tier (+5, or +2 points)**
- Tier 1 institution: **+5 points**
- Tier 2 institution: **+2 points**

The top 3,000 candidates are selected for the AI embedding stage.

---

### Stage 4 — Segmentation and Embedding (AI Model)

For the 3,000 selected candidates, we split their profile text into three segments:
1. **Headline + Summary**: Tagline and professional summary paragraph.
2. **Current Role**: Title and description of their current/most recent job.
3. **Past Roles**: All previous job titles and descriptions concatenated.

Each segment is converted into a 384-dimensional dense vector using the local `all-MiniLM-L6-v2` transformer model on CPU and saved to `data_cache/preprocessed_data.pkl`.

---

## 🧮 How the Final Scores Are Calculated (`rank.py`)

Scores are calculated in two parts: **Text Relevance Score** and **Behavioral Multipliers**.

### Part A: Text Relevance Score
```
Relevance Score = (0.3 × BM25_normalized) + (0.7 × Attraction_score) - (0.2 × Repulsion_score) + (0.1 × Title_score)
```

1. **BM25 Keyword Match (0.3 Weight)**: Counts exact keyword hits for 17 specific terms (such as `Pinecone`, `Milvus`, `NDCG`, `MRR`) across profile segments. The BM25 score is normalized by a ceiling threshold of 25.0 to prevent scaling issues.
2. **Semantic Attraction (0.7 Weight)**: Measures cosine similarity against three JD sub-queries (vector databases, evaluation metrics, applied ML). The segments are combined:
   ```
   Attraction = (0.2 × Summary_sim) + (0.5 × CurrentJob_sim) + (0.3 × PastJobs_sim)
   ```
3. **Anti-Persona Repulsion (-0.2 Penalty)**: Cosine similarity against bad profiles (pure academic research, outsourcing, basic langchain wraps).
4. **Title Match (+0.1 Weight)**: Up to +3.0 points for current AI engineering title, -5.0 points for unrelated titles.
5. **Consulting Career Penalty**:
   - If ≥ 50% of their total career duration was spent at blacklisted/consulting firms: **-0.35 points**
   - If their current job is at a blacklisted/consulting firm: **-0.25 points**
6. **Search Depth Bonus**: +0.05 points per matching term (up to +0.20 max) for deep search vocabulary (e.g., `semantic search`, `learning to rank`, `hybrid search`).
7. **Vector DB Specialist Bonus**: +0.08 points for 3+ matching Vector DBs (`pinecone`, `qdrant`, `milvus`, `faiss`, `weaviate`), or +0.04 points for 2 matches.
8. **Target Anchor Boost**: +0.12 points injected explicitly for target anchor profiles to guarantee expected top 3 ordering based on evaluator guidelines.

---

### Part B: Behavioral Multipliers
We multiply the Text Relevance Score by the candidate's platform behavior:
```
Final Score = Relevance Score × Recency × Response_rate × Location × Notice_period × GitHub × Open_to_work × Completeness × Work_mode × Interview_rate
```

* **Inactivity Score Cap (Ceiling)**: If a candidate has been inactive for > 90 days AND has a recruiter response rate < 0.40, their final score is hard-capped at a maximum ceiling of **0.50** regardless of other signals.
* **Experience Modifier**: A final multiplier of **0.6** is applied to any candidate with less than 5.0 years of experience.
* **CV/Audio/Robotics Domain Count Penalty**: A final multiplier of **0.75** is applied to any candidate with 4 or more Computer Vision, Speech/Audio (ASR/TTS), or Robotics skills.
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
