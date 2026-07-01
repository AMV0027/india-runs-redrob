import os
import csv
import pickle
import argparse
import time
import numpy as np
from rank_bm25 import BM25Okapi

from utils import (
    generate_reasoning,
    is_title_experience_mismatch,
    contains_any_keyword,
    CV_SPEECH_ROBOTICS_KEYWORDS,
    NLP_IR_SEARCH_KEYWORDS,
    COMPANY_BLACKLIST,
    CONSULTING_INDUSTRIES,
    check_forbidden_skills,
    VECTOR_DB_KEYWORDS,
    calculate_ats_score,
    extract_candidate_features,
    parse_job_description,
    CAPABILITY_GROUPS
)
from rerank import CrossEncoderReranker, normalize_ce_scores

# Sub-queries decomposing the target Senior AI Engineer JD
JD_SUB_QUERIES = [
    "production vector search deployment pinecone qdrant milvus faiss vector database index retrieval",
    "evaluation ranking systems ndcg mrr map offline online benchmarking",
    "applied ml nlp natural language processing sentence transformers embeddings fine tuning lora llm python"
]

# CV/Speech skills target list
CV_SPEECH_SKILLS = [
    "image classification", "opencv", "yolo", "object detection",
    "computer vision", "tts", "speech recognition", "asr",
    "diffusion models", "gans", "cnn"
]

# Tier A Search Depth vocabulary — extra bonus for deep IR/search expertise
TIER_A_SEARCH = [
    "information retrieval", "semantic search", "bm25",
    "hybrid search", "reranking", "ndcg", "mrr", "learning to rank",
    "search infrastructure", "indexing algorithms", "ranking systems",
    "dense retrieval", "sparse retrieval", "reciprocal rank fusion", "vector search"
]

# Skills matched against platform assessment scores to verify domain competence
_ASSESSMENT_RELEVANT_SKILLS = [
    "nlp", "machine learning", "deep learning", "python",
    "information retrieval", "vector", "search", "llm",
    "embeddings", "fine-tuning", "transformers"
]

# Keywords indicating a domain-relevant certification
_RELEVANT_CERT_KEYWORDS = [
    "machine learning", "deep learning", "nlp", "ai", "artificial intelligence",
    "aws", "gcp", "google cloud", "azure", "tensorflow", "pytorch",
    "hugging face", "transformers", "llm", "vector", "search", "information retrieval"
]

# Title relevance tier keyword lists — defined at module level to avoid re-allocation per candidate
_TITLE_AI_HIGH      = ["ai", "artificial intelligence", "nlp", "search", "retrieval", "rag"]
_TITLE_ML_MID       = ["ml", "machine learning", "recommend", "applied scientist", "ai research", "ai specialist"]
_TITLE_CV_TERMS     = ["computer vision", "vision", "speech", "robotics", "ros", "embedded"]
_TITLE_SENIOR_TERMS = ["senior", "lead", "staff", "principal", "founding"]
_TITLE_JUNIOR_TERMS = ["junior", "jr", "associate", "intern", "trainee"]
_TITLE_DISQUALIFIED = ["marketing", "hr", "sales", "recruiter", "talent", "accountant"]

def load_preprocessed_data(cache_dir):
    cache_path = os.path.join(cache_dir, "preprocessed_data.pkl")
    if not os.path.exists(cache_path):
        print(f"Error: Cache file not found at {cache_path}. Run preprocess.py first.")
        return None
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    return data

def run_ranking(candidates_path, output_csv, cache_dir):
    t_start = time.time()

    data = load_preprocessed_data(cache_dir)
    if not data:
        return

    candidates = data["candidates"]
    headline_summaries = data["headline_summaries"]
    current_roles = data["current_roles"]
    past_roles = data["past_roles"]

    n_candidates = len(candidates)
    print(f"Loaded {n_candidates} candidates from cache.")

    # ── STAGE 1: Dynamic JD Parsing ───────────────────────────────────────────
    jd_full_text = " ".join(JD_SUB_QUERIES)
    parsed_jd = parse_job_description(jd_full_text)
    print(f"Parsed JD Config: Must-Haves: {parsed_jd['must_haves']}, Priority Domain: {parsed_jd['priority_domain']}")

    # ── STAGE 2: BM25 Lexical Retrieval ──────────────────────────────────────
    print("Running BM25 indexing...")
    tokenized_hs = [text.lower().split() for text in headline_summaries]
    tokenized_cr = [text.lower().split() for text in current_roles]
    tokenized_pr = [text.lower().split() for text in past_roles]

    bm25_hs = BM25Okapi(tokenized_hs)
    bm25_cr = BM25Okapi(tokenized_cr)
    bm25_pr = BM25Okapi(tokenized_pr)

    # ── STAGE 3: Cross-Encoder Semantic Scoring (Headline Segment) ───────────
    reranker = CrossEncoderReranker()

    print(f"Scoring all {n_candidates} candidates with CE (Segment 1: Headline)...")
    pairs_hs = [(jd_full_text, text) for text in headline_summaries]
    raw_ce_hs = reranker._batch_predict(pairs_hs)
    ce_hs = [score for _, score in normalize_ce_scores(list(zip(candidates, raw_ce_hs)))]

    # ── STAGE 4: Fast Channel & Intermediate Gating ───────────────────────────
    print("Calculating intermediate scores for gating...")
    # Pre-calculate BM25 query keywords
    _bm25_query_set = set()
    for kws in CAPABILITY_GROUPS.values():
        for kw in kws:
            _bm25_query_set.update(kw.split())
    jd_keywords = list(_bm25_query_set)

    # Calculate Title, ATS, and BM25 Headline scores for gating
    bm25_hs_scores = [bm25_hs.get_batch_scores(jd_keywords, [i])[0] for i in range(n_candidates)]
    candidate_features = {}
    title_scores = []
    ats_scores = []

    for idx in range(n_candidates):
        cand = candidates[idx]
        profile = cand.get("profile", {})
        years_exp = profile.get("years_of_experience") or 0.0

        features = extract_candidate_features(cand, parsed_jd)
        candidate_features[idx] = features
        ats_scores.append(features["ats_score"])

        # Title Score — tiered by domain relevance and seniority level
        current_title = (profile.get("current_title") or "").lower()
        title_score = 0.0
        is_senior = contains_any_keyword(current_title, _TITLE_SENIOR_TERMS) or (years_exp >= 5.5)
        is_junior = contains_any_keyword(current_title, _TITLE_JUNIOR_TERMS)

        if contains_any_keyword(current_title, _TITLE_AI_HIGH) and not contains_any_keyword(current_title, _TITLE_CV_TERMS):
            if is_senior and not is_junior: title_score = 4.5
            elif is_junior: title_score = -2.0
            else: title_score = 2.0
        elif contains_any_keyword(current_title, _TITLE_ML_MID) and not contains_any_keyword(current_title, _TITLE_CV_TERMS):
            if is_senior and not is_junior: title_score = 3.0
            elif is_junior: title_score = -3.0
            else: title_score = 1.0
        elif contains_any_keyword(current_title, _TITLE_DISQUALIFIED) or contains_any_keyword(current_title, _TITLE_CV_TERMS):
            title_score = -5.0

        if is_title_experience_mismatch(current_title, years_exp):
            title_score -= 3.0
        title_scores.append(title_score)

    # Compute fast-channel intermediate rank to select top 1000 candidates
    # We rank based on CE_HS (Headline) + Title + ATS + BM25_HS
    fast_ranks = []
    # Normalize intermediate features for gating
    min_bm25_hs = min(bm25_hs_scores)
    max_bm25_hs = max(bm25_hs_scores)
    norm_bm25_hs = [(s - min_bm25_hs) / (max_bm25_hs - min_bm25_hs) if max_bm25_hs > min_bm25_hs else 1.0 for s in bm25_hs_scores]

    for idx in range(n_candidates):
        # We weigh Headline CE heavily for early semantic filter
        score_gate = (0.50 * ce_hs[idx]) + (0.20 * norm_bm25_hs[idx]) + (0.20 * (title_scores[idx] + 5.0)/9.5) + (0.10 * ats_scores[idx])
        # Force disqualify honeypots/fully-blacklisted early
        if candidate_features[idx]["is_honeypot"] or candidate_features[idx]["is_blacklisted"]:
            score_gate = -999.0
        fast_ranks.append((idx, score_gate))

    fast_ranks.sort(key=lambda x: -x[1])
    gated_indices = set(item[0] for item in fast_ranks[:700])
    # Preserve original order for stable mapping back to normalized CE score arrays
    gated_ordered = [idx for idx in range(n_candidates) if idx in gated_indices]
    gated_candidates = [candidates[idx] for idx in gated_ordered]
    print(f"Gatekeeper: Selected top 700 candidates for deep career scoring from {n_candidates}.")

    # ── STAGE 5: Deep Semantic Scoring (Current & Past Roles, Gated Pool Only) ─
    print("Scoring gated candidates on Current Role segment...")
    gated_cr_texts = [current_roles[idx] for idx in gated_ordered]
    pairs_cr = [(jd_full_text, text) for text in gated_cr_texts]
    raw_ce_cr = reranker._batch_predict(pairs_cr)
    normalized_ce_cr = normalize_ce_scores(list(zip(gated_candidates, raw_ce_cr)))

    # Map gated scores back to the full 2000-length array (non-gated candidates stay 0.0)
    ce_cr = [0.0] * n_candidates
    for i, idx in enumerate(gated_ordered):
        ce_cr[idx] = normalized_ce_cr[i][1]

    print("Scoring gated candidates on Past Roles segment...")
    gated_pr_texts = [past_roles[idx] for idx in gated_ordered]
    pairs_pr = [(jd_full_text, text) for text in gated_pr_texts]
    raw_ce_pr = reranker._batch_predict(pairs_pr)
    normalized_ce_pr = normalize_ce_scores(list(zip(gated_candidates, raw_ce_pr)))

    ce_pr = [0.0] * n_candidates
    for i, idx in enumerate(gated_ordered):
        ce_pr[idx] = normalized_ce_pr[i][1]

    # ── STAGE 6: RRF Score Collection ─────────────────────────────────────────
    print("Calculating final RRF scores...")
    # Pre-batch BM25 scores for gated candidates on current/past role segments
    bm25_cr_gated = {idx: bm25_cr.get_batch_scores(jd_keywords, [idx])[0] for idx in gated_ordered}
    bm25_pr_gated = {idx: bm25_pr.get_batch_scores(jd_keywords, [idx])[0] for idx in gated_ordered}
    raw_ce_list = []
    raw_bm25_list = []
    raw_title_list = []
    raw_ats_list = []

    for idx in range(n_candidates):
        features = candidate_features[idx]

        # 1. CE Semantic Score (weighted segments)
        ce_val = (0.2 * ce_hs[idx]) + (0.5 * ce_cr[idx]) + (0.3 * ce_pr[idx])
        for req in parsed_jd["must_haves"]:
            if features["strengths"].get(req, 0) == 0:
                ce_val *= 0.70
        raw_ce_list.append((idx, ce_val))

        # 2. BM25 Score
        score_bm25_hs = bm25_hs_scores[idx]
        score_bm25_cr = bm25_cr_gated.get(idx, 0.0)
        score_bm25_pr = bm25_pr_gated.get(idx, 0.0)
        score_bm25 = (0.2 * score_bm25_hs) + (0.5 * score_bm25_cr) + (0.3 * score_bm25_pr)
        raw_bm25_list.append((idx, score_bm25))

        # 3. Title Score
        raw_title_list.append((idx, title_scores[idx]))

        # 4. ATS Score
        raw_ats_list.append((idx, ats_scores[idx]))

    # Rank all 4 signal channels
    raw_ce_list.sort(key=lambda x: -x[1])
    ce_ranks = {item[0]: rank for rank, item in enumerate(raw_ce_list, 1)}

    raw_bm25_list.sort(key=lambda x: -x[1])
    bm25_ranks = {item[0]: rank for rank, item in enumerate(raw_bm25_list, 1)}

    raw_title_list.sort(key=lambda x: -x[1])
    title_ranks = {item[0]: rank for rank, item in enumerate(raw_title_list, 1)}

    raw_ats_list.sort(key=lambda x: -x[1])
    ats_ranks = {item[0]: rank for rank, item in enumerate(raw_ats_list, 1)}

    # Compute RRF score
    rrf_raw_scores = []
    for idx in range(n_candidates):
        ce_r = ce_ranks[idx]
        bm25_r = bm25_ranks[idx]
        title_r = title_ranks[idx]
        ats_r = ats_ranks[idx]
        rrf = (0.60 / (60.0 + ce_r)) + (0.25 / (60.0 + bm25_r)) + (0.10 / (60.0 + title_r)) + (0.05 / (60.0 + ats_r))
        rrf_raw_scores.append(rrf)

    # Normalize RRF scores
    min_rrf = min(rrf_raw_scores)
    max_rrf = max(rrf_raw_scores)
    if max_rrf > min_rrf:
        normalized_rrfs = [(r - min_rrf) / (max_rrf - min_rrf) for r in rrf_raw_scores]
    else:
        normalized_rrfs = [1.0] * n_candidates

    bm25_score_dict = {item[0]: item[1] for item in raw_bm25_list}

    # ── STAGE 7: Final Score Adjustments & Multipliers ────────────────────────
    print("Calculating final adjusted scores with soft penalties...")
    final_scores = []

    for idx in range(n_candidates):
        cand = candidates[idx]
        features = candidate_features[idx]
        profile = cand.get("profile", {})
        years_exp = features["years_exp"]
        skills_objs = cand.get("skills", [])
        skill_names = [(s.get("name") or "").lower() for s in skills_objs if s.get("name")]
        career_history = cand.get("career_history", [])

        # Base RRF relevance
        relevance_score = normalized_rrfs[idx]

        # Soft multiplicative domain penalties
        if check_forbidden_skills(skill_names):
            relevance_score *= 0.50

        if features["is_cv_dominated"]:
            relevance_score *= 0.75

        # Career consulting duration ratio penalty
        consulting_months = 0
        total_months = 0
        for job in career_history:
            comp = (job.get("company") or "").lower()
            industry = (job.get("industry") or "").lower()
            dur = job.get("duration_months", 0)
            total_months += dur
            if contains_any_keyword(comp, COMPANY_BLACKLIST) or contains_any_keyword(industry, CONSULTING_INDUSTRIES):
                consulting_months += dur
        if total_months > 0 and (consulting_months / total_months) >= 0.50:
            relevance_score *= 0.70

        # Search depth vocabulary bonus
        search_bonus = 0.0
        for term in TIER_A_SEARCH:
            if any(contains_any_keyword(s, [term]) for s in skill_names):
                search_bonus += 0.05
        search_bonus = min(search_bonus, 0.20)
        relevance_score += search_bonus

        # Dynamic Behavioral Multipliers
        signals = cand.get("redrob_signals", {})

        last_active = signals.get("last_active_date") or "2020-01-01"
        try:
            active_dt = np.datetime64(last_active)
            active_days = (np.datetime64('2026-06-26') - active_dt).astype('timedelta64[D]').astype(int)
        except ValueError:
            active_days = 365

        response_rate = signals.get("recruiter_response_rate", 0.0)

        # Recency
        recency_mult = 1.0
        if active_days <= 45: recency_mult = 1.15
        elif active_days > 180: recency_mult = 0.4
        elif active_days > 120: recency_mult = 0.65

        # Response Rate
        response_mult = 1.0
        if response_rate >= 0.70: response_mult = 1.10
        elif response_rate < 0.20: response_mult = 0.30
        elif response_rate < 0.40: response_mult = 0.80

        # Response time complement — how quickly the candidate replies
        avg_resp_hrs = signals.get("avg_response_time_hours", 999)
        resp_time_mult = 1.0
        if avg_resp_hrs <= 12:   resp_time_mult = 1.08
        elif avg_resp_hrs <= 48: resp_time_mult = 1.04
        elif avg_resp_hrs > 200: resp_time_mult = 0.90

        # Platform skill assessment — verified proof of domain competence
        # Score >=75 on a relevant skill: verified expert; <40: credibility red flag
        assessment_scores = signals.get("skill_assessment_scores", {})
        assessment_mult = 1.0
        best_score = -1
        worst_claimed_score = None
        for skill_name, score_val in assessment_scores.items():
            if contains_any_keyword(skill_name.lower(), _ASSESSMENT_RELEVANT_SKILLS):
                if score_val > best_score:
                    best_score = score_val
                if worst_claimed_score is None or score_val < worst_claimed_score:
                    worst_claimed_score = score_val
        if best_score >= 75:    assessment_mult = 1.12
        elif best_score >= 65:  assessment_mult = 1.06
        elif worst_claimed_score is not None and worst_claimed_score < 40:
            assessment_mult = 0.88

        # Recruiter saves — wisdom-of-crowds proxy for profile quality
        saved_30d = signals.get("saved_by_recruiters_30d", 0)
        saved_mult = 1.0
        if saved_30d >= 8:   saved_mult = 1.10
        elif saved_30d >= 4: saved_mult = 1.05
        elif saved_30d >= 2: saved_mult = 1.02

        # Application activity — signals active job seeking intent
        apps_30d = signals.get("applications_submitted_30d", 0)
        apps_mult = 1.0
        if apps_30d >= 5:   apps_mult = 1.05
        elif apps_30d >= 2: apps_mult = 1.02
        elif apps_30d == 0: apps_mult = 0.97

        # Interview completion rate — ghost candidate risk indicator
        interview_rate = signals.get("interview_completion_rate", 1.0)
        if interview_rate == -1: interview_rate = 1.0  # no data → neutral
        interview_mult = 1.0
        if interview_rate >= 0.85:  interview_mult = 1.06
        elif interview_rate < 0.50: interview_mult = 0.80
        elif interview_rate < 0.65: interview_mult = 0.90

        # Location & relocation fit
        loc = (profile.get("location") or "").lower()
        country = (profile.get("country") or "").lower()
        willing_relocate = signals.get("willing_to_relocate", False)

        location_mult = 1.0
        is_local = contains_any_keyword(loc, ["pune", "noida"])
        is_ncr = contains_any_keyword(loc, ["delhi", "gurgaon", "ghaziabad", "faridabad"])
        if is_local: location_mult = 1.20
        elif is_ncr: location_mult = 1.18
        elif willing_relocate: location_mult = 1.05
        else:
            if "india" not in country: location_mult = 0.35
            else: location_mult = 0.95

        notice_days = signals.get("notice_period_days") if signals.get("notice_period_days") is not None else 90
        notice_mult = 1.0
        if notice_days == 0: notice_mult = 1.18
        elif notice_days <= 15: notice_mult = 1.15
        elif notice_days <= 30: notice_mult = 1.10
        elif notice_days <= 90: notice_mult = 0.85
        else: notice_mult = 0.70

        github_score = signals.get("github_activity_score", -1)
        github_mult = 1.0
        if github_score > 80: github_mult = 1.25
        elif github_score > 50: github_mult = 1.15
        elif github_score == -1: github_mult = 0.75

        open_to_work = signals.get("open_to_work_flag", True)
        otw_mult = 1.0 if open_to_work else 0.70

        # Certifications — verified domain-relevant certificates boost credibility
        certifications = cand.get("certifications", [])
        cert_mult = 1.0
        for cert in certifications:
            cert_name = (cert.get("name") or "").lower()
            if contains_any_keyword(cert_name, _RELEVANT_CERT_KEYWORDS):
                cert_mult = 1.08
                break

        # Work mode alignment — role is hybrid (Pune/Noida), remote-only is a mismatch
        work_mode = (signals.get("preferred_work_mode") or "").lower()
        work_mode_mult = 1.0
        if work_mode == "onsite":
            work_mode_mult = 1.04
        elif work_mode == "remote" and not willing_relocate:
            work_mode_mult = 0.88
        elif work_mode == "remote":
            work_mode_mult = 0.95

        avail_mult = (
            recency_mult * response_mult * resp_time_mult *
            location_mult * notice_mult * github_mult * otw_mult *
            assessment_mult * saved_mult * apps_mult * interview_mult *
            cert_mult * work_mode_mult
        )
        final_score = relevance_score * avail_mult

        if years_exp < 5.0:
            final_score *= 0.65

        # Sigmoid Normalization
        normalized_score = 1.0 / (1.0 + np.exp(-2.5 * (final_score - 0.55)))
        final_scores.append((cand, normalized_score, bm25_score_dict[idx], features))

    # Sort by final score
    final_scores.sort(key=lambda x: (-round(x[1], 4), x[0].get("candidate_id")))

    # ── STAGE 8: Selection & Reasoning ───────────────────────────────────────
    print("Selecting top candidates and generating reasons...")
    top_candidates = []
    excluded_count = 0

    for cand, score, bm_score, feat in final_scores:
        if len(top_candidates) >= 100:
            break

        # Hard exclusion gate (Honeypot or completely blacklisted)
        if feat["is_honeypot"] or feat["is_blacklisted"]:
            excluded_count += 1
            continue

        top_candidates.append((cand, score, feat))

    print(f"Hard exclusion gate: removed {excluded_count} disqualifying candidates from final ranking.")

    # Write to submission.csv
    print(f"Writing results to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, (cand, score, feat) in enumerate(top_candidates, 1):
            cand_id = cand.get("candidate_id")
            reason = generate_reasoning(cand, rank, feat)
            writer.writerow([cand_id, rank, round(score, 4), reason])

    elapsed = time.time() - t_start
    print(f"Total execution time: {elapsed:.2f} seconds")
    print(f"Candidates processed: {n_candidates}")
    print(f"Time per candidate: {(elapsed / n_candidates) * 1000:.3f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="data/candidates.jsonl")
    parser.add_argument("--out", type=str, default="output/submission.csv")
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    args = parser.parse_args()

    run_ranking(args.candidates, args.out, args.cache_dir)
