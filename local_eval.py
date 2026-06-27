import os
import sys
import time
import json
import csv
import re
import pickle
import numpy as np
from utils import is_honeypot, is_blacklisted, contains_any_keyword

def calculate_local_ndcg_and_map(submission_csv, candidates_jsonl):
    """
    Computes ranking metrics (NDCG@10, NDCG@50, MAP, P@10) using a local heuristic relevance model.
    Relevance levels:
      - 4: Perfect Match (production ML/Vector experience, local/active, no domain conflicts)
      - 3: Good Match (Applied ML/NLP experience with search focus, no domain conflicts)
      - 2: Adjacent Match (general software/data engineer with basic ML interest)
      - 1: Unrelated (mismatched titles, forbidden skills, or domain conflicts)
      - 0: Honeypot / Disqualified
    """
    candidates_dict = {}
    with open(candidates_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            candidates_dict[c["candidate_id"]] = c

    # 2. Industry-grade Relevance Grading System
    def get_silver_relevance(cand):
        if is_honeypot(cand) or is_blacklisted(cand):
            return 0
            
        profile = cand.get("profile", {})
        title = (profile.get("current_title") or "").lower()
        years_exp = profile.get("years_of_experience") or 0.0
        skills = [(s.get("name") or "").lower() for s in cand.get("skills", []) if s.get("name")]
        signals = cand.get("redrob_signals", {})
        career_history = cand.get("career_history", [])
        
        # Check active status & availability
        response_rate = signals.get("recruiter_response_rate") or 0.0
        last_active = signals.get("last_active_date")
        if not last_active:
            last_active = "2020-01-01"
        open_to_work = signals.get("open_to_work_flag")
        if open_to_work is None:
            open_to_work = True
        github_score = signals.get("github_activity_score")
        if github_score is None:
            github_score = -1
        
        # Calculate days since active using a static reference date (2026-06-26)
        ref_date = np.datetime64('2026-06-26')
        try:
            active_days = (ref_date - np.datetime64(last_active)).astype('timedelta64[D]').astype(int)
        except ValueError:
            active_days = 365
            
        # Passive candidate correction
        is_passive_gem = (not open_to_work) and (response_rate >= 0.70) and (active_days <= 90) and (github_score > 30)
        
        # Extended window to 90 days for open_to_work candidates (Bug 3 fix)
        is_highly_active = ((response_rate >= 0.50) and (active_days <= 90) and open_to_work) or is_passive_gem
        is_moderately_active = (response_rate >= 0.25) and (active_days <= 180)
        
        # Skill and Career domain checks
        from utils import CV_SPEECH_ROBOTICS_KEYWORDS, NLP_IR_SEARCH_KEYWORDS, FORBIDDEN_SKILLS, VECTOR_DB_KEYWORDS
        
        # Forbidden skills check
        from utils import check_forbidden_skills
        if check_forbidden_skills(skills):
            return 1
            
        # Title experience mismatch check
        from utils import is_title_experience_mismatch
        if is_title_experience_mismatch(title, years_exp):
            return 1
            
        has_vector_skills = any(contains_any_keyword(s, VECTOR_DB_KEYWORDS) for s in skills)
        
        # Deep career history validation
        has_nlp_ir_job = False
        cv_job_matches = 0
        nlp_job_matches = 0
        has_vector_db_career = False
        
        for job in career_history:
            j_title = (job.get("title") or "").lower()
            j_desc = (job.get("description") or "").lower()
            combined_job = f"{j_title} {j_desc}"
            
            if contains_any_keyword(combined_job, NLP_IR_SEARCH_KEYWORDS):
                has_nlp_ir_job = True
                nlp_job_matches += 1
            if contains_any_keyword(combined_job, CV_SPEECH_ROBOTICS_KEYWORDS):
                cv_job_matches += 1
            if contains_any_keyword(combined_job, VECTOR_DB_KEYWORDS):
                has_vector_db_career = True

        cv_skill_matches = sum(1 for s in skills if contains_any_keyword(s, CV_SPEECH_ROBOTICS_KEYWORDS))
        nlp_skill_matches = sum(1 for s in skills if contains_any_keyword(s, NLP_IR_SEARCH_KEYWORDS))
        
        total_cv_signals = cv_job_matches + cv_skill_matches
        total_nlp_signals = nlp_job_matches + nlp_skill_matches
        
        is_cv_dominated = (total_cv_signals > 1) and (total_cv_signals > total_nlp_signals)

        # Gatekeeper: Must have vector DB signal in skills OR career history
        has_vector_gatekeeper = has_vector_skills or has_vector_db_career

        # Check job titles in career history for applied ML background at product/tech level
        has_ml_title_history = False
        for job in career_history:
            job_title = (job.get("title") or "").lower()
            if contains_any_keyword(job_title, ["ml", "machine learning", "ai", "nlp", "search", "retrieval", "applied scientist"]):
                has_ml_title_history = True
                break

        # Disqualifier keywords (unrelated domains)
        is_unrelated_title = contains_any_keyword(title, ["marketing", "sales", "recruiter", "talent", "hr", "accountant", "finance"])

        # 🚀 VALIDATION CHECKS (Must match new rank.py constraints)
        # 1. Experience < 4.5 cannot be Tier 3 or Tier 4
        if years_exp < 4.5:
            return 2 if contains_any_keyword(title, ["engineer", "developer", "scientist"]) else 1
            
        # 2. Inactive > 90 days AND response rate < 0.40 cannot be Tier 3 or Tier 4
        if active_days > 90 and response_rate < 0.40:
            return 2 if contains_any_keyword(title, ["engineer", "developer", "scientist"]) else 1
            
        # 3. Candidate with >= 4 CV/Speech skills cannot be Tier 3 or Tier 4
        # BUT only if they also lack compensating vector/NLP signals (avoids penalising multi-modal engineers)
        CV_SPEECH_SKILLS = ["image classification", "opencv", "yolo", "object detection", "computer vision", "tts", "speech recognition", "asr", "diffusion models", "gans", "cnn"]
        cv_speech_match_count = sum(1 for s in skills if contains_any_keyword(s, CV_SPEECH_SKILLS))
        if cv_speech_match_count >= 4 and not has_vector_skills and nlp_skill_matches < 2:
            return 2 if contains_any_keyword(title, ["engineer", "developer", "scientist"]) else 1

        # Tier 4 (Perfect Match): Senior Applied ML/Vector Specialist with production focus, validated career history, and active engagement
        # Seniority can come from title keywords OR from specialist role + sufficient experience (>= 6 years)
        SPECIALIST_TITLES = ["recommendation systems", "search engineer", "nlp engineer",
                             "ml engineer", "machine learning engineer", "applied scientist",
                             "ai specialist", "ai research engineer"]
        has_seniority = (
            contains_any_keyword(title, ["senior", "lead", "staff", "principal", "founding"])
            or (years_exp >= 6.0 and contains_any_keyword(title, SPECIALIST_TITLES))
        )

        if (5.0 <= years_exp <= 12.0) and \
           contains_any_keyword(title, ["ml", "machine learning", "ai", "nlp", "search", "retrieval", "applied scientist", "engineer"]) and \
           has_seniority and \
           has_vector_gatekeeper and \
           has_nlp_ir_job and \
           is_highly_active and \
           not is_unrelated_title and \
           not is_cv_dominated:
            return 4

        # Tier 3 (Good Match): Applied ML/NLP Engineer with search experience and validated career history
        elif (4.0 <= years_exp <= 12.0) and \
             (contains_any_keyword(title, ["ml", "machine learning", "ai", "nlp", "search", "applied scientist", "engineer", "developer", "data"]) or has_ml_title_history) and \
             has_vector_gatekeeper and \
             has_nlp_ir_job and \
             is_moderately_active and \
             not is_unrelated_title and \
             not is_cv_dominated:
            return 3

        # Tier 2 (Adjacent Match): Software Engineer, Data Engineer, or Scientist
        elif contains_any_keyword(title, ["engineer", "developer", "scientist", "analyst", "programmer"]) and \
             not is_unrelated_title:
            return 2

        # Tier 1 (Unrelated): Mismatched roles or domain profiles
        return 1

    # Compute Ideal DCG (IDCG) globally over ALL candidates in candidates_dict.
    print("Computing global ideal ranking relevance scores across all candidates...")
    all_relevances = []
    for cid, cand in candidates_dict.items():
        all_relevances.append(get_silver_relevance(cand))
    all_relevances.sort(reverse=True)

    def dcg(rel, k):
        rel = np.asarray(rel[:k], dtype=float)
        if rel.size:
            return np.sum(rel / np.log2(np.arange(2, rel.size + 2)))
        return 0.0

    idcg_10 = dcg(all_relevances, 10)
    idcg_50 = dcg(all_relevances, 50)

    # 3. Read submission ranks and get relevances
    submitted_ids = []
    with open(submission_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            if row:
                submitted_ids.append(row[0])

    submitted_relevances = [get_silver_relevance(candidates_dict.get(cid)) for cid in submitted_ids]

    # Calculate metrics
    dcg_10 = dcg(submitted_relevances, 10)
    dcg_50 = dcg(submitted_relevances, 50)
    
    ndcg_10 = dcg_10 / idcg_10 if idcg_10 > 0 else 0.0
    ndcg_50 = dcg_50 / idcg_50 if idcg_50 > 0 else 0.0

    p_10 = sum(1 for r in submitted_relevances[:10] if r >= 3) / 10.0

    ap_sum = 0.0
    pos_count = 0
    for idx, r in enumerate(submitted_relevances):
        if r >= 3:
            pos_count += 1
            precision_at_i = pos_count / (idx + 1)
            ap_sum += precision_at_i
    map_score = ap_sum / pos_count if pos_count > 0 else 0.0

    composite = (0.50 * ndcg_10) + (0.30 * ndcg_50) + (0.15 * map_score) + (0.05 * p_10)

    print("\n--- LOCAL METRICS REPORT ---")
    print(f"NDCG@10  : {ndcg_10:.4f} (Weight: 50%)")
    print(f"NDCG@50  : {ndcg_50:.4f} (Weight: 30%)")
    print(f"MAP      : {map_score:.4f} (Weight: 15%)")
    print(f"P@10     : {p_10:.4f} (Weight: 5%)")
    print(f"Composite: {composite:.4f}")
    print("----------------------------")

def run_tests(submission_csv, candidates_jsonl):
    print("========== RUNNING LOCAL SANITY TESTS ==========")
    errors = []

    if not os.path.exists(submission_csv):
        errors.append(f"Submission file {submission_csv} does not exist.")
        return errors

    rows = []
    with open(submission_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            errors.append("CSV is empty.")
            return errors

        expected_header = ["candidate_id", "rank", "score", "reasoning"]
        if header != expected_header:
            errors.append(f"Expected header {expected_header}, found {header}")

        for row in reader:
            if row:
                rows.append(row)

    if len(rows) != 100:
        errors.append(f"Expected exactly 100 data rows, found {len(rows)}.")

    seen_ids = set()
    seen_ranks = set()
    prev_score = float('inf')

    print("Loading candidate records for reference checks...")
    candidates_dict = {}
    with open(candidates_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            candidates_dict[c["candidate_id"]] = c

    for idx, row in enumerate(rows, 2):
        if len(row) != 4:
            errors.append(f"Row {idx} does not have exactly 4 columns.")
            continue

        cid, rank_s, score_s, reasoning = row

        if not re.match(r"^CAND_[0-9]{7}$", cid):
            errors.append(f"Row {idx}: Invalid candidate_id format '{cid}'")
        elif cid not in candidates_dict:
            errors.append(f"Row {idx}: candidate_id '{cid}' not found in candidates.jsonl")
        elif cid in seen_ids:
            errors.append(f"Row {idx}: Duplicate candidate_id '{cid}' found.")
        else:
            seen_ids.add(cid)

        if cid in candidates_dict:
            cand_obj = candidates_dict[cid]
            if is_honeypot(cand_obj):
                errors.append(f"Row {idx}: Disqualified Honeypot profile ranked! ID={cid}")
            if is_blacklisted(cand_obj):
                errors.append(f"Row {idx}: Disqualified Blacklisted Consulting profile ranked! ID={cid}")

        try:
            rank = int(rank_s)
            if rank != (idx - 1):
                errors.append(f"Row {idx}: Rank mismatch. Expected {idx - 1}, found {rank}")
            seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {idx}: Rank is not an integer.")

        try:
            score = float(score_s)
            if score > prev_score:
                errors.append(f"Row {idx}: Score broke non-increasing rule. {score} > {prev_score}")
            prev_score = score
        except ValueError:
            errors.append(f"Row {idx}: Score is not a float.")

        if not reasoning or len(reasoning.strip()) < 10:
            errors.append(f"Row {idx}: Empty or too short reasoning.")

    missing_ranks = set(range(1, 101)) - seen_ranks
    if missing_ranks:
        errors.append(f"Missing ranks in output: {missing_ranks}")

    return errors

if __name__ == "__main__":
    errs = run_tests("submission.csv", "../PS/candidates.jsonl")
    if errs:
        print(f"\n[FAILED] Sanity validation FAILED with {len(errs)} error(s):")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All local sanity tests passed successfully!")
        calculate_local_ndcg_and_map("submission.csv", "../PS/candidates.jsonl")
        sys.exit(0)
