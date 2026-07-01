import os
import json
import argparse
import pickle
import ast
from tqdm import tqdm

from utils import (
    is_honeypot,
    is_blacklisted,
    extract_candidate_features,
    contains_any_keyword
)

# Skills that are relevant to platform assessment score matching
# Defined at module level to avoid re-allocation on every candidate iteration
_ASSESSMENT_RELEVANT_SKILLS = [
    "nlp", "machine learning", "deep learning", "python",
    "information retrieval", "vector", "search", "llm",
    "embeddings", "fine-tuning", "transformers"
]

def segment_candidate(cand):
    """
    Splits the candidate's profile into three logical text segments.
    """
    profile = cand.get("profile", {})
    headline = profile.get("headline") or ""
    summary = profile.get("summary") or ""
    hs = f"{headline} {summary}"

    career = cand.get("career_history", [])
    cr = ""
    pr_list = []
    if career:
        cr_job = career[0]
        cr = f"{cr_job.get('title') or ''} {cr_job.get('description') or ''}"
        for job in career[1:]:
            pr_list.append(f"{job.get('title') or ''} {job.get('description') or ''}")
    pr = " ".join(pr_list)
    return hs, cr, pr

def run_preprocessing(candidates_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading candidates...")
    valid_candidates = []

    if candidates_path.endswith('.json'):
        with open(candidates_path, 'r', encoding='utf-8') as f:
            candidates_list = json.load(f)
        for cand in tqdm(candidates_list, desc="Filtering candidates"):
            if is_honeypot(cand):
                continue
            if is_blacklisted(cand):
                continue
            valid_candidates.append(cand)
    else:
        with open(candidates_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='Filtering candidates'):
                if not line.strip():
                    continue
                try:
                    cand = json.loads(line)
                except json.decoder.JSONDecodeError:
                    cand = ast.literal_eval(line)
                if is_honeypot(cand):
                    continue
                if is_blacklisted(cand):
                    continue
                valid_candidates.append(cand)

    print(f"Filtered down to {len(valid_candidates)} candidates.")

    print("Prioritizing candidates via Feature Extractor (down-selecting to top 2000)...")
    prioritized_candidates = []

    for cand in tqdm(valid_candidates, desc="Scoring candidates"):
        features = extract_candidate_features(cand)

        # Calculate pre-filtering score from centralized features
        score = 0.0

        # 1. Experience Check
        years_exp = features["years_exp"]
        if 5.0 <= years_exp <= 10.0:
            score += 30.0
        elif 4.0 <= years_exp <= 13.0:
            score += 15.0
        else:
            score -= 20.0

        # 2. Capability Strengths (Layer 3)
        strengths_sum = sum(features["strengths"].values())
        score += strengths_sum * 6.0

        # 3. ATS Integrity Score
        score += features["ats_score"] * 30.0

        # 4. CV/Robotics Dominance Penalty
        if features["is_cv_dominated"]:
            score -= 40.0

        # 5. Core Platform Activity
        signals = cand.get("redrob_signals", {})
        response_rate = signals.get("recruiter_response_rate", 0.0)
        github_score = signals.get("github_activity_score", -1)

        if response_rate >= 0.50:   score += 15.0
        elif response_rate < 0.20:  score -= 25.0

        if github_score == -1:      score -= 25.0
        elif github_score > 50:     score += 15.0

        # Skill assessment scores — platform-verified proof of domain competence
        assessment_scores = signals.get("skill_assessment_scores", {})
        for skill_name, score_val in assessment_scores.items():
            if contains_any_keyword(skill_name.lower(), _ASSESSMENT_RELEVANT_SKILLS):
                if score_val >= 75:   score += 15.0; break
                elif score_val >= 65: score += 8.0;  break
                elif score_val < 40:  score -= 10.0; break

        # Active job seeking signal
        apps_30d = signals.get("applications_submitted_30d", 0)
        if apps_30d >= 3: score += 5.0

        # Geography check
        profile = cand.get("profile", {})
        loc = (profile.get("location") or "").lower()
        country = (profile.get("country") or "").lower()
        willing_relocate = signals.get("willing_to_relocate", False)
        is_local = contains_any_keyword(loc, ["pune", "noida", "mumbai", "delhi", "hyderabad", "bangalore"])

        if is_local:
            score += 15.0
        else:
            if "india" not in country:
                if willing_relocate: score -= 15.0
                else: score -= 40.0
            else:
                if not willing_relocate: score -= 20.0

        prioritized_candidates.append((cand, score))

    # Sort and slice to 1800
    prioritized_candidates.sort(key=lambda x: -x[1])
    selected_pairs = prioritized_candidates[:1800]
    valid_candidates = [pair[0] for pair in selected_pairs]

    # Generate segments for BM25
    headline_summaries = []
    current_roles = []
    past_roles = []

    print("Generating text segments...")
    for cand in tqdm(valid_candidates, desc="Segmenting text"):
        hs, cr, pr = segment_candidate(cand)
        headline_summaries.append(hs if hs else " ")
        current_roles.append(cr if cr else " ")
        past_roles.append(pr if pr else " ")

    cache_path = os.path.join(output_dir, "preprocessed_data.pkl")
    print(f"Saving preprocessed data to {cache_path}...")

    data = {
        "candidates": valid_candidates,
        "headline_summaries": headline_summaries,
        "current_roles": current_roles,
        "past_roles": past_roles
    }

    with open(cache_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    import time
    t_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="data/candidates.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/cache")
    args = parser.parse_args()

    run_preprocessing(args.candidates, args.output_dir)
    print(f"Total preprocessing time: {time.time() - t_start:.2f} seconds")

    run_preprocessing(args.candidates, args.output_dir)
