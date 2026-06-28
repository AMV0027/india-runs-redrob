import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils import is_honeypot, is_blacklisted, contains_any_keyword

def segment_candidate(candidate):
    """
    Extracts text segments from candidate profile:
    1. Headline + Summary
    2. Current Role (Title + Description)
    3. Past Roles (Combined Titles + Descriptions)
    """
    profile = candidate.get("profile", {})
    headline = profile.get("headline") or ""
    summary = profile.get("summary") or ""
    headline_summary = f"{headline} {summary}".strip()

    career_history = candidate.get("career_history", [])
    current_role_text = ""
    past_roles_text_list = []

    for job in career_history:
        title = job.get("title") or ""
        desc = job.get("description") or ""
        job_str = f"{title} {desc}".strip()
        
        if job.get("is_current"):
            current_role_text = job_str
        else:
            if job_str:
                past_roles_text_list.append(job_str)

    past_roles_text = " | ".join(past_roles_text_list)
    return headline_summary, current_role_text, past_roles_text

def run_preprocessing(candidates_path, output_dir):
    """
    Loads raw candidates, filters out honeypots and blacklists, segments texts,
    and pre-computes embeddings on CPU. Saves results to cache.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading candidates from {candidates_path}...")
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
            for line in tqdm(f, desc="Filtering candidates"):
                if not line.strip():
                    continue
                cand = json.loads(line)
                
                if is_honeypot(cand):
                    continue
                if is_blacklisted(cand):
                    continue

                valid_candidates.append(cand)
            
    print(f"Filtered {len(valid_candidates)} candidates.")

    # Prioritize the candidates using fast heuristics so we only embed the top 3,000
    print("Prioritizing candidates for embedding generation (down-selecting to top 3000)...")
    prioritized_candidates = []
    
    for cand in valid_candidates:
        profile = cand.get("profile", {})
        title = (profile.get("current_title") or "").lower()
        years_exp = profile.get("years_of_experience") or 0.0
        skills_objs = cand.get("skills", [])
        skill_names = [(s.get("name") or "").lower() for s in skills_objs if s.get("name")]
        career_history = cand.get("career_history", [])
        
        score = 0.0

        # ── 1. YEARS OF EXPERIENCE (Sweet Spot: 5.0 - 12.0 years) ──────────
        if 5.0 <= years_exp <= 10.0:
            score += 30.0  # Ideal range
        elif 4.0 <= years_exp <= 13.0:
            score += 15.0
        else:
            score -= 20.0  # Outlier penalty

        # ── 2. SENIORITY & TITLE EXPERIENCE MISMATCH ──────────────────────────
        from utils import is_title_experience_mismatch
        if is_title_experience_mismatch(title, years_exp):
            score -= 30.0  # Structural mismatch

        # ── 3. CURRENT TITLE ──────────────────────────────────────────────────
        ai_title_terms = ["ml", "machine learning", "ai", "artificial intelligence", "nlp", "search", "retrieval", "recommend", "applied scientist", "ai research", "ai specialist", "engineer", "scientist"]
        bad_title_terms = ["marketing", "hr", "sales", "recruiter",
                           "accountant", "talent", "finance", "content writer",
                           "graphic designer", "operations manager", "civil engineer",
                           "project manager", "product manager"]
        cv_title_terms = ["computer vision", "vision", "speech", "robotics", "ros", "embedded"]
        
        # Check title role seniority specifically
        is_senior = contains_any_keyword(title, ["senior", "lead", "staff", "principal", "founding"])
        is_junior = contains_any_keyword(title, ["junior", "jr", "associate", "intern", "trainee"])
        
        if contains_any_keyword(title, ai_title_terms) and not contains_any_keyword(title, cv_title_terms):
            if is_senior:
                score += 35.0  # Senior ML Engineer / Senior AI Scientist
            elif is_junior:
                score -= 15.0  # Junior ML Engineer (underpowered)
            else:
                score += 15.0  # Standard ML Engineer
        elif contains_any_keyword(title, bad_title_terms) or contains_any_keyword(title, cv_title_terms):
            score -= 40.0

        # ── 4. FORBIDDEN SKILLS CHECK (Gatekeeper) ───────────────────────────
        from utils import check_forbidden_skills
        if check_forbidden_skills(skill_names):
            score -= 60.0  # Massively penalize non-technical keywords stuffers

        # ── 5. VECTOR DB GATEKEEPER (Pinecone, FAISS, Milvus, Qdrant, Weaviate) 
        from utils import VECTOR_DB_KEYWORDS
        has_vector_db = any(contains_any_keyword(s, VECTOR_DB_KEYWORDS) for s in skill_names)
        
        # Scan job descriptions for vector db keywords
        has_vector_db_career = False
        for job in career_history:
            combined_desc = f"{job.get('title') or ''} {job.get('description') or ''}".lower()
            if contains_any_keyword(combined_desc, VECTOR_DB_KEYWORDS):
                has_vector_db_career = True
                break
                
        # If absolutely no vector DB signals exist, penalize heavily
        if not (has_vector_db or has_vector_db_career):
            score -= 50.0  # Vector DB is a hard prerequisite

        # ── 6. DOMAIN RELEVANCE VS CONFLICT PENALIZATION (CV/Speech/Robotics Check) ──
        from utils import CV_SPEECH_ROBOTICS_KEYWORDS, NLP_IR_SEARCH_KEYWORDS
        
        cv_count = 0
        nlp_count = 0
        
        for s in skill_names:
            if contains_any_keyword(s, CV_SPEECH_ROBOTICS_KEYWORDS):
                cv_count += 1
            if contains_any_keyword(s, NLP_IR_SEARCH_KEYWORDS):
                nlp_count += 1
                
        for job in career_history:
            combined_job_text = f"{job.get('title') or ''} {job.get('description') or ''}".lower()
            if contains_any_keyword(combined_job_text, CV_SPEECH_ROBOTICS_KEYWORDS):
                cv_count += 1
            if contains_any_keyword(combined_job_text, NLP_IR_SEARCH_KEYWORDS):
                nlp_count += 1

        # Heavy penalty if CV is dominant and they lack core NLP search signals
        if cv_count > 1 and cv_count > nlp_count:
            score -= 40.0
        elif cv_count > 0:
            score -= (cv_count * 4.0)

        # ── 7. DEEP CAREER HISTORY VALIDATION (Search/NLP Career Proof) ───────
        has_nlp_ir_career = False
        for job in career_history:
            combined_text = f"{job.get('title') or ''} {job.get('description') or ''}".lower()
            if contains_any_keyword(combined_text, NLP_IR_SEARCH_KEYWORDS):
                has_nlp_ir_career = True
                break

        if has_nlp_ir_career:
            score += 25.0
        else:
            score -= 20.0

        # ── 8. SKILLS LIST matches ───────────────────────────────────────────
        vector_db_kw   = ["pinecone", "qdrant", "milvus", "faiss", "weaviate",
                          "vector search", "vector database"]
        search_infra_kw = ["elasticsearch", "opensearch", "information retrieval",
                           "hybrid search", "retrieval", "rerank", "ndcg"]
        core_ml_kw     = ["machine learning", "applied ml", "nlp",
                          "sentence transformers", "hugging face transformers", "langchain",
                          "embeddings", "llm", "llms", "lora", "qlora", "peft", "fine tuning", "fine-tuning"]

        if any(contains_any_keyword(s, vector_db_kw) for s in skill_names):     score += 20.0
        if any(contains_any_keyword(s, search_infra_kw) for s in skill_names):  score += 15.0
        if any(contains_any_keyword(s, core_ml_kw) for s in skill_names):       score += 10.0

        # ── 9. SKILL ASSESSMENT SCORES (Platform-Verified) ─────────────────────
        assessment_kw = ["nlp", "machine learning", "deep learning", "python",
                         "information retrieval", "vector", "search", "llm",
                         "embeddings", "fine-tuning"]
        skill_assessment_scores = cand.get("redrob_signals", {}).get("skill_assessment_scores", {})
        for skill_name, score_val in skill_assessment_scores.items():
            if contains_any_keyword(skill_name.lower(), assessment_kw):
                if score_val >= 75:
                    score += 15.0
                    break
                elif score_val >= 50:
                    score += 7.0
                    break

        # ── 10. COMPANY INDUSTRY & SIZE ──────────────────────────────────────
        from utils import PRODUCT_INDUSTRIES, CONSULTING_INDUSTRIES
        product_job_count = 0
        consulting_job_count = 0
        for job in career_history:
            industry = (job.get("industry") or "").lower()
            if contains_any_keyword(industry, PRODUCT_INDUSTRIES):
                product_job_count += 1
            elif contains_any_keyword(industry, CONSULTING_INDUSTRIES):
                consulting_job_count += 1

        score += product_job_count * 6.0
        score -= consulting_job_count * 5.0

        for job in career_history:
            if job.get("company_size") == "10001+":
                score -= 3.0

        # ── 11. PLATFORM ENGAGEMENT & GEOGRAPHY ──────────────────────────────
        signals = cand.get("redrob_signals", {})
        response_rate = signals.get("recruiter_response_rate", 0.0)
        open_to_work  = signals.get("open_to_work_flag", True)
        apps_30d      = signals.get("applications_submitted_30d", 0)
        saved_30d     = signals.get("saved_by_recruiters_30d", 0)
        avg_resp_hrs  = signals.get("avg_response_time_hours", 999)
        github_score  = signals.get("github_activity_score", -1)

        # Recruiter response rate
        if response_rate >= 0.50:   score += 15.0
        elif response_rate < 0.20:  score -= 25.0

        # GitHub Presence (No GitHub is a major negative)
        if github_score == -1:
            score -= 25.0
        elif github_score > 50:
            score += 15.0

        # Open to work
        score += 8.0 if open_to_work else -5.0

        # Job application activity
        if apps_30d >= 3:    score += 8.0
        elif apps_30d >= 1:  score += 4.0

        # Saved by recruiters
        if saved_30d >= 5:   score += 8.0
        elif saved_30d >= 2: score += 4.0

        # Geography check (Tighter filter)
        loc = (profile.get("location") or "").lower()
        country = (profile.get("country") or "").lower()
        willing_relocate = signals.get("willing_to_relocate", False)
        
        is_local = contains_any_keyword(loc, ["pune", "noida", "mumbai", "delhi", "hyderabad", "bangalore"])
        if is_local:
            score += 15.0
        else:
            if "india" not in country:
                if willing_relocate:
                    score -= 15.0
                else:
                    score -= 40.0  # Distant overseas candidate not relocating
            else:
                if not willing_relocate:
                    score -= 20.0  # Indian candidate not local and not relocating

        # ── 12. EDUCATION ─────────────────────────────────────────────────────
        education = cand.get("education", [])
        for edu in education:
            if edu.get("tier") == "tier_1":
                score += 5.0
                break
            elif edu.get("tier") == "tier_2":
                score += 2.0
                break

        prioritized_candidates.append((cand, score))
        
    # Sort and slice
    prioritized_candidates.sort(key=lambda x: -x[1])
    selected_pairs = prioritized_candidates[:2500]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="../PS/candidates.jsonl")
    parser.add_argument("--output_dir", type=str, default="data_cache")
    args = parser.parse_args()
    
    run_preprocessing(args.candidates, args.output_dir)
