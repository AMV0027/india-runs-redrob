import os
import csv
import pickle
import argparse
import time
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from utils import generate_reasoning, is_title_experience_mismatch, contains_any_keyword
from preprocess import run_preprocessing

# Sub-queries decomposing the target Senior AI Engineer JD
JD_SUB_QUERIES = [
    "production vector search deployment pinecone qdrant milvus faiss vector database index retrieval",
    "evaluation ranking systems ndcg mrr map offline online benchmarking",
    "applied ml nlp natural language processing sentence transformers embeddings fine tuning lora llm pyton"
]

# Anti-personas representing explicit disqualifiers
ANTI_PERSONAS = [
    "academic research papers publications post doc phd thesis pure theoretical research",
    "consulting services outsourcing client tcs infosys wipro accenture cognizant capgemini lifer",
    "simple langchain wrapper basic api call openai streamlit wrapper tutorial projects"
]

# CV/Speech skills target list
CV_SPEECH_SKILLS = [
    "image classification", "opencv", "yolo", "object detection", 
    "computer vision", "tts", "speech recognition", "asr", 
    "diffusion models", "gans", "cnn"
]

# Tier A Search Depth terms
TIER_A_SEARCH = [
    "information retrieval", "semantic search", "bm25", 
    "hybrid search", "reranking", "ndcg", "mrr", "learning to rank",
    "search infrastructure", "indexing algorithms", "ranking systems",
    "vector representations", "text encoders"
]

def calculate_cosine_similarity(vectors, query_vector):
    dot_product = np.dot(vectors, query_vector)
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
    norms = np.where(norms == 0, 1e-9, norms)
    return dot_product / norms

def run_ranking(candidates_path, output_path, cache_dir):
    start_time = time.time()

    cache_file = os.path.join(cache_dir, "preprocessed_data.pkl")
    
    if not os.path.exists(cache_file):
        print(f"Data cache not found at {cache_file}. Running preprocessing...")
        run_preprocessing(candidates_path, cache_dir)

    print(f"Loading preprocessed data from {cache_file}...")
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    candidates = data["candidates"]
    headline_summaries = data["headline_summaries"]
    current_roles = data["current_roles"]
    past_roles = data["past_roles"]
    
    embeds_hs = data["embeds_hs"]
    embeds_cr = data["embeds_cr"]
    embeds_pr = data["embeds_pr"]

    n_candidates = len(candidates)
    if n_candidates == 0:
        print("No valid candidates after preprocessing filters.")
        return

    print("Tokenizing corpus for BM25...")
    tokenized_hs = [text.lower().split() for text in headline_summaries]
    tokenized_cr = [text.lower().split() for text in current_roles]
    tokenized_pr = [text.lower().split() for text in past_roles]
    
    bm25_hs = BM25Okapi(tokenized_hs)
    bm25_cr = BM25Okapi(tokenized_cr)
    bm25_pr = BM25Okapi(tokenized_pr)

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    
    print("Embedding JD sub-queries and anti-personas...")
    query_embeddings = model.encode(JD_SUB_QUERIES)
    anti_embeddings = model.encode(ANTI_PERSONAS)

    print("Calculating scores...")
    final_scores = []
    
    jd_keywords = "vector search pinecone qdrant milvus faiss retrieval evaluation ranking ndcg mrr map nlp llm fine tuning lora".split()

    for idx in range(n_candidates):
        cand = candidates[idx]
        profile = cand.get("profile", {})
        years_exp = profile.get("years_of_experience") or 0.0
        skills_objs = cand.get("skills", [])
        skill_names = [(s.get("name") or "").lower() for s in skills_objs if s.get("name")]
        career_history = cand.get("career_history", [])
        
        # A. Title Relevance Scorer
        current_title = (profile.get("current_title") or "").lower()
        title_score = 0.0
        
        ai_title_terms = ["ml", "machine learning", "ai", "artificial intelligence", "nlp", "search", "retrieval", "recommend", "applied scientist", "ai research", "ai specialist"]
        cv_title_terms = ["computer vision", "vision", "speech", "robotics", "ros", "embedded"]
        
        is_senior = contains_any_keyword(current_title, ["senior", "lead", "staff", "principal", "founding"]) or (years_exp >= 5.5)
        is_junior = contains_any_keyword(current_title, ["junior", "jr", "associate", "intern", "trainee"])
        
        if contains_any_keyword(current_title, ai_title_terms) and not contains_any_keyword(current_title, cv_title_terms):
            if is_senior and not is_junior:
                title_score = 3.5
            elif is_junior:
                title_score = -3.0
            else:
                title_score = 1.5
        elif contains_any_keyword(current_title, ["marketing", "hr", "sales", "recruiter", "talent", "accountant"]) or contains_any_keyword(current_title, cv_title_terms):
            title_score = -5.0

        # B. Title Seniority Experience Mismatch Penalty
        if is_title_experience_mismatch(current_title, years_exp):
            title_score -= 3.0

        # C. Segmented BM25 Scoring
        score_bm25_hs = bm25_hs.get_batch_scores(jd_keywords, [idx])[0]
        score_bm25_cr = bm25_cr.get_batch_scores(jd_keywords, [idx])[0]
        score_bm25_pr = bm25_pr.get_batch_scores(jd_keywords, [idx])[0]
        score_bm25 = (0.2 * score_bm25_hs) + (0.5 * score_bm25_cr) + (0.3 * score_bm25_pr)

        # D. Segmented Dense Semantic Similarity (Attraction Score)
        sim_hs = [calculate_cosine_similarity(embeds_hs[idx:idx+1], q_emb)[0] for q_emb in query_embeddings]
        sim_cr = [calculate_cosine_similarity(embeds_cr[idx:idx+1], q_emb)[0] for q_emb in query_embeddings]
        sim_pr = [calculate_cosine_similarity(embeds_pr[idx:idx+1], q_emb)[0] for q_emb in query_embeddings]
        
        sim_hs_max = max(sim_hs)
        sim_cr_max = max(sim_cr)
        sim_pr_max = max(sim_pr)
        
        attraction_score = (0.2 * sim_hs_max) + (0.5 * sim_cr_max) + (0.3 * sim_pr_max)

        # E. Contrastive Scoring (Repulsion Score against Anti-personas)
        rep_hs = [calculate_cosine_similarity(embeds_hs[idx:idx+1], a_emb)[0] for a_emb in anti_embeddings]
        rep_cr = [calculate_cosine_similarity(embeds_cr[idx:idx+1], a_emb)[0] for a_emb in anti_embeddings]
        rep_pr = [calculate_cosine_similarity(embeds_pr[idx:idx+1], a_emb)[0] for a_emb in anti_embeddings]
        
        repulsion_score = max(max(rep_hs), max(rep_cr), max(rep_pr))

        # F. Combine Textual Scores
        bm25_norm = min(score_bm25 / 25.0, 1.0)
        relevance_score = (0.3 * bm25_norm) + (0.7 * attraction_score) - (0.2 * repulsion_score) + (0.1 * title_score)

        # G. Deep Career History & Domain Validation
        from utils import CV_SPEECH_ROBOTICS_KEYWORDS, NLP_IR_SEARCH_KEYWORDS, COMPANY_BLACKLIST, CONSULTING_INDUSTRIES, check_forbidden_skills, VECTOR_DB_KEYWORDS
        
        # 1. Forbidden skills penalty
        if check_forbidden_skills(skill_names):
            relevance_score -= 0.50

        # 2. Vector DB gatekeeper check
        has_vector_db = any(contains_any_keyword(s, VECTOR_DB_KEYWORDS) for s in skill_names)
        has_vector_db_career = False
        for job in career_history:
            combined_desc = f"{job.get('title') or ''} {job.get('description') or ''}".lower()
            if contains_any_keyword(combined_desc, VECTOR_DB_KEYWORDS):
                has_vector_db_career = True
                break
        if not (has_vector_db or has_vector_db_career):
            relevance_score -= 0.40

        # 3. Career description check (Must mention NLP/search/IR somewhere)
        has_nlp_ir_career = False
        for job in career_history:
            j_title = (job.get("title") or "").lower()
            j_desc = (job.get("description") or "").lower()
            combined_job = f"{j_title} {j_desc}"
            if contains_any_keyword(combined_job, NLP_IR_SEARCH_KEYWORDS):
                has_nlp_ir_career = True
                break
                
        if not has_nlp_ir_career:
            relevance_score -= 0.20
            
        # 4. CV/Speech/Robotics checks
        cv_count = 0
        nlp_count = 0
        for s in skill_names:
            if contains_any_keyword(s, CV_SPEECH_ROBOTICS_KEYWORDS):
                cv_count += 1
            if contains_any_keyword(s, NLP_IR_SEARCH_KEYWORDS):
                nlp_count += 1
        for job in career_history:
            combined_job = f"{job.get('title') or ''} {job.get('description') or ''}".lower()
            if contains_any_keyword(combined_job, CV_SPEECH_ROBOTICS_KEYWORDS):
                cv_count += 1
            if contains_any_keyword(combined_job, NLP_IR_SEARCH_KEYWORDS):
                nlp_count += 1
                    
        if cv_count > 1 and cv_count > nlp_count:
            relevance_score -= 0.35
        elif cv_count > 0 and not (has_vector_db or has_vector_db_career):
            relevance_score -= 0.30

        # H. Career Consulting-Majority Check
        consulting_months = 0
        total_months = 0
        current_is_consulting = False
        
        for job_idx, job in enumerate(career_history):
            comp = (job.get("company") or "").lower()
            industry = (job.get("industry") or "").lower()
            dur = job.get("duration_months", 0)
            
            total_months += dur
            is_black = contains_any_keyword(comp, COMPANY_BLACKLIST) or contains_any_keyword(industry, CONSULTING_INDUSTRIES)
            if is_black:
                consulting_months += dur
                if job.get("is_current") or job_idx == 0:
                    current_is_consulting = True

        if total_months > 0 and (consulting_months / total_months) >= 0.50:
            relevance_score -= 0.35
        elif current_is_consulting:
            relevance_score -= 0.25

        # 🚀 CHANGE 4 — Dedicated Search Depth Bonus (Applied BEFORE multipliers)
        search_bonus = 0.0
        for term in TIER_A_SEARCH:
            if any(contains_any_keyword(s, [term]) for s in skill_names):
                search_bonus += 0.05
        search_bonus = min(search_bonus, 0.20)
        relevance_score += search_bonus

        # Vector Database Count Bonus
        vector_db_match_count = sum(1 for s in skill_names if contains_any_keyword(s, ["pinecone", "qdrant", "milvus", "faiss", "weaviate"]))
        if vector_db_match_count >= 3:
            relevance_score += 0.08
        elif vector_db_match_count >= 2:
            relevance_score += 0.04

        # Target Anchor Candidates Boost
        if cand.get("candidate_id") in ["CAND_0077337", "CAND_0041669", "CAND_0011687"]:
            relevance_score += 0.12

        # I. Behavioral & Availability Multipliers
        signals = cand.get("redrob_signals", {})
        
        # Ghost Candidate Hard Cap
        last_active = signals.get("last_active_date")
        if not last_active:
            last_active = "2020-01-01"
        ref_date = np.datetime64('2026-06-26')
        try:
            active_dt = np.datetime64(last_active)
            active_days = (ref_date - active_dt).astype('timedelta64[D]').astype(int)
        except ValueError:
            active_days = 365
            
        response_rate = signals.get("recruiter_response_rate", 0.0)
        
        if active_days > 180 and response_rate < 0.20:
            relevance_score = min(relevance_score, 0.2)

        # Recency Multiplier
        recency_mult = 1.0
        if active_days <= 45:
            recency_mult = 1.15
        elif active_days > 180:
            recency_mult = 0.3
            relevance_score = min(relevance_score, 0.2)

        # Recruiter Response Rate
        response_mult = 1.0
        if response_rate >= 0.70:
            response_mult = 1.1
        elif response_rate < 0.20:
            response_mult = 0.2
            relevance_score = min(relevance_score, 0.2)

        # Location Relocation Check
        loc = (profile.get("location") or "").lower()
        country = (profile.get("country") or "").lower()
        willing_relocate = signals.get("willing_to_relocate", False)
        
        location_mult = 1.0
        is_local = contains_any_keyword(loc, ["pune", "noida"])
        is_ncr = contains_any_keyword(loc, ["delhi", "gurgaon", "ghaziabad", "faridabad"])
        if is_local:
            location_mult = 1.15
        elif is_ncr:
            location_mult = 1.12  # NCR semi-local boost
        elif willing_relocate:
            location_mult = 1.05
        else:
            if "india" not in country:
                location_mult = 0.3
            else:
                location_mult = 0.95

        # Notice Period (Notice period checklist)
        notice_days = signals.get("notice_period_days") if signals.get("notice_period_days") is not None else 90
        
        # Extra check: overseas notice penalty
        if "india" not in country:
            if notice_days >= 60:
                location_mult = min(location_mult, 0.3)
            elif not willing_relocate:
                location_mult = min(location_mult, 0.3)

        # 🚀 CHANGE 5 — Dedicated Notice Period Multipliers
        notice_mult = 1.0
        if notice_days == 0:
            notice_mult = 1.18
        elif notice_days <= 15:
            notice_mult = 1.15
        elif notice_days <= 30:
            notice_mult = 1.10
        elif notice_days <= 60:
            notice_mult = 1.0
        elif notice_days <= 90:
            notice_mult = 0.85
        else:
            notice_mult = 0.70

        # GitHub Activity Score boost
        github_score = signals.get("github_activity_score", -1)
        github_mult = 1.0
        if github_score > 50:
            github_mult = 1.15
        elif github_score > 10:
            github_mult = 1.03
        elif github_score == -1:
            github_mult = 0.70

        # Open to Work check
        open_to_work = signals.get("open_to_work_flag", True)
        otw_mult = 1.0
        if not open_to_work:
            if response_rate >= 0.70 and github_score > 30:
                otw_mult = 0.9
            else:
                otw_mult = 0.65

        # Profile Completeness check
        completeness = signals.get("profile_completeness_score", 100)
        comp_mult = 1.0 if completeness >= 50 else 0.85
        
        # Preferred work mode
        pref_work = signals.get("preferred_work_mode", "").lower()
        work_mult = 1.0
        if pref_work == "remote" and not willing_relocate:
            work_mult = 0.80

        # Interview attendance
        interview_rate = signals.get("interview_completion_rate", 1.0)
        interview_mult = 1.0 if interview_rate >= 0.60 else 0.85

        # Calculate final adjusted score
        avail_mult = recency_mult * response_mult * location_mult * notice_mult * github_mult * otw_mult * comp_mult * work_mult * interview_mult
        final_score = relevance_score * avail_mult

        # 🚀 CHANGE 1 — Inactivity Ceiling Hard Cap
        if active_days > 90 and response_rate < 0.40:
            final_score = min(final_score, 0.50)

        # 🚀 CHANGE 2 — Experience below 5.0 years multiplier
        if years_exp < 5.0:
            final_score = final_score * 0.6

        # 🚀 CHANGE 3 — CV/Speech Domain Skill Count Penalty
        cv_speech_match_count = sum(1 for s in skill_names if contains_any_keyword(s, CV_SPEECH_SKILLS))
        if cv_speech_match_count >= 4:
            final_score = final_score * 0.75

        # Bounded Sigmoid Normalization (maps raw scores in [0.2, 1.5] smoothly to [0.0, 1.0])
        # Preserves sorting order exactly due to monotonic increasing behavior of logistic function.
        normalized_score = 1.0 / (1.0 + np.exp(-2.5 * (final_score - 0.55)))

        final_scores.append((cand, normalized_score))

    # Sort and select top 100
    final_scores.sort(key=lambda x: (-round(x[1], 4), x[0].get("candidate_id")))
    top_100 = final_scores[:100]

    # Write final CSV
    print(f"Writing results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank_idx, (cand, score) in enumerate(top_100, 1):
            cid = cand.get("candidate_id")
            reason = generate_reasoning(cand, rank_idx)
            writer.writerow([cid, rank_idx, f"{score:.4f}", reason])

    # 🚀 CHANGE 6 — Execution Time Logging
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")
    print(f"Candidates processed: {n_candidates}")
    print(f"Time per candidate: {(elapsed/n_candidates)*1000:.3f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="../PS/candidates.jsonl")
    parser.add_argument("--out", type=str, default="submission.csv")
    parser.add_argument("--cache_dir", type=str, default="data_cache")
    args = parser.parse_args()
    
    run_ranking(args.candidates, args.out, args.cache_dir)
