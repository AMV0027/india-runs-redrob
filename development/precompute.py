"""
precompute.py - Offline pre-computation for the Redrob Hackathon ranking system.

Processes candidates.jsonl to produce:
  - features.parquet     : Structured candidate features (experience, penalties, signals)
  - skills.index         : FAISS L2 index over skills embeddings
  - exp.index            : FAISS L2 index over experience embeddings
  - bm25.pkl             : BM25 index + combined corpus texts + candidate_ids list

Design goals:
  - Stream JSONL line-by-line to avoid loading 487MB into RAM at once
  - Replace slow NLI zero-shot classifier with fast rule-based keyword heuristics
  - Use batch embedding for efficiency
  - Target total runtime: ~10 minutes on a 16GB CPU machine
"""

import json
import argparse
import re
import os
import pickle
from datetime import datetime, date

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CONSULTING_FIRMS = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'ibm',
    'hcl', 'mphasis', 'hexaware', 'l&t infotech', 'mindtree', 'tech mahindra',
    'niit technologies', 'mastech', 'cyient', 'kpit',
}

# Keywords suggesting pure academic / paper-writing work
RESEARCH_KEYWORDS = re.compile(
    r'\b(phd|arxiv|preprint|paper|publication|published|conference|journal|'
    r'research lab|research intern|research scientist|postdoc|university lab|'
    r'ieee|neurips|iclr|icml|acl|emnlp)\b',
    re.IGNORECASE
)

# Keywords for "LangChain wrapper" / API-only AI experience
LANGCHAIN_KEYWORDS = re.compile(
    r'\b(langchain|langsmith|llamaindex|llama.?index|openai api|chatgpt api|'
    r'claude api|anthropic api|no.?code|low.?code|prompt engineering only|'
    r'gpt wrapper|api wrapper)\b',
    re.IGNORECASE
)

# Keywords that indicate genuine production ML/AI work
PRODUCTION_AI_KEYWORDS = re.compile(
    r'\b(deployed|production|serving|inference|pipeline|vector search|'
    r'embedding|retrieval|ranking|recommendation|fine.?tun|rlhf|'
    r'bert|transformer|xgboost|pytorch|tensorflow|faiss|elasticsearch|'
    r'pinecone|weaviate|qdrant|milvus|mlflow|kubeflow|a/b test|'
    r'real.?time|latency|throughput|scale|million|billion)\b',
    re.IGNORECASE
)

REFERENCE_DATE = date(2025, 1, 1)   # Treat dataset as current ~Jan 2025


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def is_consulting_only(career_history: list) -> bool:
    """True only if every employer is a known consulting/services firm."""
    if not career_history:
        return False
    for exp in career_history:
        company = exp.get('company', '').lower()
        if not any(firm in company for firm in CONSULTING_FIRMS):
            return False
    return True


def get_title_chaser_flag(career_history: list, years_exp: float) -> bool:
    """True if average tenure < 1.5 years across > 2 jobs."""
    if len(career_history) <= 2:
        return False
    avg_tenure_months = sum(e.get('duration_months', 0) for e in career_history) / len(career_history)
    return avg_tenure_months < 18  # 18 months = 1.5 years


def get_honeypot_flag(skills: list) -> bool:
    """True if any skill has 0 months used but 'expert' proficiency."""
    for sk in skills:
        if sk.get('duration_months', 1) == 0 and sk.get('proficiency', '').lower() == 'expert':
            return True
    return False


def get_ghost_flag(signals: dict) -> bool:
    """
    True if candidate is effectively unavailable:
      - last active > 6 months ago AND recruiter response rate < 10%
    """
    last_active_str = signals.get('last_active_date', '')
    response_rate = signals.get('recruiter_response_rate', 1.0)
    try:
        last_active = datetime.strptime(last_active_str, '%Y-%m-%d').date()
        days_inactive = (REFERENCE_DATE - last_active).days
        if days_inactive > 180 and response_rate < 0.10:
            return True
    except (ValueError, TypeError):
        pass
    return False


def classify_research_vs_production(career_history: list) -> tuple:
    """
    Fast regex-based classification, replacing slow NLI model.
    Returns (is_pure_research: bool, is_langchain_wrapper: bool)
    """
    all_descriptions = ' '.join(
        exp.get('description', '') for exp in career_history
    )

    research_hits = len(RESEARCH_KEYWORDS.findall(all_descriptions))
    langchain_hits = len(LANGCHAIN_KEYWORDS.findall(all_descriptions))
    production_hits = len(PRODUCTION_AI_KEYWORDS.findall(all_descriptions))

    # "Pure research" if heavy research signal and weak production signal
    is_pure_research = research_hits >= 3 and production_hits < 3

    # "LangChain wrapper" if heavy API-calling and weak production signal
    is_langchain_wrapper = langchain_hits >= 2 and production_hits < 2

    return is_pure_research, is_langchain_wrapper


def build_candidate_text(cand: dict) -> tuple:
    """Return (skills_text, exp_text, profile_text, combined_text) for a candidate."""
    profile = cand.get('profile', {})
    skills = cand.get('skills', [])
    career = cand.get('career_history', [])

    skills_text = ' '.join(s.get('name', '') for s in skills)
    exp_text = ' '.join(
        f"{e.get('title', '')} {e.get('description', '')}" for e in career
    )
    profile_text = f"{profile.get('headline', '')} {profile.get('summary', '')}"
    combined = f"{skills_text} {exp_text} {profile_text}"
    return skills_text, exp_text, profile_text, combined


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Offline pre-computation for Redrob ranker.')
    parser.add_argument('--input', type=str, required=True, help='Path to candidates.jsonl or sample_candidates.json')
    parser.add_argument('--out_dir', type=str, default='.', help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=512, help='Embedding batch size')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Stream JSONL / JSON ─────────────────────────────────────────────────
    print(f"[1/5] Streaming candidates from {args.input} ...")
    features_rows = []
    skills_texts = []
    exp_texts = []
    combined_texts = []
    candidate_ids = []

    def process_candidate(cand):
        cid = cand.get('candidate_id', '')
        profile = cand.get('profile', {})
        career = cand.get('career_history', [])
        skills = cand.get('skills', [])
        signals = cand.get('redrob_signals', {})

        years_exp = profile.get('years_of_experience', 0.0)

        is_pure_research, is_langchain_wrapper = classify_research_vs_production(career)

        row = {
            'candidate_id': cid,
            'years_of_experience': years_exp,
            'honeypot_flag': get_honeypot_flag(skills),
            'ghost_flag': get_ghost_flag(signals),
            'is_consulting_only': is_consulting_only(career),
            'title_chaser_flag': get_title_chaser_flag(career, years_exp),
            'is_pure_research': is_pure_research,
            'is_langchain_wrapper': is_langchain_wrapper,
            'notice_period_days': signals.get('notice_period_days', 90),
            'willing_to_relocate': bool(signals.get('willing_to_relocate', False)),
            'profile_completeness_score': signals.get('profile_completeness_score', 0),
            'open_to_work': bool(signals.get('open_to_work_flag', False)),
            'recruiter_response_rate': signals.get('recruiter_response_rate', 0.5),
            'github_activity_score': signals.get('github_activity_score', -1),
        }

        st, et, _, ct = build_candidate_text(cand)
        return row, st, et, ct

    if args.input.endswith('.jsonl'):
        with open(args.input, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                cand = json.loads(line)
                row, st, et, ct = process_candidate(cand)
                features_rows.append(row)
                skills_texts.append(st)
                exp_texts.append(et)
                combined_texts.append(ct)
                candidate_ids.append(cand.get('candidate_id', ''))
                if (i + 1) % 10000 == 0:
                    print(f"  Processed {i+1:,} candidates...")
    else:
        with open(args.input, 'r', encoding='utf-8') as f:
            candidates = json.load(f)
        for cand in candidates:
            row, st, et, ct = process_candidate(cand)
            features_rows.append(row)
            skills_texts.append(st)
            exp_texts.append(et)
            combined_texts.append(ct)
            candidate_ids.append(cand.get('candidate_id', ''))

    total = len(features_rows)
    print(f"  Done. {total:,} candidates processed.")

    # ── 2. Save features ───────────────────────────────────────────────────────
    print("[2/5] Saving features.parquet ...")
    df_features = pd.DataFrame(features_rows)
    df_features.to_parquet(os.path.join(args.out_dir, 'features.parquet'), index=False)

    # ── 3. Embeddings ─────────────────────────────────────────────────────────
    print("[3/5] Computing embeddings (all-MiniLM-L6-v2) ...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    print("  Encoding skills ...")
    skills_vecs = embedder.encode(
        skills_texts, batch_size=args.batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True
    )
    print("  Encoding experience ...")
    exp_vecs = embedder.encode(
        exp_texts, batch_size=args.batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True
    )

    # ── 4. FAISS indices ───────────────────────────────────────────────────────
    print("[4/5] Building FAISS indices ...")
    dim = skills_vecs.shape[1]

    # Use Inner Product (IP) on normalized vecs == cosine similarity → faster & better
    idx_skills = faiss.IndexFlatIP(dim)
    idx_skills.add(skills_vecs.astype(np.float32))
    faiss.write_index(idx_skills, os.path.join(args.out_dir, 'skills.index'))

    idx_exp = faiss.IndexFlatIP(dim)
    idx_exp.add(exp_vecs.astype(np.float32))
    faiss.write_index(idx_exp, os.path.join(args.out_dir, 'exp.index'))

    # ── 5. BM25 ────────────────────────────────────────────────────────────────
    print("[5/5] Building BM25 index ...")
    tokenized = [doc.lower().split() for doc in combined_texts]
    bm25 = BM25Okapi(tokenized)

    with open(os.path.join(args.out_dir, 'bm25.pkl'), 'wb') as f:
        pickle.dump({
            'bm25': bm25,
            'candidate_ids': candidate_ids,
            'combined_texts': combined_texts,
        }, f, protocol=4)

    print("\nPrecomputation complete. Artifacts saved to:", args.out_dir)
    print(f"  features.parquet : {total:,} rows")
    print(f"  skills.index     : {idx_skills.ntotal:,} vectors (dim={dim})")
    print(f"  exp.index        : {idx_exp.ntotal:,} vectors (dim={dim})")
    print(f"  bm25.pkl         : BM25 over {total:,} docs")


if __name__ == '__main__':
    main()
