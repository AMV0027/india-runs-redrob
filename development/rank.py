"""
rank.py - Runtime ranking script for the Redrob Hackathon.

Must complete in ≤ 5 minutes on CPU, ≤ 16 GB RAM, no network access.

Pipeline:
  1. Load precomputed FAISS indices, BM25, features.parquet
  2. Embed JD query (skills + experience text)
  3. Hybrid retrieval: Union of top-K from FAISS (cosine) + BM25
  4. Additive scoring: exp_match, skills_match, bm25, structured_exp, behavioral
  5. Hard multipliers: honeypot=0, ghost=0, consulting=0.5, title_chaser=0.5, pure_research=0.1
  6. Cross-Encoder reranking on top-300 candidates
  7. Write top-100 to submission.csv with reasoning
"""

import argparse
import json
import os
import time
import pickle

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

# ──────────────────────────────────────────────────────────────────────────────
# JD Definition (Senior AI Engineer — Founding Team)
# ──────────────────────────────────────────────────────────────────────────────

JD_SKILLS = (
    "Python embeddings vector search retrieval ranking NLP LLMs fine-tuning "
    "sentence-transformers FAISS Pinecone Weaviate Qdrant hybrid search "
    "NDCG MRR MAP evaluation A/B testing recommendation system"
)
JD_EXP = (
    "5 to 9 years applied ML AI engineer production deployment real users "
    "end-to-end ranking search recommendation pipeline product company "
    "shipped vector search embedding retrieval at scale"
)
JD_FULL = JD_SKILLS + " " + JD_EXP

# Ideal experience band from JD
EXP_MIN, EXP_MAX = 5, 9


# ──────────────────────────────────────────────────────────────────────────────
# Reasoning generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_reasoning(row: pd.Series, cand_lookup: dict) -> str:
    """
    Build a factual 1-2 sentence reasoning string for a candidate.
    Only uses data actually present in the candidate's profile.
    """
    cid = row['candidate_id']
    raw = cand_lookup.get(cid, {})
    profile = raw.get('profile', {})
    signals = raw.get('redrob_signals', {})
    skills_list = raw.get('skills', [])

    title = profile.get('current_title', 'Engineer')
    yoe = row['years_of_experience']
    response_rate = signals.get('recruiter_response_rate', 0.0)
    notice = signals.get('notice_period_days', 90)
    open_to_work = signals.get('open_to_work_flag', False)

    # Count AI-relevant skills
    ai_keywords = {'python', 'ml', 'ai', 'nlp', 'llm', 'deep learning',
                   'machine learning', 'data science', 'embedding', 'retrieval',
                   'ranking', 'recommendation', 'pytorch', 'tensorflow',
                   'vector', 'transformer', 'bert'}
    ai_skills = [s['name'] for s in skills_list
                 if any(k in s.get('name', '').lower() for k in ai_keywords)]
    ai_skills_str = ', '.join(ai_skills[:4]) if ai_skills else 'no explicit AI skills listed'

    # Build base sentence
    base = (
        f"{title} with {yoe:.1f} yrs experience; "
        f"AI-relevant skills: {ai_skills_str}; "
        f"recruiter response rate {response_rate:.0%}."
    )

    # Add nuance sentence
    nuances = []
    if row['honeypot_flag']:
        nuances.append("Flagged as honeypot (impossible skill proficiency/duration).")
    elif row['ghost_flag']:
        nuances.append("Low recent platform engagement — likely unavailable.")
    else:
        if open_to_work:
            nuances.append("Actively seeking new role.")
        if notice <= 30:
            nuances.append(f"Sub-30-day notice period ({notice}d) — ideal for quick hire.")
        elif notice <= 60:
            nuances.append(f"Moderate notice period ({notice}d).")
        if row['is_consulting_only']:
            nuances.append("Career exclusively at services/consulting firms.")
        if row['is_pure_research']:
            nuances.append("Background skews research/academic, limited production deployment.")
        if row['is_langchain_wrapper']:
            nuances.append("AI experience primarily via LangChain API calls.")
        if row['title_chaser_flag']:
            nuances.append("Short average tenure — possible title-chasing pattern.")

    nuance_str = ' '.join(nuances)
    full = f"{base} {nuance_str}".strip()
    return full[:280]   # Submission spec says 1-2 sentences; keep readable


# ──────────────────────────────────────────────────────────────────────────────
# Structured experience score
# ──────────────────────────────────────────────────────────────────────────────

def structured_exp_score(yoe: float) -> float:
    """
    Peaks at 1.0 for 5–9 years, decays symmetrically outside the band.
    Candidates with <2 yrs or >15 yrs score near 0.
    """
    if EXP_MIN <= yoe <= EXP_MAX:
        return 1.0
    elif yoe < EXP_MIN:
        return max(0.0, yoe / EXP_MIN)          # linear ramp from 0→5 yrs
    else:
        return max(0.0, 1.0 - (yoe - EXP_MAX) / 10.0)   # linear decay 9→19 yrs


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True, help='candidates.jsonl or sample_candidates.json')
    parser.add_argument('--out', required=True, help='Output submission.csv path')
    parser.add_argument('--data_dir', default='.', help='Directory with precomputed artifacts')
    parser.add_argument('--top_k_retrieval', type=int, default=2000, help='Top-K per retrieval method')
    parser.add_argument('--top_k_rerank', type=int, default=300, help='Candidates sent to cross-encoder')
    args = parser.parse_args()

    # ── 1. Load artifacts ──────────────────────────────────────────────────────
    print("[1/6] Loading precomputed artifacts ...")
    df = pd.read_parquet(os.path.join(args.data_dir, 'features.parquet'))

    with open(os.path.join(args.data_dir, 'tfidf_skills.pkl'), 'rb') as f:
        tfidf_skills_store = pickle.load(f)
    tfidf_skills_vec    = tfidf_skills_store['vectorizer']
    tfidf_skills_matrix = tfidf_skills_store['matrix']   # (n_docs, vocab) sparse, L2-normed

    with open(os.path.join(args.data_dir, 'tfidf_exp.pkl'), 'rb') as f:
        tfidf_exp_store = pickle.load(f)
    tfidf_exp_vec    = tfidf_exp_store['vectorizer']
    tfidf_exp_matrix = tfidf_exp_store['matrix']         # (n_docs, vocab) sparse, L2-normed

    with open(os.path.join(args.data_dir, 'bm25.pkl'), 'rb') as f:
        bm25_store = pickle.load(f)
    bm25 = bm25_store['bm25']
    combined_texts = bm25_store['combined_texts']
    cid_list = bm25_store['candidate_ids']

    # ── 2. Load raw candidates for reasoning (only what we need) ──────────────
    print("[2/6] Loading raw candidate data for reasoning ...")
    cand_lookup = {}
    if args.candidates.endswith('.jsonl'):
        with open(args.candidates, 'r', encoding='utf-8') as f:
            for line in f:
                c = json.loads(line)
                cand_lookup[c['candidate_id']] = c
    else:
        with open(args.candidates, 'r', encoding='utf-8') as f:
            for c in json.load(f):
                cand_lookup[c['candidate_id']] = c
    print(f"  Loaded {len(cand_lookup):,} candidates.")

    # ── 3. TF-IDF retrieval ────────────────────────────────────────────────────
    print("[3/6] TF-IDF retrieval for JD query ...")
    k = min(args.top_k_retrieval, len(df))

    # Transform JD text using the saved vectorizers; L2-norm for cosine sim
    q_skills_vec = normalize(tfidf_skills_vec.transform([JD_SKILLS]), norm='l2')  # (1, vocab)
    q_exp_vec    = normalize(tfidf_exp_vec.transform([JD_EXP]),    norm='l2')  # (1, vocab)

    # Cosine sim = dot product since both query and matrix are L2-normed
    # linear_kernel returns (1, n_docs); squeeze to 1-D array
    skills_sims = linear_kernel(q_skills_vec, tfidf_skills_matrix).flatten()  # shape (n_docs,)
    exp_sims    = linear_kernel(q_exp_vec,    tfidf_exp_matrix).flatten()

    top_skills_idx = np.argpartition(skills_sims, -k)[-k:]   # unordered top-K (fast)
    top_exp_idx    = np.argpartition(exp_sims,    -k)[-k:]

    # Build score maps (index -> similarity) for later scoring
    skills_score_map = dict(zip(top_skills_idx.tolist(), skills_sims[top_skills_idx].tolist()))
    exp_score_map    = dict(zip(top_exp_idx.tolist(),    exp_sims[top_exp_idx].tolist()))

    # BM25
    q_tokens = JD_FULL.lower().split()
    bm25_scores_all = np.array(bm25.get_scores(q_tokens), dtype=np.float32)
    top_bm25_idx = np.argsort(bm25_scores_all)[::-1][:k]

    # Union of candidate indices
    pool_set = (
        set(top_skills_idx.tolist()) |
        set(top_exp_idx.tolist()) |
        set(top_bm25_idx.tolist())
    )
    pool_set.discard(-1)
    pool_indices = sorted(pool_set)
    print(f"  Pool size after union: {len(pool_indices):,}")

    # ── 4. Score the candidate pool ────────────────────────────────────────────
    print("[4/6] Additive scoring ...")
    pool_df = df.iloc[pool_indices].copy()
    pool_df['_idx'] = pool_indices

    # TF-IDF cosine scores are in [0, 1] (L2-normed dot product);
    # shift to [0.5, 1] to match the spirit of the old FAISS (x+1)/2 rescaling.
    pool_df['skills_match'] = pool_df['_idx'].map(skills_score_map).fillna(0.0).apply(lambda x: 0.5 + x / 2)
    pool_df['exp_match']    = pool_df['_idx'].map(exp_score_map).fillna(0.0).apply(lambda x: 0.5 + x / 2)

    # BM25 normalized
    bm25_pool = bm25_scores_all[pool_indices]
    bm25_max = bm25_pool.max()
    pool_df['bm25_score'] = bm25_pool / bm25_max if bm25_max > 0 else 0

    # Structured signals
    pool_df['struct_exp']  = pool_df['years_of_experience'].apply(structured_exp_score)
    pool_df['behavioral']  = (
        pool_df['profile_completeness_score'] / 100.0 * 0.4 +
        pool_df['recruiter_response_rate'] * 0.4 +
        pool_df['open_to_work'].astype(float) * 0.2
    )

    # Additive base score (weights sum to 1.0)
    pool_df['base_score'] = (
        0.35 * pool_df['exp_match'] +
        0.25 * pool_df['skills_match'] +
        0.15 * pool_df['bm25_score'] +
        0.15 * pool_df['struct_exp'] +
        0.10 * pool_df['behavioral']
    )

    # Hard multipliers
    pool_df['multiplier'] = 1.0
    pool_df.loc[pool_df['honeypot_flag'],       'multiplier'] = 0.0
    pool_df.loc[pool_df['ghost_flag'],          'multiplier'] = 0.0
    pool_df.loc[pool_df['is_pure_research'],    'multiplier'] = pool_df.loc[pool_df['is_pure_research'], 'multiplier'].clip(upper=0.1)
    pool_df.loc[pool_df['is_langchain_wrapper'],'multiplier'] = pool_df.loc[pool_df['is_langchain_wrapper'], 'multiplier'].clip(upper=0.2)
    pool_df.loc[pool_df['is_consulting_only'],  'multiplier'] = pool_df.loc[pool_df['is_consulting_only'],  'multiplier'].clip(upper=0.5)
    pool_df.loc[pool_df['title_chaser_flag'],   'multiplier'] = pool_df.loc[pool_df['title_chaser_flag'],   'multiplier'].clip(upper=0.7)

    pool_df['score'] = pool_df['base_score'] * pool_df['multiplier']

    # ── 5. Cross-Encoder reranking on top-K ───────────────────────────────────
    top_k_re = min(args.top_k_rerank, len(pool_df))
    top_ce = pool_df.nlargest(top_k_re, 'score').copy()
    print(f"[5/6] Cross-encoder reranking {len(top_ce)} candidates ...")

    try:
        ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
        ce_pairs  = [[JD_FULL, combined_texts[i]] for i in top_ce['_idx'].tolist()]
        ce_raw    = ce_model.predict(ce_pairs, show_progress_bar=True)

        ce_min, ce_max = ce_raw.min(), ce_raw.max()
        ce_norm = (ce_raw - ce_min) / (ce_max - ce_min + 1e-9)
        top_ce['ce_score'] = ce_norm

        # Final score: blend cross-encoder (semantic quality) + base (structured + behavioral)
        top_ce['final_score'] = (
            (0.55 * top_ce['ce_score'] + 0.45 * top_ce['base_score'])
            * top_ce['multiplier']
        )
    except Exception as exc:
        print(f"  Cross-encoder failed ({exc}), falling back to base score.")
        top_ce['final_score'] = top_ce['score']

    # Tie-breaker: sub-30 notice period preferred (JD says they can buy out 30 days)
    top_ce['notice_boost'] = top_ce['notice_period_days'].apply(lambda d: 0.001 if d <= 30 else 0)
    top_ce['final_score'] += top_ce['notice_boost']

    # ── 6. Build submission ────────────────────────────────────────────────────
    print("[6/6] Writing submission.csv ...")
    top_100 = (
        top_ce
        .sort_values(['final_score', 'notice_period_days', 'candidate_id'],
                     ascending=[False, True, True])
        .head(100)
        .copy()
    )
    top_100 = top_100.reset_index(drop=True)
    top_100['rank'] = range(1, len(top_100) + 1)
    top_100['reasoning'] = top_100.apply(
        lambda r: generate_reasoning(r, cand_lookup), axis=1
    )

    submission = top_100[['candidate_id', 'rank', 'final_score', 'reasoning']].rename(
        columns={'final_score': 'score'}
    )
    submission.to_csv(args.out, index=False)

    elapsed = time.time() - t0
    print(f"\nDone. Submission saved to {args.out}")
    print(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Top candidate: {top_100.iloc[0]['candidate_id']}  score={top_100.iloc[0]['final_score']:.4f}")


if __name__ == '__main__':
    main()
