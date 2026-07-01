"""
precompute.py - Offline pre-computation for the Redrob Hackathon ranking system.

Processes candidates.jsonl to produce:
  - features.parquet     : Structured candidate features (experience, penalties, signals)
  - tfidf_skills.pkl     : TF-IDF vectorizer + L2-normalised sparse matrix for skills
  - tfidf_exp.pkl        : TF-IDF vectorizer + L2-normalised sparse matrix for experience
  - bm25.pkl             : BM25 index + combined corpus texts + candidate_ids list

Key optimisations:
  [#1]  Parallel feature extraction via ProcessPoolExecutor
  [#7]  Streaming chunks to bound peak RAM
  [#8]  PyArrow batched Parquet writes
  [#9]  orjson for 2-5x faster JSON parsing
  [#11] Direct token-list construction, no intermediate strings
  [#15] casefold() for BM25 tokenisation
  [NEW] TF-IDF sparse vectors replace neural embeddings (56 min -> ~30 s)

Runtime target: ~1-2 min on any CPU (vs 56 min with neural embeddings).
"""

import argparse
import os
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, date
from itertools import islice

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rank_bm25 import BM25Okapi

# Optional fast JSON – fall back to stdlib if unavailable
try:
    import orjson as json_lib          # [#9] orjson is 2–5× faster
    def _loads(b): return json_lib.loads(b)
except ImportError:
    import json as json_lib
    def _loads(b): return json_lib.loads(b)

# ──────────────────────────────────────────────────────────────────────────────
# Constants  (module-level so pickling works for multiprocessing)
# ──────────────────────────────────────────────────────────────────────────────

CONSULTING_FIRMS = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'ibm',
    'hcl', 'mphasis', 'hexaware', 'l&t infotech', 'mindtree', 'tech mahindra',
    'niit technologies', 'mastech', 'cyient', 'kpit',
}

RESEARCH_KEYWORDS = re.compile(
    r'\b(phd|arxiv|preprint|paper|publication|published|conference|journal|'
    r'research lab|research intern|research scientist|postdoc|university lab|'
    r'ieee|neurips|iclr|icml|acl|emnlp)\b',
    re.IGNORECASE
)

LANGCHAIN_KEYWORDS = re.compile(
    r'\b(langchain|langsmith|llamaindex|llama.?index|openai api|chatgpt api|'
    r'claude api|anthropic api|no.?code|low.?code|prompt engineering only|'
    r'gpt wrapper|api wrapper)\b',
    re.IGNORECASE
)

PRODUCTION_AI_KEYWORDS = re.compile(
    r'\b(deployed|production|serving|inference|pipeline|vector search|'
    r'embedding|retrieval|ranking|recommendation|fine.?tun|rlhf|'
    r'bert|transformer|xgboost|pytorch|tensorflow|faiss|elasticsearch|'
    r'pinecone|weaviate|qdrant|milvus|mlflow|kubeflow|a/b test|'
    r'real.?time|latency|throughput|scale|million|billion)\b',
    re.IGNORECASE
)

REFERENCE_DATE = date(2025, 1, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers  (all pure functions – safe to pickle/multiprocess)
# ──────────────────────────────────────────────────────────────────────────────

def is_consulting_only(career_history: list) -> bool:
    if not career_history:
        return False
    for exp in career_history:
        company = exp.get('company', '').lower()
        if not any(firm in company for firm in CONSULTING_FIRMS):
            return False
    return True


def get_title_chaser_flag(career_history: list) -> bool:
    if len(career_history) <= 2:
        return False
    avg_tenure = sum(e.get('duration_months', 0) for e in career_history) / len(career_history)
    return avg_tenure < 18


def get_honeypot_flag(skills: list) -> bool:
    for sk in skills:
        if sk.get('duration_months', 1) == 0 and sk.get('proficiency', '').lower() == 'expert':
            return True
    return False


def get_ghost_flag(signals: dict) -> bool:
    last_active_str = signals.get('last_active_date', '')
    response_rate   = signals.get('recruiter_response_rate', 1.0)
    try:
        last_active  = datetime.strptime(last_active_str, '%Y-%m-%d').date()
        days_inactive = (REFERENCE_DATE - last_active).days
        if days_inactive > 180 and response_rate < 0.10:
            return True
    except (ValueError, TypeError):
        pass
    return False


def classify_research_vs_production(career_history: list) -> tuple:
    all_desc = ' '.join(exp.get('description', '') for exp in career_history)
    r_hits = len(RESEARCH_KEYWORDS.findall(all_desc))
    l_hits = len(LANGCHAIN_KEYWORDS.findall(all_desc))
    p_hits = len(PRODUCTION_AI_KEYWORDS.findall(all_desc))
    return (r_hits >= 3 and p_hits < 3), (l_hits >= 2 and p_hits < 2)


def build_candidate_texts(cand: dict) -> tuple:
    """
    Return (skills_tokens, exp_tokens, combined_tokens).

    [#11] Build token lists directly – avoids large intermediate strings
    and lets us reuse them for both embedding and BM25.
    """
    profile = cand.get('profile', {})
    skills  = cand.get('skills', [])
    career  = cand.get('career_history', [])

    skills_tokens   = [s.get('name', '') for s in skills]
    exp_tokens      = []
    for e in career:
        exp_tokens.append(e.get('title', ''))
        exp_tokens.append(e.get('description', ''))
    profile_tokens  = [profile.get('headline', ''), profile.get('summary', '')]

    # Flatten to strings for embedding; keep token lists for BM25
    skills_text  = ' '.join(skills_tokens)
    exp_text     = ' '.join(exp_tokens)
    combined_tok = [t.casefold() for t in (skills_tokens + exp_tokens + profile_tokens)
                    if t]                                           # [#15] casefold

    return skills_text, exp_text, combined_tok


# ──────────────────────────────────────────────────────────────────────────────
# Top-level worker  (must be importable at module level for ProcessPoolExecutor)
# ──────────────────────────────────────────────────────────────────────────────

def process_candidate(cand: dict) -> tuple:
    """
    Extract one candidate's features + texts.
    Returns (row_dict, skills_text, exp_text, combined_tokens, candidate_id).
    """
    cid     = cand.get('candidate_id', '')
    profile = cand.get('profile', {})
    career  = cand.get('career_history', [])
    skills  = cand.get('skills', [])
    signals = cand.get('redrob_signals', {})

    years_exp                       = profile.get('years_of_experience', 0.0)
    is_pure_research, is_lc_wrapper = classify_research_vs_production(career)

    row = {
        'candidate_id'              : cid,
        'years_of_experience'       : years_exp,
        'honeypot_flag'             : get_honeypot_flag(skills),
        'ghost_flag'                : get_ghost_flag(signals),
        'is_consulting_only'        : is_consulting_only(career),
        'title_chaser_flag'         : get_title_chaser_flag(career),
        'is_pure_research'          : is_pure_research,
        'is_langchain_wrapper'      : is_lc_wrapper,
        'notice_period_days'        : signals.get('notice_period_days', 90),
        'willing_to_relocate'       : bool(signals.get('willing_to_relocate', False)),
        'profile_completeness_score': signals.get('profile_completeness_score', 0),
        'open_to_work'              : bool(signals.get('open_to_work_flag', False)),
        'recruiter_response_rate'   : signals.get('recruiter_response_rate', 0.5),
        'github_activity_score'     : signals.get('github_activity_score', -1),
    }

    st, et, ctok = build_candidate_texts(cand)
    return row, st, et, ctok, cid


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _chunked(iterable, n):
    """Yield successive n-sized chunks from an iterable."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk


def _build_tfidf(texts: list, max_features: int = 60_000):
    """
    Fit a TF-IDF vectorizer on `texts` and return (vectorizer, L2-normalised sparse matrix).

    L2 normalisation means dot-product == cosine similarity at query time,
    matching the behaviour of the old FAISS IndexFlatIP.
    max_features=60_000 caps vocabulary to keep RAM manageable (100k × 60k sparse).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,          # log(1+tf) dampens very frequent terms
        min_df=2,                   # ignore hapax legomena
        ngram_range=(1, 2),         # unigrams + bigrams improve recall
        dtype=np.float32,
    )
    matrix = vec.fit_transform(texts)   # scipy sparse (n_docs, vocab)
    matrix = normalize(matrix, norm='l2', copy=False)  # in-place L2 norm
    return vec, matrix


# ──────────────────────────────────────────────────────────────────────────────
# PyArrow Parquet writer
# ──────────────────────────────────────────────────────────────────────────────

_PARQUET_SCHEMA = pa.schema([
    pa.field('candidate_id',               pa.string()),
    pa.field('years_of_experience',        pa.float32()),
    pa.field('honeypot_flag',              pa.bool_()),
    pa.field('ghost_flag',                 pa.bool_()),
    pa.field('is_consulting_only',         pa.bool_()),
    pa.field('title_chaser_flag',          pa.bool_()),
    pa.field('is_pure_research',           pa.bool_()),
    pa.field('is_langchain_wrapper',       pa.bool_()),
    pa.field('notice_period_days',         pa.int32()),
    pa.field('willing_to_relocate',        pa.bool_()),
    pa.field('profile_completeness_score', pa.float32()),
    pa.field('open_to_work',               pa.bool_()),
    pa.field('recruiter_response_rate',    pa.float32()),
    pa.field('github_activity_score',      pa.float32()),
])


def _rows_to_pa_batch(rows: list) -> pa.RecordBatch:
    """Convert a list of row dicts → PyArrow RecordBatch. [#8]"""
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    arrays = {
        'candidate_id'               : pa.array(cols['candidate_id'],               type=pa.string()),
        'years_of_experience'        : pa.array(cols['years_of_experience'],        type=pa.float32()),
        'honeypot_flag'              : pa.array(cols['honeypot_flag'],              type=pa.bool_()),
        'ghost_flag'                 : pa.array(cols['ghost_flag'],                 type=pa.bool_()),
        'is_consulting_only'         : pa.array(cols['is_consulting_only'],         type=pa.bool_()),
        'title_chaser_flag'          : pa.array(cols['title_chaser_flag'],          type=pa.bool_()),
        'is_pure_research'           : pa.array(cols['is_pure_research'],           type=pa.bool_()),
        'is_langchain_wrapper'       : pa.array(cols['is_langchain_wrapper'],       type=pa.bool_()),
        'notice_period_days'         : pa.array(cols['notice_period_days'],         type=pa.int32()),
        'willing_to_relocate'        : pa.array(cols['willing_to_relocate'],        type=pa.bool_()),
        'profile_completeness_score' : pa.array(cols['profile_completeness_score'], type=pa.float32()),
        'open_to_work'               : pa.array(cols['open_to_work'],               type=pa.bool_()),
        'recruiter_response_rate'    : pa.array(cols['recruiter_response_rate'],    type=pa.float32()),
        'github_activity_score'      : pa.array(cols['github_activity_score'],      type=pa.float32()),
    }
    field_names = [_PARQUET_SCHEMA.field(i).name for i in range(len(_PARQUET_SCHEMA))]
    return pa.record_batch([arrays[n] for n in field_names],
                           schema=_PARQUET_SCHEMA)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Offline pre-computation for Redrob ranker.')
    parser.add_argument('--input',        type=str, required=True)
    parser.add_argument('--out_dir',      type=str, default='.')
    parser.add_argument('--chunk_size',   type=int, default=10_000,
                        help='Feature-extraction chunk size for streaming [#7]')
    parser.add_argument('--workers',      type=int, default=None,
                        help='Worker processes for feature extraction (default: CPU count)')
    parser.add_argument('--max_features', type=int, default=60_000,
                        help='TF-IDF max vocabulary size per index')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    wall0 = time.time()

    # ── 1. Stream & extract features in parallel ───────────────────────────────
    print(f"[1/5] Streaming candidates from {args.input} …")
    t0 = time.time()

    def iter_candidates():
        """Yield parsed candidate dicts one by one."""
        if args.input.endswith('.jsonl'):
            with open(args.input, 'rb') as f:
                for line in f:
                    yield _loads(line)
        else:
            import json as _json
            with open(args.input, 'r', encoding='utf-8') as f:
                for cand in _json.load(f):
                    yield cand

    # Accumulators
    all_rows           = []
    all_skills_texts   = []
    all_exp_texts      = []
    all_combined_toks  = []
    all_candidate_ids  = []

    # [#7] Process in streaming chunks to bound RAM
    # [#1] Parallel feature extraction within each chunk
    workers = args.workers or os.cpu_count() or 4

    parquet_writer = None
    parquet_path   = os.path.join(args.out_dir, 'features.parquet')

    total_processed = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for chunk in _chunked(iter_candidates(), args.chunk_size):
            # [#1] Parallel map over the chunk
            results = list(ex.map(process_candidate, chunk, chunksize=64))

            chunk_rows, chunk_st, chunk_et, chunk_ct, chunk_ids = zip(*results)
            all_rows.extend(chunk_rows)
            all_skills_texts.extend(chunk_st)
            all_exp_texts.extend(chunk_et)
            all_combined_toks.extend(chunk_ct)
            all_candidate_ids.extend(chunk_ids)

            # [#8] Write features to Parquet in batches (streaming, avoids one huge DF)
            batch = _rows_to_pa_batch(list(chunk_rows))
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(parquet_path, _PARQUET_SCHEMA,
                                                  compression='snappy')
            parquet_writer.write_batch(batch)

            total_processed += len(chunk)
            print(f"  Processed {total_processed:,} candidates …", end='\r')

    if parquet_writer:
        parquet_writer.close()

    total = total_processed
    print(f"\n  Done. {total:,} candidates  |  stage time: {time.time()-t0:.1f}s")

    # ── 2. features.parquet already written above ─────────────────────────────
    print(f"[2/4] features.parquet written ({total:,} rows).")

    # ── 3. TF-IDF indices (replaces 56-min neural embedding stage) ────────────
    print("[3/4] Building TF-IDF sparse indices …")
    t2 = time.time()

    print(f"  Fitting skills TF-IDF on {total:,} documents …")
    skills_vec, skills_matrix = _build_tfidf(all_skills_texts, args.max_features)
    with open(os.path.join(args.out_dir, 'tfidf_skills.pkl'), 'wb') as f:
        pickle.dump({'vectorizer': skills_vec, 'matrix': skills_matrix,
                     'candidate_ids': all_candidate_ids}, f, protocol=4)
    del skills_matrix  # free RAM before fitting next vectorizer

    print(f"  Fitting experience TF-IDF on {total:,} documents …")
    exp_vec, exp_matrix = _build_tfidf(all_exp_texts, args.max_features)
    with open(os.path.join(args.out_dir, 'tfidf_exp.pkl'), 'wb') as f:
        pickle.dump({'vectorizer': exp_vec, 'matrix': exp_matrix,
                     'candidate_ids': all_candidate_ids}, f, protocol=4)
    del exp_matrix

    print(f"  TF-IDF done  |  stage time: {time.time()-t2:.1f}s")

    # ── 4. BM25 ────────────────────────────────────────────────────────────────
    print("[4/4] Building BM25 index …")
    t4 = time.time()
    # [#15] combined_toks already casefolded; [#11] no intermediate string needed
    bm25 = BM25Okapi(all_combined_toks)

    # Reconstruct combined_texts for downstream callers that expect strings
    combined_texts = [' '.join(tok) for tok in all_combined_toks]

    with open(os.path.join(args.out_dir, 'bm25.pkl'), 'wb') as f:
        pickle.dump({
            'bm25'           : bm25,
            'candidate_ids'  : all_candidate_ids,
            'combined_texts' : combined_texts,
        }, f, protocol=4)
    print(f"  BM25 done  |  stage time: {time.time()-t4:.1f}s")

    # ── Summary ────────────────────────────────────────────────────────────────
    wall_total = time.time() - wall0
    print("\n" + "-" * 60)
    print("Precomputation complete. Artifacts saved to:", args.out_dir)
    print(f"  features.parquet : {total:,} rows")
    print(f"  tfidf_skills.pkl : {total:,} docs")
    print(f"  tfidf_exp.pkl    : {total:,} docs")
    print(f"  bm25.pkl         : BM25 over {total:,} docs")
    print(f"\n  Total wall time  : {wall_total/60:.1f} min  ({wall_total:.0f}s)")
    print("-" * 60)


if __name__ == '__main__':
    main()