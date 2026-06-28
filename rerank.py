"""
rerank.py — Cross-Encoder Re-Ranking Module
============================================
Stage 3 of the two-stage retrieval + re-ranking pipeline.

Given a shortlist of ~400 pre-filtered candidates (from Stage 2 Bi-Encoder),
this module jointly encodes the Job Description and each candidate's profile
text through a Cross-Encoder transformer, yielding a precise relevance score
that captures fine-grained cross-attention interactions.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS-MARCO passage ranking benchmark
  - 66M parameters, CPU-friendly (~50–100ms per pair)
  - Outputs a single relevance logit per (query, passage) pair
"""

import time
from typing import List, Dict, Any, Tuple

from sentence_transformers import CrossEncoder
from tqdm import tqdm


# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BATCH_SIZE = 32          # Number of pairs per cross-encoder inference call
MAX_CANDIDATE_CHARS = 512  # Truncate candidate text to keep tokenization fast


# ─── Candidate Text Builder ───────────────────────────────────────────────────

def build_candidate_text(candidate: Dict[str, Any]) -> str:
    """
    Constructs a concise, informative text snippet for the candidate that the
    cross-encoder will jointly encode with the JD query.

    Prioritizes: current title, company, top skills, headline/summary, and
    a snippet of the current role description.
    """
    profile   = candidate.get("profile", {})
    title     = profile.get("current_title") or ""
    company   = profile.get("current_company") or ""
    location  = profile.get("location") or ""
    years_exp = profile.get("years_of_experience") or 0.0

    # Top 12 skills by duration
    skills_raw = candidate.get("skills", [])
    skills_sorted = sorted(skills_raw, key=lambda s: s.get("duration_months", 0), reverse=True)
    skill_names = [s.get("name", "") for s in skills_sorted[:12] if s.get("name")]

    # Headline / profile summary
    headline = candidate.get("profile_headline") or candidate.get("summary") or ""

    # Current role description (first 200 chars)
    career = candidate.get("career_history", [])
    current_desc = ""
    if career:
        current_desc = (career[0].get("description") or "")[:200]

    # Assemble the text
    parts = []
    if title:
        parts.append(f"{title} at {company}" if company else title)
    if location:
        parts.append(f"Location: {location}")
    parts.append(f"{years_exp:.1f} years experience")
    if skill_names:
        parts.append(f"Skills: {', '.join(skill_names)}")
    if headline:
        parts.append(headline[:200])
    if current_desc:
        parts.append(current_desc)

    text = ". ".join(p.strip() for p in parts if p.strip())
    return text[:MAX_CANDIDATE_CHARS]


# ─── Cross-Encoder Re-Ranker ──────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder model to re-rank a candidate
    shortlist by precise relevance to a Job Description query.

    Usage:
        reranker = CrossEncoderReranker()
        results  = reranker.rerank(jd_text, candidates, top_k=150)
        # results is a list of (candidate_dict, ce_score) tuples, sorted desc
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"Loading CrossEncoder model: {model_name}")
        t0 = time.time()
        self.model = CrossEncoder(model_name, max_length=512)
        print(f"CrossEncoder loaded in {time.time() - t0:.2f}s")

    def rerank(
        self,
        jd_text: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 150,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Re-ranks a list of candidates against the JD text using cross-attention.

        Args:
            jd_text:    The full Job Description text (query side).
            candidates: List of raw candidate dicts (from preprocessed_data.pkl).
            top_k:      Number of top candidates to return after re-ranking.

        Returns:
            List of (candidate_dict, cross_encoder_score) tuples, sorted by
            score descending, truncated to top_k.
        """
        if not candidates:
            return []

        print(f"\nCross-Encoder: building {len(candidates)} input pairs...")
        t0 = time.time()

        # Build (query, passage) pairs
        pairs: List[Tuple[str, str]] = []
        for cand in candidates:
            cand_text = build_candidate_text(cand)
            pairs.append((jd_text, cand_text))

        # Batch inference
        print(f"Cross-Encoder: scoring {len(pairs)} pairs in batches of {BATCH_SIZE}...")
        scores = self._batch_predict(pairs)

        # Pair candidates with their scores and sort
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        elapsed = time.time() - t0
        print(f"Cross-Encoder: re-ranking complete in {elapsed:.2f}s")
        if scored:
            print(f"  Score range: {scored[-1][1]:.4f} -> {scored[0][1]:.4f}")

        return scored[:top_k]

    def _batch_predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Runs cross-encoder inference in batches and returns raw logit scores.
        """
        all_scores: List[float] = []
        total_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in tqdm(range(0, len(pairs), BATCH_SIZE),
                      total=total_batches,
                      desc="CE Batches"):
            batch = pairs[i : i + BATCH_SIZE]
            batch_scores = self.model.predict(batch, show_progress_bar=False)
            # predict() returns a numpy array; convert to Python floats
            all_scores.extend(float(s) for s in batch_scores)

        return all_scores


# ─── Score Normalizer ─────────────────────────────────────────────────────────

def normalize_ce_scores(
    scored_candidates: List[Tuple[Dict[str, Any], float]],
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Min-max normalizes raw cross-encoder logits to the [0.0, 1.0] range
    so they can be blended with rule-based scores on the same scale.

    The cross-encoder outputs raw logits (can be negative or > 1). We
    normalize empirically so the best candidate in the shortlist = 1.0.
    """
    if not scored_candidates:
        return []

    raw_scores = [s for _, s in scored_candidates]
    min_s = min(raw_scores)
    max_s = max(raw_scores)
    span  = max_s - min_s

    if span < 1e-9:
        # All scores identical — return uniform 0.5
        return [(c, 0.5) for c, _ in scored_candidates]

    normalized = [(c, (s - min_s) / span) for c, s in scored_candidates]
    return normalized
