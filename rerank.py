"""
rerank.py — Cross-Encoder Re-Ranking Module
============================================
Stage 3 of the multi-stage ranking pipeline.

Given a shortlist of 2,000 pre-filtered candidates (from preprocess.py),
this module jointly encodes the Job Description and each candidate's profile
text through a Cross-Encoder transformer, yielding a precise relevance score
that captures fine-grained cross-attention interactions across all 3 text segments.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS-MARCO passage ranking benchmark
  - 66M parameters, CPU-friendly (~1.8s per batch of 32 pairs)
  - Outputs a single relevance logit per (query, passage) pair
"""
import os
import re
import time
from typing import List, Dict, Any, Tuple

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BATCH_SIZE = 32          # Number of pairs per cross-encoder inference call
MAX_CANDIDATE_CHARS = 768  # Expanded to prevent loss of achievement and skill information


# ─── Candidate Text Builder ───────────────────────────────────────────────────

def build_candidate_text(candidate: Dict[str, Any]) -> str:
    """
    Constructs a rich candidate representation for the Cross-Encoder.
    Prioritizes: ownership level, career progression, production impact, and key capability evidence.
    """
    profile = candidate.get("profile", {})
    title = profile.get("current_title") or "Software Engineer"
    company = profile.get("current_company") or "Technology Company"
    years_exp = profile.get("years_of_experience") or 0.0

    # Extract dynamic capabilities from utils to represent strengths
    from utils import extract_candidate_features
    features = extract_candidate_features(candidate)
    
    # List top 2 strongest capability groups
    sorted_caps = sorted(features["strengths"].items(), key=lambda x: -x[1])
    top_caps = [c[0] for c in sorted_caps[:2] if c[1] > 0]
    caps_str = ", ".join(top_caps) if top_caps else "General Software Engineering"

    # Skills summary
    skills_raw = candidate.get("skills", [])
    skills_sorted = sorted(skills_raw, key=lambda s: s.get("duration_months", 0), reverse=True)
    skill_names = [s.get("name", "") for s in skills_sorted[:8] if s.get("name")]

    # Career history progression & ownership achievements
    career = candidate.get("career_history", [])
    achievements = []
    ownership_signals = ["led", "managed", "designed", "built", "implemented", "scaled", "optimized", "production", "deployment"]
    
    for idx, job in enumerate(career[:3]):
        desc = (job.get("description") or "").lower()
        # Extract sentences with ownership words
        sentences = re.split(r'\. |\n', desc)
        for s in sentences:
            if any(word in s for word in ownership_signals):
                achievements.append(s.strip()[:100])
                break # 1 achievement per job to keep text concise
                
    achievements_text = "; ".join(achievements[:3])
    
    parts = [
        f"{title} at {company}",
        f"{years_exp:.1f} years of experience",
        f"Core Capabilities: {caps_str}",
        f"Top Skills: {', '.join(skill_names)}",
        f"Achievements & Production Experience: {achievements_text}" if achievements_text else ""
    ]

    text = ". ".join(p.strip() for p in parts if p.strip())
    return text[:MAX_CANDIDATE_CHARS]


# ─── Cross-Encoder Re-Ranker ──────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Wraps an ONNX-optimized CrossEncoder model to re-rank a candidate
    shortlist by precise relevance to a Job Description query.

    Automatically handles dynamic model export to ONNX on first run.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.cache_dir = "data_cache"
        self.onnx_path = os.path.join(self.cache_dir, "model.onnx")

        # Ensure model is exported to ONNX
        if not os.path.exists(self.onnx_path):
            self._export_model_to_onnx()

        print(f"Loading Tokenizer for: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        print(f"Loading ONNX Inference Session from: {self.onnx_path}")
        t0 = time.time()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4  # Sweet spot for CPU context execution
        opts.inter_op_num_threads = 4
        self.sess = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        print(f"ONNX Session loaded in {time.time() - t0:.2f}s")

    def _export_model_to_onnx(self):
        print(f"ONNX model not found. Exporting {self.model_name} to ONNX...")
        t0 = time.time()
        os.makedirs(self.cache_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.eval()

        dummy_text = [
            ("query placeholder 1", "passage placeholder 1"),
            ("query placeholder 2", "passage placeholder 2")
        ]
        inputs = tokenizer(dummy_text, padding=True, truncation=True, max_length=512, return_tensors="pt")

        batch_dim = torch.export.Dim("batch_size", min=1, max=1024)
        seq_dim = torch.export.Dim("sequence_length", min=1, max=512)
        
        dynamic_shapes = {
            "input_ids": {0: batch_dim, 1: seq_dim},
            "attention_mask": {0: batch_dim, 1: seq_dim},
            "token_type_ids": {0: batch_dim, 1: seq_dim},
        }

        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
            self.onnx_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_shapes=dynamic_shapes,
            opset_version=18,
        )
        print(f"ONNX export completed in {time.time() - t0:.2f}s")

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
        Runs cross-encoder inference in batches using ONNX runtime and returns raw logit scores.
        """
        all_scores: List[float] = []
        total_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in tqdm(range(0, len(pairs), BATCH_SIZE),
                      total=total_batches,
                      desc="CE Batches"):
            batch = pairs[i : i + BATCH_SIZE]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np"
            )

            onnx_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "token_type_ids": inputs["token_type_ids"],
            }

            logits = self.sess.run(["logits"], onnx_inputs)[0]
            scores = logits.squeeze(axis=-1).tolist()
            if isinstance(scores, float):
                scores = [scores]

            all_scores.extend(scores)

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
