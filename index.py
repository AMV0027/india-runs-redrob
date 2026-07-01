import json
import numpy as np
import pandas as pd
from datetime import datetime

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return datetime(2023, 1, 1) # Fallback for dataset consistency

candidate_records = []
profile_texts = []

print("Extracting features and generating texts...")
with open("candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        cand_id = data["candidate_id"]
        signals = data.get("redrob_signals", {})
        
        # Structure profile text for semantic search
        experience_str = " ".join([f"{j.get('title')} at {j.get('company')}: {j.get('description')}" for j in data.get("experience", [])])
        skills_str = ", ".join(data.get("skills", []))
        full_text = f"Title: {data.get('current_title')}. Skills: {skills_str}. History: {experience_str}"
        profile_texts.append(full_text)
        
        # Calculate behavioral recency signal
        last_active = parse_date(signals.get("last_active_date", ""))
        days_since_active = (datetime(2026, 6, 22) - last_active).days # Tuned to current hackathon context year
        recency_score = max(0, 1 - (days_since_active / 365)) # 1 if active today, 0 if inactive over a year
        
        # Store metadata and numerical metrics
        candidate_records.append({
            "candidate_id": cand_id,
            "response_rate": float(signals.get("recruiter_response_rate", 0.0)),
            "recency_score": float(recency_score),
            "open_to_work": 1.0 if signals.get("open_to_work_flag", False) else 0.5,
            "raw_text": full_text # Preserved to generate the reasoning column later
        })

# 2. Compute text embeddings locally using TF-IDF
print("Encoding TF-IDF vectors (this will be very fast)...")
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import joblib

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(profile_texts)

# 3. Save artifacts as a local offline compressed cache
scipy.sparse.save_npz("candidate_tfidf.npz", tfidf_matrix)
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
df_meta = pd.DataFrame(candidate_records)
df_meta.to_parquet("candidate_metadata.parquet")
print("Pre-computation complete! Compressed artifacts saved locally.")
