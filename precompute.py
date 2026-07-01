import json
import orjson
from datetime import datetime
import numpy as np
import pandas as pd
import time
import os

def is_honeypot(data, signals):
    """
    Returns True if the profile contains subtly impossible synthetic anomalies.
    """
    career_history = data.get("career_history", [])
    for job in career_history:
        try:
            start_year = int(str(job.get("start_date", ""))[:4])
            end_year = job.get("end_date")
            if end_year:
                end_year = int(str(end_year)[:4])
                if start_year > end_year:
                    return True
        except (ValueError, TypeError):
            continue

    skills = data.get("skills", [])
    skill_assessments = signals.get("skill_assessment_scores", {})
    expert_count = sum(1 for score in skill_assessments.values() if score > 85)
    
    if expert_count > 8 and len(career_history) == 0:
        return True

    return False

def stream_multi_line_json(filename):
    """
    Streams and extracts valid JSON objects from a pretty-printed 
    multi-line JSONL file using lazy chunking and high-performance parser.
    """
    decoder = json.JSONDecoder()
    with open(filename, "r", encoding="utf-8") as f:
        buffer = ""
        while True:
            chunk = f.read(1024 * 1024 * 10) # 10MB chunks
            if not chunk:
                if buffer:
                    buffer = buffer.strip()
                    while buffer:
                        idx = buffer.find('{')
                        if idx == -1:
                            break
                        try:
                            obj, next_idx = decoder.raw_decode(buffer, idx=idx)
                            yield obj
                            buffer = buffer[next_idx:]
                        except json.JSONDecodeError:
                            break
                break
            
            buffer += chunk
            while True:
                idx = buffer.find('{')
                if idx == -1:
                    buffer = ""
                    break
                
                try:
                    obj, next_idx = decoder.raw_decode(buffer, idx=idx)
                    yield obj
                    buffer = buffer[next_idx:]
                except json.JSONDecodeError:
                    # Incomplete object, wait for next chunk
                    buffer = buffer[idx:]
                    break

def main():
    print("Initializing TF-IDF vectorizer...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    import scipy.sparse
    import joblib
    vectorizer = TfidfVectorizer(stop_words='english')
    
    candidate_records = []
    profile_texts = []
    
    print("Streaming and parsing multi-line candidate structures...")
    count = 0
    batch_idx = 0
    
    # We will write out to a directory instead of a single file 
    # to avoid loading/saving a massive pandas dataframe
    os.makedirs("candidate_metadata.parquet", exist_ok=True)
    
    core_keywords = {"pinecone", "faiss", "milvus", "retrieval", "ranking", "llm", "llms", "nlp", "machine learning", "deep learning"}

    for data in stream_multi_line_json("candidates.jsonl"):
        if not isinstance(data, dict) or "candidate_id" not in data:
            continue
            
        cand_id = data["candidate_id"]
        profile = data.get("profile", {})
        signals = data.get("redrob_signals", {})
        
        # Honeypot filter safety pass
        if is_honeypot(data, signals):
            continue
            
        # Parse nested career history structural text
        exp_list = []
        is_pure_services = True
        career_history = data.get("career_history", [])
        if not career_history:
            is_pure_services = False
        
        for j in career_history:
            exp_list.append(f"{j.get('title', '')} at {j.get('company', '')}: {j.get('description', '')}")
            if j.get("industry", "").lower() != "it services":
                is_pure_services = False

        experience_str = " ".join(exp_list)
        
        # FIX: Extract text names from the skills array safely regardless of schema shape
        extracted_skills = []
        core_skills_count = 0
        adjacent_skills_count = 0

        for s in data.get("skills", []):
            if isinstance(s, dict):
                skill_name = s.get("name") or s.get("skill_name") or s.get("title") or str(s)
            elif isinstance(s, str):
                skill_name = s
            else:
                continue
            
            extracted_skills.append(skill_name)
            s_low = skill_name.lower()
            if any(k in s_low for k in core_keywords):
                core_skills_count += 1
            else:
                adjacent_skills_count += 1
                
        skills_str = ", ".join(extracted_skills)
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        
        # Combine fields into a clean textual block for the sentence transformer
        full_text = f"Headline: {headline}. Summary: {summary}. Skills: {skills_str}. History: {experience_str}"
        profile_texts.append(full_text[:600])
        
        # Calculate behavioral recency signal score
        try:
            last_active_str = signals.get("last_active_date", "2023-01-01")
            clean_date_str = last_active_str.split("T")[0]
            year, month, day = map(int, clean_date_str.split('-'))
            last_active = datetime(year, month, day)
            days_since_active = (datetime(2026, 6, 22) - last_active).days
            recency_score = max(0.0, 1.0 - (days_since_active / 365.0))
        except Exception:
            recency_score = 0.0
            
        candidate_records.append({
            "candidate_id": cand_id,
            "response_rate": float(signals.get("recruiter_response_rate", 0.0)),
            "recency_score": recency_score,
            "open_to_work": 1.0 if signals.get("open_to_work_flag", False) else 0.5,
            "raw_text": full_text,
            "is_pure_services": is_pure_services,
            "core_skills_count": core_skills_count,
            "adjacent_skills_count": adjacent_skills_count,
            "title": profile.get("current_title", "Unknown"),
            "years_of_experience": profile.get("years_of_experience", 0.0),
            "skills_list": extracted_skills
        })
        
        count += 1
        
        # Write columnar data in small batches to flatten memory profile
        if count % 10000 == 0:
            df_meta = pd.DataFrame(candidate_records)
            df_meta.to_parquet(f"candidate_metadata.parquet/part_{batch_idx}.parquet")
            batch_idx += 1
            candidate_records.clear()
            print(f"-> Successfully processed {count} profiles...")

    if candidate_records:
        df_meta = pd.DataFrame(candidate_records)
        df_meta.to_parquet(f"candidate_metadata.parquet/part_{batch_idx}.parquet")
        candidate_records.clear()

    print(f"\nTotal profiles loaded: {len(profile_texts)}")
    if len(profile_texts) == 0:
        print("Error: Could not extract profiles. Please verify file integrity.")
        return

    print(f"Generating TF-IDF index for valid profiles...")
    tfidf_matrix = vectorizer.fit_transform(profile_texts)
    
    # Save optimized local storage binary data assets
    scipy.sparse.save_npz("candidate_tfidf.npz", tfidf_matrix)
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    print("Pre-computation phase finished successfully! Ready to run rank.py")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")
