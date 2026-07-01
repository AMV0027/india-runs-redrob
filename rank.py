import argparse
import os
import numpy as np
import pandas as pd
import time
import scipy.sparse
import joblib
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to input candidates file")
    parser.add_argument("--out", required=True, help="Path to output submission CSV")
    args = parser.parse_args()

    if not os.path.exists("candidate_tfidf.npz") or not os.path.exists("candidate_metadata.parquet"):
        print("Error: Precomputed artifact files not found. Please run precompute.py first.")
        return

    # 1. Load the pre-computed arrays directly into system memory
    tfidf_matrix = scipy.sparse.load_npz("candidate_tfidf.npz")
    
    # Read the directory of parquet files
    df = pd.read_parquet("candidate_metadata.parquet")

    # 2. Define the exact semantic mapping requirement extracted from the target JD
    jd_criteria = (
        "Senior AI Engineer Founding Team. Deep technical depth in modern ML systems: "
        "embeddings, retrieval, ranking, LLMs, fine-tuning. Production experience with "
        "embeddings-based retrieval systems and vector databases like FAISS, Pinecone, or Milvus. "
        "Strong Python. Designing evaluation frameworks like NDCG, MRR, MAP. Product builder."
    )

    # 3. Stage 1: Sparse matrix recall
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    jd_embedding = vectorizer.transform([jd_criteria])

    # Perform vector inner-product multiplication for rapid Cosine Similarity scoring
    df["sparse_score"] = tfidf_matrix.dot(jd_embedding.T).toarray().flatten()

    # Fast filter to top 300 to bound O(n) complexity
    df_top300 = df.nlargest(300, "sparse_score").copy()

    # 4. Stage 2: Dense semantic re-ranker
    print("Loading SentenceTransformer model for Stage 2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    jd_dense = model.encode([jd_criteria])[0]
    
    print("Encoding candidate texts...")
    candidate_texts = df_top300["raw_text"].tolist()
    candidate_dense = model.encode(candidate_texts)
    
    # Calculate cosine similarity for dense embeddings
    jd_dense_norm = jd_dense / norm(jd_dense)
    candidate_dense_norm = candidate_dense / norm(candidate_dense, axis=1, keepdims=True)
    dense_scores = np.dot(candidate_dense_norm, jd_dense_norm)
    
    df_top300["semantic_score"] = dense_scores

    # 5. Composite ranking formula incorporating semantic compatibility and behavioral engagement
    df_top300["final_score"] = (
        (df_top300["semantic_score"] * 0.65) + 
        (df_top300["response_rate"] * 0.15) + 
        (df_top300["recency_score"] * 0.10) + 
        (df_top300["open_to_work"] * 0.10)
    )

    # Apply Business Logic Functional Multipliers
    is_pure_services = df_top300["is_pure_services"] == True
    df_top300.loc[is_pure_services, "final_score"] *= 0.5
    
    penalty_mask = df_top300["adjacent_skills_count"] > (df_top300["core_skills_count"] * 2)
    df_top300.loc[penalty_mask, "final_score"] *= 0.8

    # 6. Sort by final score descending, breaking ties deterministically via candidate_id ascending
    df_sorted = df_top300.sort_values(by=["final_score", "candidate_id"], ascending=[False, True]).head(100).copy()
    df_sorted["rank"] = range(1, 101)

    # 7. Dynamic, Evidence-Based Output Generation
    reasons = []
    
    for _, row in df_sorted.iterrows():
        skills = row.get("skills_list", [])
        if isinstance(skills, np.ndarray):
            skills = skills.tolist()
            
        core_matches = [s for s in skills if s.lower() in ["pinecone", "faiss", "milvus", "retrieval", "ranking"]]
        title = row.get("title", "Unknown")
        yoe = row.get("years_of_experience", 0.0)
        response_rate = row.get("response_rate", 0.0)
        
        if row.get("is_pure_services"):
            reason = f"Ranked lower despite {yoe} yrs as {title}; background entirely in consulting, conflicting with product-builder mandate."
        elif core_matches:
            reason = f"Strong fit: {yoe} yrs as {title}. Demonstrates production experience with {', '.join(core_matches[:2])}. Highly responsive ({int(response_rate*100)}%)."
        else:
            reason = f"{title} with {yoe} yrs. Shows ML competency but lacks direct vector database/retrieval infrastructure experience."
            
        reasons.append(reason)

    df_sorted["reasoning"] = reasons

    # 8. Export the structured file ensuring compliance with column headers and order constraints
    output_df = df_sorted[["candidate_id", "rank", "final_score", "reasoning"]].rename(columns={"final_score": "score"})
    output_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Success! Output csv exported with exactly {len(output_df)} ranked rows to {args.out}")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")
