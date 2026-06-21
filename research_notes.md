# Research Notes: State-of-the-Art Candidate Ranking & Semantic ATS (2024-2025)

Based on a review of recent literature, open-source repositories, and modern Applicant Tracking System (ATS) architectures, here are the key trends and best practices for building an intelligent candidate ranking system.

## 1. Hybrid Search Architecture is the Gold Standard
Modern AI-ATS systems (like Milvus/Google Vertex AI Search patterns) have moved away from pure semantic search due to precision issues with domain-specific terms (e.g., matching "Python" to "Java" because they are both programming languages). The standard is now a **Hybrid Retrieval** pipeline:
- **Lexical/Sparse Search (BM25)**: Handles exact keyword matching for specific acronyms, tools, and IDs.
- **Semantic/Dense Search (Vector Embeddings)**: Captures intent and context using models like `Sentence-BERT` or `BGE`.
- **Rank Fusion**: Uses algorithms like Reciprocal Rank Fusion (RRF) to combine sparse and dense results.

## 2. Advanced Multi-Stage Reranking
A "Retrieve & Rerank" pattern is standard:
- **Stage 1 (Retrieval)**: Fast ANN (Approximate Nearest Neighbor) vector search combined with BM25 to get a candidate pool.
- **Stage 2 (Cross-Encoder / ML Reranking)**: The top `K` candidates are reranked using a more computationally expensive model (like a HuggingFace Cross-Encoder) that takes both the JD and the candidate profile simultaneously, OR a feature-based ML model (XGBoost) that scores structured entities.

## 3. Structured Entity Extraction & Graph Neural Networks
State-of-the-art parsers don't just embed raw text. They extract a **knowledge graph** or structured entities:
- **Ontology Mapping**: Systems use HR ontologies to map related skills (e.g., mapping `React` to `Frontend`).
- **Structured Scoring**: Skills, years of experience, and education tiers are extracted and scored independently against weighted criteria in the job description.

## 4. Notable Open-Source Implementations
- **[reqcore-inc/reqcore]**: A modern open-source ATS focusing on "transparent AI ranking" and pipeline management.
- **[vectornguyen76/resume-ranking] & [interviewstreet/hiring-agent]**: Leverage local LLMs (like Ollama/Llama 3) offline to parse resumes into markdown/JSON, augment with external data (GitHub), and generate objective evaluation scores.
- **[srbhr/Resume-Matcher]**: Focuses on tuning resumes to JDs using vector search and NLP-based skill extraction.

## 5. Explainable AI (XAI) and Bias Mitigation
Due to the EU AI Act and bias concerns, modern systems must be fully explainable. "Black box" neural network scores are heavily penalized. 
- **Feature Attribution**: The system must explicitly state the percentage contribution of skills, experience, and behavioral metrics.
- **Human-in-the-Loop**: The AI provides a detailed report rather than just a final score, allowing recruiters to verify the logic.

---

## How this applies to our Hackathon Strategy
Our updated `implementation_plan.md` is actually very well-aligned with these SOTA practices:
1. **We are using Hybrid Retrieval** (FAISS + BM25) to solve the exact-match vs semantic-match problem.
2. **We are using Structured Scoring** (Extracting skills/tenure offline and scoring them independently).
3. **We are using an Explainable Additive Formula** to guarantee XAI transparency for the judges.

**Potential Upgrades based on Research:**
- **Cross-Encoder Reranking**: We could add a lightweight cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) in `rank.py` to re-score the top 2000 candidates. It's more accurate than cosine similarity but might be tight within the 5-minute CPU constraint.
- **Reciprocal Rank Fusion (RRF)**: We can use RRF to cleanly merge the FAISS and BM25 results before applying our business logic multipliers.

Would you like to incorporate Cross-Encoders or Reciprocal Rank Fusion into the plan, or stick with our current weighted hybrid approach?
