# Hackathon Understanding & Problem Interpretation Document

## Overview

The objective of this hackathon is to design and develop an intelligent AI-powered candidate ranking system capable of identifying the most suitable candidates for a given job role. Unlike traditional Applicant Tracking Systems (ATS) that primarily rely on keyword matching, the proposed solution should evaluate candidates using contextual understanding, semantic relevance, experience alignment, skill applicability, and explainable reasoning.

The challenge focuses on building a recruitment intelligence platform that can accurately rank candidates based on their suitability for a specific job description while providing transparent explanations for every ranking decision.

---

# Problem Statement Interpretation

Modern recruitment systems often fail to identify high-quality candidates because they depend heavily on exact keyword matching.

For example:

### Job Description Requirement

- Large Language Models (LLMs)
- NLP
- AI Application Development

### Candidate Resume

- Built Retrieval-Augmented Generation (RAG) systems
- Developed LangChain applications
- Fine-tuned Llama models

Although the candidate possesses highly relevant experience, a traditional keyword-based ATS may fail to identify them because the exact terms from the job description are not present.

The goal of this hackathon is to overcome such limitations by building an intelligent ranking system that understands semantic relationships between candidate profiles and job requirements.

---

# Challenge Objective

Build an AI-powered system that:

1. Accepts a job description as input.
2. Processes a large dataset of candidate profiles.
3. Evaluates candidate suitability using intelligent matching techniques.
4. Produces a ranked list of candidates.
5. Provides explainable reasoning behind ranking decisions.
6. Demonstrates transparency and reproducibility in its methodology.

---

# Inputs Provided by Organizers

## Candidate Dataset

Participants will receive a dataset containing candidate information.

Potential attributes may include:

- Candidate ID
- Skills
- Experience
- Education
- Certifications
- Projects
- Work History
- Technical Expertise
- Additional Metadata

Example:

```json
{
  "candidate_id": 101,
  "skills": ["Python", "Machine Learning", "AWS"],
  "experience_years": 5,
  "education": "B.Tech",
  "projects": ["Fraud Detection System", "Customer Recommendation Engine"]
}
```

---

## Job Description

A target role description specifying:

- Required Skills
- Preferred Skills
- Experience Requirements
- Domain Knowledge
- Educational Requirements
- Responsibilities

Example:

```text
Senior AI Engineer

Requirements:
- Python
- NLP
- LLM Development
- AWS
- 5+ Years Experience
```

---

# Expected Output

The final system should generate a ranked candidate list.

Example:

| Rank | Candidate ID | Score |
| ---- | ------------ | ----- |
| 1    | 451          | 96.5  |
| 2    | 122          | 94.8  |
| 3    | 781          | 92.3  |

The ranking should reflect overall candidate suitability for the provided role.

---

# Evaluation Criteria

## 1. Ranking Quality

The primary objective is to identify the most relevant candidates.

The ranking should consider:

- Skill relevance
- Experience alignment
- Project relevance
- Domain expertise
- Semantic similarity
- Overall suitability

The system should consistently rank stronger candidates above weaker candidates.

---

## 2. Methodology Clarity

Participants must clearly explain:

- Data preprocessing methods
- Feature extraction techniques
- Ranking logic
- Scoring strategy
- AI models used
- Design decisions

Judges should be able to understand how rankings are produced.

---

## 3. Explainability

Every ranking decision should be understandable.

Example:

### Candidate Score: 94.5

Strengths:

- Strong NLP experience
- Relevant LLM project portfolio
- Exceeds required experience

Gaps:

- Missing cloud certification

The system should provide meaningful reasoning rather than opaque scores.

---

# Key Expectations from Organizers

The challenge is not about building another keyword search engine.

Instead, organizers are looking for:

### Semantic Understanding

The system should recognize related concepts.

Example:

```text
Job Description:
Large Language Models

Candidate:
Built RAG applications using LangChain and Llama
```

The system should identify this as highly relevant experience.

---

### Contextual Matching

Candidates should be evaluated based on overall profile quality rather than isolated keywords.

---

### Intelligent Ranking

The ranking process should combine multiple dimensions:

- Skills
- Experience
- Projects
- Education
- Certifications
- Semantic Relevance

---

### Explainable AI

The solution should provide transparent reasoning behind rankings.

---

# Recommended Technical Direction

A strong solution would typically include:

## Stage 1: Data Processing

- Clean candidate data
- Normalize skill names
- Standardize text fields
- Handle missing values

---

## Stage 2: Information Extraction

Extract:

- Skills
- Experience
- Education
- Certifications
- Project Information

---

## Stage 3: Semantic Matching

Convert job descriptions and candidate profiles into embeddings.

Possible models:

- BGE
- E5
- Jina Embeddings
- Sentence Transformers

Measure similarity using:

- Cosine Similarity
- Vector Search

---

## Stage 4: Feature-Based Scoring

Generate scores for:

- Skill Match
- Experience Match
- Education Match
- Project Relevance
- Certification Relevance

---

## Stage 5: Ranking Engine

Combine all scores into a final ranking score.

Example:

```text
Final Score =
40% Skill Match
25% Experience Match
15% Project Relevance
10% Education Match
10% Semantic Similarity
```

---

## Stage 6: Explanation Generation

Generate human-readable reasoning.

Example:

```json
{
  "candidate_id": 451,
  "score": 96.2,
  "strengths": [
    "Strong NLP expertise",
    "Relevant LLM projects",
    "7 years experience"
  ],
  "gaps": ["No cloud certification"]
}
```

---

# Submission Requirements

## GitHub Repository

Must contain:

- Source Code
- Documentation
- Installation Instructions
- Dependencies
- Execution Workflow

---

## Ranked Output File

A structured file containing:

- Candidate ID
- Rank
- Score

As specified by organizers.

---

## Methodology Document

Should explain:

- Problem Understanding
- Solution Architecture
- AI Models Used
- Ranking Methodology
- Explainability Strategy
- Future Improvements

---

# Success Criteria

A successful submission should demonstrate:

1. Accurate candidate ranking.
2. Semantic understanding beyond keywords.
3. Explainable and transparent decision-making.
4. Reproducible methodology.
5. Scalable architecture capable of handling large candidate datasets.

The ideal solution behaves like an intelligent recruiter that understands both job requirements and candidate capabilities, rather than a simple keyword-matching ATS.
