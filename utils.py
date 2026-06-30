import re
from datetime import datetime

def contains_any_keyword(text: str, keywords) -> bool:
    """Returns True if any keyword from the list appears as a whole word in text."""
    if not text:
        return False
    pattern = r'\b(?:' + '|'.join(map(re.escape, keywords)) + r')\b'
    return bool(re.search(pattern, text, flags=re.IGNORECASE))

# ─── HIERARCHICAL CAPABILITY GROUPS ──────────────────────────────────────────
CAPABILITY_GROUPS = {
    "Vector Retrieval": [
        "pinecone", "qdrant", "milvus", "faiss", "weaviate", "hnsw", "ivf", "ann search", "vector search", "vector database", "dense retrieval", "dense representations"
    ],
    "Search Infrastructure": [
        "elasticsearch", "opensearch", "solr", "lucene", "hybrid search", "bm25", "tf-idf", "inverted index", "search backend", "search service"
    ],
    "Recommendation Systems": [
        "recommendation systems", "recommender", "collaborative filtering", "matrix factorization", "personalization", "learning to rank", "ltr", "reranking", "reciprocal rank fusion", "rrf"
    ],
    "Production ML": [
        "production ml", "fine tuning", "quantization", "docker", "triton", "onnx", "tensorrt", "latency profiling", "model serving", "kubernetes", "k8s", "aws", "gcp", "ci/cd"
    ],
    "LLM Engineering": [
        "llm", "llms", "lora", "qlora", "peft", "langchain", "llama", "gpt", "rag", "retrieval augmented generation", "prompt engineering"
    ],
    "Evaluation Metrics": [
        "ndcg", "mrr", "map", "bleu", "rouge", "benchmarking", "offline evaluation", "online evaluation", "ab testing", "a/b testing"
    ]
}

def parse_job_description(jd_text: str) -> dict:
    """
    Dynamically extracts requirements and priority areas from the JD.
    """
    jd_lower = jd_text.lower()
    
    # Seniority experience target
    exp_match = re.search(r'(\d+)\+?\s*years?', jd_lower)
    required_experience = float(exp_match.group(1)) if exp_match else 5.0
    
    # Must-have capabilities mapped to capability groups
    must_haves = []
    if contains_any_keyword(jd_lower, ["vector", "pinecone", "qdrant", "weaviate", "milvus", "faiss"]):
        must_haves.append("Vector Retrieval")
    if contains_any_keyword(jd_lower, ["search", "elasticsearch", "retrieval", "hybrid"]):
        must_haves.append("Search Infrastructure")
    if contains_any_keyword(jd_lower, ["evaluation", "ndcg", "mrr", "benchmark"]):
        must_haves.append("Evaluation Metrics")
        
    # Determine the priority domain
    priority_domain = "Search"
    if "recommend" in jd_lower:
        priority_domain = "Recommendation"
    elif "llm" in jd_lower or "generative" in jd_lower or "prompt" in jd_lower:
        priority_domain = "LLM"
        
    return {
        "required_experience": required_experience,
        "must_haves": must_haves,
        "priority_domain": priority_domain
    }

# Blacklist of consulting/IT services companies + fictional/trap companies (name-based matching)
COMPANY_BLACKLIST = {
    # Real consulting/IT services firms
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "l&t", "lnt",
    "mindtree", "mphasis", "ntt data", "ust global", "ey", "pwc", "deloitte", "kpmg",
    # Fictional/honeypot trap companies (TV shows, movies, comics)
    "wayne enterprises", "wayne corp",           # Batman
    "stark industries", "stark corp",            # Iron Man
    "dunder mifflin",                            # The Office
    "globex", "globex inc",                      # The Simpsons
    "pied piper",                                # Silicon Valley
    "hooli",                                     # Silicon Valley
    "acme corp", "acme corporation",             # Looney Tunes / generic trap
    "initech",                                   # Office Space
    "umbrella corporation", "umbrella corp",     # Resident Evil
    "cyberdyne systems", "cyberdyne",            # Terminator
    "weyland-yutani",                            # Alien
    "nakatomi",                                  # Die Hard
    "oscorp",                                    # Spider-Man
    "massive dynamic",                           # Fringe
    "buy n large",                               # WALL-E
    "soylent",                                   # Soylent Green
}

# Industries that indicate outsourcing/consulting work
CONSULTING_INDUSTRIES = {
    "it services", "it consulting", "consulting", "outsourcing",
    "managed services", "staffing", "bpo", "kpo"
}

# Industries that indicate a product/tech company — positive signal
PRODUCT_INDUSTRIES = {
    "software", "internet", "fintech", "e-commerce", "healthtech", 
    "healthtech ai", "edtech", "saas", "ai/ml", "artificial intelligence",
    "conversational ai", "ai services", "voice ai",
    "cybersecurity", "gaming", "adtech", "food delivery", "transportation"
}

# Non-relevant/disqualified domains (CV, Speech, Robotics, Hardware)
CV_SPEECH_ROBOTICS_KEYWORDS = {
    "opencv", "yolo", "yolov5", "yolov8", "yolov9", "yolov10", "computer vision",
    "image classification", "object detection", "image segmentation", "tts", "asr",
    "speech recognition", "text-to-speech", "speech-to-text", "audio processing",
    "robotics", "ros", "lidar", "radar", "slam", "cuda", "hardware", "iot", "sensor",
    "microcontroller", "embedded", "autonomous driving", "self-driving"
}

# Core NLP, IR, and Vector search terms (Positive domains)
NLP_IR_SEARCH_KEYWORDS = {
    "vector search", "vector database", "pinecone", "qdrant", "milvus", "faiss", "weaviate",
    "elasticsearch", "opensearch", "information retrieval", "hybrid search", "retrieval",
    "rerank", "ndcg", "mrr", "nlp", "llm", "llms", "lora", "qlora", "peft", "fine tuning",
    "sentence transformers", "hugging face transformers", "langchain", "embeddings", 
    "semantic search", "recommendation systems", "rag"
}

# Forbidden/Non-technical skills that indicate out-of-domain profiles (exact match list)
FORBIDDEN_SKILLS = {
    "accounting", "bookkeeping", "finance", "audit", "marketing", "digital marketing",
    "seo", "sales", "inside sales", "business development", "content writing",
    "copywriting", "creative writing", "human resources", "talent acquisition",
    "recruitment", "payroll", "graphic design", "photoshop", "illustrator",
    "civil engineering", "structural engineering", "mechanical engineering",
    "electrical engineering", "project management", "scrum master", "hr"
}

# Vector database specific terms for gatekeeping
VECTOR_DB_KEYWORDS = {
    "pinecone", "qdrant", "milvus", "faiss", "weaviate", "vector search", "vector database"
}

# Hard-disqualify titles — candidates with these current titles are eliminated
# regardless of any other signals. They are structurally irrelevant to the JD.
HARD_DISQUALIFY_TITLES = {
    "graphic designer", "operations manager", "civil engineer",
    "customer support", "content writer", "project manager",
    "business analyst", "mechanical engineer", "electrical engineer",
    "hr manager", "human resources", "accountant", "finance manager",
    "marketing manager", "sales manager", "recruiter", "talent acquisition"
}


def check_forbidden_skills(skill_names) -> bool:
    """
    Checks if a list of skills contains any forbidden/non-technical keywords.
    Uses exact token-based matching to prevent false positive substring matches
    (e.g., 'Salesforce CRM' should not trigger the 'sales' penalty).
    """
    for s in skill_names:
        s_lower = s.lower()
        if s_lower in FORBIDDEN_SKILLS:
            return True
        # Split into words to avoid substring matching issues
        tokens = re.findall(r'[a-z]+', s_lower)
        for t in tokens:
            if t in {"accounting", "bookkeeping", "finance", "audit", "marketing", "sales", "copywriting", "recruitment", "payroll", "photoshop", "illustrator", "hr"}:
                return True
    return False


def is_title_experience_mismatch(title: str, years_exp: float) -> bool:
    """
    Checks for structural mismatches between stated title seniority and experience.
    """
    title_lower = title.lower() if title else ""
    
    # Junior/Associate title but high experience
    if contains_any_keyword(title_lower, ["junior", "jr", "associate", "intern", "trainee"]) and years_exp > 4.5:
        if not contains_any_keyword(title_lower, ["professor", "director"]):
            return True
            
    # Senior / Lead title but very low experience
    if contains_any_keyword(title_lower, ["senior", "sr", "lead", "staff", "principal", "manager", "head", "director"]) and years_exp < 2.5:
        return True
        
    return False


def is_honeypot(candidate) -> bool:
    """
    Checks if a candidate is a honeypot based on logical timeline contradictions.
    """
    profile = candidate.get("profile", {})
    years_exp = profile.get("years_of_experience", 0.0)
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])

    if not career_history:
        return False

    # 1. Total months computed vs years_of_experience
    total_months_limit = (years_exp * 12) + 12
    computed_months = 0
    first_start_date = None
    last_end_date = None

    for job in career_history:
        dur = job.get("duration_months", 0)
        computed_months += dur

        start_str = job.get("start_date")
        end_str = job.get("end_date")

        if start_str:
            try:
                start_dt = datetime.strptime(start_str, "%Y-%m-%d")
                if first_start_date is None or start_dt < first_start_date:
                    first_start_date = start_dt
            except ValueError:
                pass

        if end_str:
            try:
                end_dt = datetime.strptime(end_str, "%Y-%m-%d")
                if last_end_date is None or end_dt > last_end_date:
                    last_end_date = end_dt
            except ValueError:
                pass

    if computed_months > total_months_limit:
        return True

    # 2. Check overall career span vs years_of_experience
    if first_start_date:
        ref_end = last_end_date if last_end_date else datetime(2026, 6, 26)
        span_years = (ref_end - first_start_date).days / 365.25
        if span_years > (years_exp + 2.0):
            return True

    # 3. Check individual skill durations vs years_of_experience
    for skill in skills:
        skill_months = skill.get("duration_months", 0)
        if skill_months > (years_exp * 12 + 6):
            return True

    # 4. Check for impossible skill proficiencies
    expert_skills_count = sum(1 for s in skills if s.get("proficiency") in ["expert", "advanced"])
    total_skill_duration = sum(s.get("duration_months", 0) for s in skills)
    if expert_skills_count >= 8 and total_skill_duration < 12:
        return True

    # 5. Education graduation year vs years_of_experience
    REFERENCE_YEAR = 2026
    if education:
        latest_grad_year = None
        for edu in education:
            end_yr = edu.get("end_year")
            if end_yr and isinstance(end_yr, int):
                if latest_grad_year is None or end_yr > latest_grad_year:
                    latest_grad_year = end_yr

        if latest_grad_year:
            max_possible_exp_years = (REFERENCE_YEAR - latest_grad_year) + 1
            if years_exp > max_possible_exp_years + 2:
                return True

    return False


def is_blacklisted(candidate) -> bool:
    """
    Checks if a candidate has spent their entire career at blacklisted IT services firms or industries.
    """
    career = candidate.get("career_history", [])
    if not career:
        return True

    blacklisted_count = 0
    for job in career:
        comp = (job.get("company") or "").lower()
        industry = (job.get("industry") or "").lower()

        name_blacklisted = contains_any_keyword(comp, COMPANY_BLACKLIST)
        industry_blacklisted = contains_any_keyword(industry, CONSULTING_INDUSTRIES)

        if name_blacklisted or industry_blacklisted:
            blacklisted_count += 1
            
    return blacklisted_count == len(career)


def extract_candidate_features(candidate: dict, parsed_jd: dict = None) -> dict:
    """
    Unified candidate feature extraction (Layer 2 & Layer 3).
    Extracts numerical capability strengths (0-5) and checks disqualifiers.
    """
    profile = candidate.get("profile", {})
    years_exp = profile.get("years_of_experience") or 0.0
    skills_objs = candidate.get("skills", [])
    skills = [(s.get("name") or "").lower() for s in skills_objs if s.get("name")]
    career_history = candidate.get("career_history", [])
    
    # Compute capability strengths (0-5) weighted by skill endorsements
    strengths = {}
    evidence = {}
    # Build endorsement lookup: skill_name_lower -> endorsements
    endorsement_map = {
        (s.get("name") or "").lower(): s.get("endorsements", 0)
        for s in skills_objs if s.get("name")
    }
    for group, keywords in CAPABILITY_GROUPS.items():
        # Match skills and accumulate endorsement-weighted score
        skill_matches = []
        skill_score = 0.0
        for s in skills:
            if contains_any_keyword(s, keywords):
                skill_matches.append(s)
                endorsements = endorsement_map.get(s, 0)
                # Endorsements scale the contribution: 0 = 1.0x, 10+ = 1.5x, 50+ = 2.0x
                if endorsements >= 50:   endorse_weight = 2.0
                elif endorsements >= 20: endorse_weight = 1.5
                elif endorsements >= 5:  endorse_weight = 1.2
                else:                    endorse_weight = 1.0
                skill_score += endorse_weight

        # Match count in job descriptions (career verified = stronger signal)
        job_matches = 0
        job_matched_terms = []
        for job in career_history:
            job_text = f"{job.get('title') or ''} {job.get('description') or ''}".lower()
            found = [kw for kw in keywords if kw in job_text]
            if found:
                job_matches += 1
                job_matched_terms.extend(found)
                
        # Raw score: endorsement-weighted skill score + career proof bonus
        raw_score = skill_score + (2 * job_matches)
        
        # Scale to 0-5
        strength = 0
        if raw_score >= 8: strength = 5
        elif raw_score >= 5: strength = 4
        elif raw_score >= 3: strength = 3
        elif raw_score >= 1: strength = 2
        elif raw_score > 0: strength = 1
        
        strengths[group] = strength
        evidence[group] = list(set(skill_matches + job_matched_terms))[:3]
        
    # CV/Speech domination check
    cv_speech_match_count = sum(1 for s in skills if contains_any_keyword(s, CV_SPEECH_ROBOTICS_KEYWORDS))
    is_cv_dominated = cv_speech_match_count >= 3
    
    ats_score = calculate_ats_score(candidate)
    
    return {
        "strengths": strengths,
        "evidence": evidence,
        "years_exp": years_exp,
        "ats_score": ats_score,
        "is_cv_dominated": is_cv_dominated,
        "is_honeypot": is_honeypot(candidate),
        "is_blacklisted": is_blacklisted(candidate)
    }


def generate_reasoning(candidate: dict, rank: int, features: dict = None) -> str:
    """
    Generates a factual, non-hallucinated 1-2 sentence justification for the candidate's rank.
    """
    profile = candidate.get("profile", {})
    title = profile.get("current_title") or "Engineer"
    years_exp = profile.get("years_of_experience") or 0.0
    company = profile.get("current_company") or "Product Company"
    signals = candidate.get("redrob_signals", {})
    rr = int((signals.get("recruiter_response_rate") or 0.0) * 100)
    loc = profile.get("location") or "India"
    
    if features is None:
        features = extract_candidate_features(candidate)
        
    strengths = features.get("strengths", {})
    # Find strongest capability group
    sorted_caps = sorted(strengths.items(), key=lambda x: -x[1])
    strongest_cap = sorted_caps[0][0] if sorted_caps else "Applied ML"
    cap_evidence = ", ".join(features.get("evidence", {}).get(strongest_cap, []))
    
    evidence_str = f" (evidence: {cap_evidence})" if cap_evidence else ""
    reason = (f"Candidate has {years_exp:.1f} yrs experience as a {title} at {company}. "
              f"Demonstrates top capability in {strongest_cap}{evidence_str}. "
              f"Highly responsive ({rr}% response rate) and located in {loc}.")
    return reason


def calculate_ats_score(candidate) -> float:
    """
    Computes an ATS resume-integrity score [0.0, 1.0] representing:
    1. Skill Coverage (40%)
    2. Job Stability / average tenure (30%)
    3. Gap Penalties (15%)
    4. Seniority / Career Growth progression (15%)
    """
    profile = candidate.get("profile", {})
    skills_objs = candidate.get("skills", [])
    skills = [(s.get("name") or "").lower() for s in skills_objs if s.get("name")]
    career_history = candidate.get("career_history", [])
    
    # 1. Skill Coverage Ratio
    target_skills = [
        "vector search", "vector database", "pinecone", "qdrant", "milvus", "faiss", "weaviate",
        "elasticsearch", "opensearch", "information retrieval", "hybrid search", "retrieval",
        "rerank", "ndcg", "nlp", "llm", "langchain", "embeddings"
    ]
    matched_skills = sum(1 for ts in target_skills if any(ts in s for s in skills))
    skill_coverage = min(matched_skills / 6.0, 1.0) # Cap at 6 matches for full score
    
    # 2. Stability and Tenure
    stability_score = 1.0
    if career_history:
        tenures = [job.get("duration_months") or 0 for job in career_history]
        non_zero_tenures = [t for t in tenures if t > 0]
        if non_zero_tenures:
            avg_tenure_months = sum(non_zero_tenures) / len(non_zero_tenures)
            if avg_tenure_months < 15: # Less than 1.2 years on average
                stability_score = 0.5 # Job hopping indicator
            elif avg_tenure_months > 36: # More than 3 years average
                stability_score = 1.2 # Loyalty bonus
                
    # 3. Gap Penalties
    gap_score = 1.0
    if len(career_history) > 1:
        for idx in range(len(career_history) - 1):
            curr_job = career_history[idx]
            prev_job = career_history[idx + 1]
            curr_start = curr_job.get("start_date")
            prev_end = prev_job.get("end_date")
            if curr_start and prev_end:
                try:
                    c_start = datetime.strptime(curr_start, "%Y-%m-%d")
                    p_end = datetime.strptime(prev_end, "%Y-%m-%d")
                    gap_months = (c_start - p_end).days / 30.436875
                    if gap_months > 12.0:
                        gap_score = 0.6 # Severe gap penalty
                        break
                except ValueError:
                    pass
                    
    # 4. Career Progression
    progression_score = 1.0
    if len(career_history) > 1:
        oldest_job = career_history[-1]
        newest_job = career_history[0]
        old_title = (oldest_job.get("title") or "").lower()
        new_title = (newest_job.get("title") or "").lower()
        
        old_is_junior = contains_any_keyword(old_title, ["junior", "jr", "associate", "intern", "trainee", "entry"])
        new_is_senior = contains_any_keyword(new_title, ["senior", "sr", "lead", "staff", "principal", "founding", "director", "head"])
        
        if old_is_junior and new_is_senior:
            progression_score = 1.25 # Career growth bonus!
            
    # Combine weights
    ats_score = (0.40 * skill_coverage) + (0.30 * stability_score) + (0.15 * gap_score) + (0.15 * progression_score)
    return min(max(ats_score, 0.0), 1.0)

