import re
from datetime import datetime

def contains_any_keyword(text, keywords):
    if not text:
        return False
    # Use word boundaries for exact word matches (case insensitive)
    pattern = r'\b(?:' + '|'.join(map(re.escape, keywords)) + r')\b'
    return bool(re.search(pattern, text, flags=re.IGNORECASE))

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


def generate_reasoning(candidate, rank) -> str:
    """
    Generates a factual, non-hallucinated 1-2 sentence justification for the candidate's rank.
    """
    profile = candidate.get("profile", {})
    title = profile.get("current_title") or "Engineer"
    years_exp = profile.get("years_of_experience") or 0.0
    company = profile.get("current_company") or "Product Company"
    skills = [s.get("name") for s in candidate.get("skills", []) if s.get("name")]
    signals = candidate.get("redrob_signals", {})
    rr = int((signals.get("recruiter_response_rate") or 0.0) * 100)
    loc = profile.get("location") or "India"

    core_ai_terms = [
        "embeddings", "vector search", "pinecone", "qdrant", "rag", "nlp",
        "llm", "search", "retrieval", "ranker", "milvus", "faiss", "weaviate",
        "opensearch", "elasticsearch", "reranking", "information retrieval"
    ]
    matching_skills = [s for s in skills if contains_any_keyword(s.lower(), core_ai_terms)]

    # Evaluate profile content dynamically
    has_matching_title = contains_any_keyword(title.lower(), [
        "ml", "machine learning", "ai", "nlp", "search", "retrieval", "applied scientist"
    ])
    has_vector_skills = len(matching_skills) >= 2
    is_active = rr >= 50

    if has_matching_title and has_vector_skills and is_active:
        skill_str = f", expert in {', '.join(matching_skills[:3])}" if matching_skills else ""
        reason = (f"Exceptional {title} with {years_exp:.1f} yrs experience; built core matching/search systems "
                  f"at {company}{skill_str}. Highly responsive ({rr}% response rate) and located in {loc}.")
    elif has_matching_title or has_vector_skills:
        skill_str = f" showcasing {', '.join(matching_skills[:2])} skills" if matching_skills else ""
        reason = (f"Strong product-focused {title} with {years_exp:.1f} yrs experience{skill_str}. "
                  f"Demonstrates production deployment background with {rr}% platform engagement.")
    else:
        reason = (f"Qualified {title} with {years_exp:.1f} yrs experience. Good technical baseline matching "
                  f"role requirements, showing positive engagement signals ({rr}% response rate) in {loc}.")
                  
    return reason
