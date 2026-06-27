import json
from collections import Counter

def analyze_dataset(jsonl_path):
    title_counter = Counter()
    skill_counter = Counter()
    industry_counter = Counter()
    
    print(f"Analyzing {jsonl_path}...")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            profile = cand.get("profile", {})
            
            # Titles
            title = profile.get("current_title")
            if title:
                title_counter[title.strip()] += 1
                
            # Skills
            for skill in cand.get("skills", []):
                name = skill.get("name")
                if name:
                    skill_counter[name.strip().lower()] += 1
                    
            # Industries
            for job in cand.get("career_history", []):
                ind = job.get("industry")
                if ind:
                    industry_counter[ind.strip()] += 1
                    
    print("\n--- TOP 100 TITLES ---")
    for k, v in title_counter.most_common(100):
        print(f"{v}: {k}")
        
    print("\n--- TOP 100 SKILLS ---")
    for k, v in skill_counter.most_common(100):
        print(f"{v}: {k}")
        
    print("\n--- TOP 50 INDUSTRIES ---")
    for k, v in industry_counter.most_common(50):
        print(f"{v}: {k}")

if __name__ == "__main__":
    analyze_dataset("../PS/candidates.jsonl")
