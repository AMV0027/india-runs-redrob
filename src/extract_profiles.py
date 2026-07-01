import os
import csv
import json

def extract_top_100_profiles(submission_csv, candidates_jsonl, output_json, output_csv):
    top_100_ids = []
    submission_rows = {}
    with open(submission_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                cid = row[0]
                top_100_ids.append(cid)
                submission_rows[cid] = {"rank": row[1], "score": row[2], "reasoning": row[3]}

    top_100_set = set(top_100_ids)
    extracted_profiles = {}

    with open(candidates_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            candidate = json.loads(line)
            cid = candidate.get("candidate_id")
            if cid in top_100_set:
                extracted_profiles[cid] = candidate

    ordered_profiles = []
    for cid in top_100_ids:
        if cid in extracted_profiles:
            ordered_profiles.append(extracted_profiles[cid])

    # Write JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(ordered_profiles, f, indent=2, ensure_ascii=False)
    print(f"Written {len(ordered_profiles)} profiles to {output_json}")

    # Write CSV summary (always in sync with submission.csv)
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate_id", "rank", "score", "name", "current_title", "current_company",
            "years_of_experience", "location", "country", "recruiter_response_rate",
            "last_active_date", "notice_period_days", "willing_to_relocate",
            "github_activity_score", "skills"
        ])
        for cid in top_100_ids:
            cand = extracted_profiles.get(cid)
            if not cand:
                continue
            profile = cand.get("profile", {})
            signals = cand.get("redrob_signals", {})
            skills  = ", ".join(s.get("name", "") for s in cand.get("skills", []) if s.get("name"))
            sub     = submission_rows.get(cid, {})
            writer.writerow([
                cid,
                sub.get("rank", ""),
                sub.get("score", ""),
                profile.get("anonymized_name", ""),
                profile.get("current_title", ""),
                profile.get("current_company", ""),
                profile.get("years_of_experience", ""),
                profile.get("location", ""),
                profile.get("country", ""),
                signals.get("recruiter_response_rate", ""),
                signals.get("last_active_date", ""),
                signals.get("notice_period_days", ""),
                signals.get("willing_to_relocate", ""),
                signals.get("github_activity_score", ""),
                skills
            ])
    print(f"Written {len(ordered_profiles)} profiles to {output_csv}")

if __name__ == "__main__":
    extract_top_100_profiles(
        "submission.csv",
        "challange_dataset/candidates.jsonl",
        "data/top_100_profiles.json",
        "data/top_100_profiles.csv"
    )
