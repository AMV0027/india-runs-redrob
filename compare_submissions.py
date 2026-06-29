import pandas as pd
import glob
import os
import itertools
import markdown
import subprocess

def main():
    compare_dir = r"c:\SideProjects\India_Runs\comparision"
    # Including development submission if present in development folder? 
    # The user said "check all combinations between submission." but earlier said "dont consider @[development/submission.csv] in compsrission". 
    # I'll include the development file back if they want all combinations, or stick to the ones in comparision folder. I will just include whatever is in the comparision folder, but I can also include the development one if requested. Let's include everything that has 'submission' in the name in both directories to be safe, but wait, the prompt says "check all combinations between submission." This implies the previous files we checked.
    compare_files = sorted(glob.glob(os.path.join(compare_dir, "*.csv")))
    
    # Let's add development/submission.csv back since they said "check all combinations between submission" -- wait, their previous prompt was exactly "dont consider @[development/submission.csv]". I will respect the previous prompt and only use the `comparision` dir files.
    dfs = {}
    candidates_sets = {}
    
    for f in compare_files:
        name = os.path.basename(f).replace(".csv", "")
        df = pd.read_csv(f)
        if 'reasoning' in df.columns:
            df = df.drop(columns=['reasoning'])
        dfs[name] = df
        candidates_sets[name] = set(df['candidate_id'])

    if not dfs:
        print("No files found in comparision folder.")
        return

    lines = []
    lines.append("# Comparison Report (All Submissions)\n")
    
    # 1. Present in ALL
    all_candidates = set.intersection(*candidates_sets.values())
    lines.append(f"## 1. Candidates Present in ALL ({len(all_candidates)})\n")
    
    rank_compare = []
    for c in all_candidates:
        row = {"candidate_id": c}
        for name, df in dfs.items():
            r = df[df['candidate_id'] == c]['rank'].values
            s = df[df['candidate_id'] == c]['score'].values
            if len(r) > 0:
                row[f"{name}_Rank"] = r[0]
                row[f"{name}_Score"] = round(float(s[0]), 4)
        rank_compare.append(row)
    
    rank_df = pd.DataFrame(rank_compare)
    if not rank_df.empty:
        first_col = list(dfs.keys())[0]
        rank_df = rank_df.sort_values(by=f"{first_col}_Rank")
        lines.append(rank_df.to_markdown(index=False))
    lines.append("\n")

    # 2. Present in ANY (Union)
    any_candidates = set.union(*candidates_sets.values())
    lines.append(f"## 2. Total Unique Candidates Across All Files: {len(any_candidates)}\n")

    # 3. Pairwise Differences & Edge Cases
    lines.append("## 3. Pairwise Differences & Edge Cases\n")
    
    pairs = list(itertools.combinations(dfs.keys(), 2))
    for name1, name2 in pairs:
        lines.append(f"### {name1} vs {name2}")
        
        c_set1 = candidates_sets[name1]
        c_set2 = candidates_sets[name2]
        
        only_in_1 = c_set1 - c_set2
        only_in_2 = c_set2 - c_set1
        
        lines.append(f"- **Only in {name1}**: {len(only_in_1)} candidates")
        if len(only_in_1) > 0:
            lines.append(f"  - {list(only_in_1)[:10]}{'...' if len(only_in_1)>10 else ''}")
        
        lines.append(f"- **Only in {name2}**: {len(only_in_2)} candidates")
        if len(only_in_2) > 0:
            lines.append(f"  - {list(only_in_2)[:10]}{'...' if len(only_in_2)>10 else ''}")
        
        intersection = c_set1.intersection(c_set2)
        if len(intersection) > 1:
            merged = pd.merge(dfs[name1], dfs[name2], on="candidate_id", suffixes=('_1', '_2'))
            corr = merged['rank_1'].corr(merged['rank_2'], method='spearman')
            lines.append(f"- **Rank Correlation (Spearman)**: {corr:.3f}")
            
            merged['rank_diff'] = merged['rank_2'] - merged['rank_1']
            merged['abs_rank_diff'] = merged['rank_diff'].abs()
            biggest_shifts = merged.sort_values(by='abs_rank_diff', ascending=False).head(3)
            lines.append("- **Biggest Rank Shifts (Top 3)**:")
            for _, row in biggest_shifts.iterrows():
                lines.append(f"  - {row['candidate_id']}: {name1} Rank {row['rank_1']} vs {name2} Rank {row['rank_2']} (Diff: {row['rank_diff']})")
        lines.append("\n")

    md_content = "\n".join(lines)
    
    # Save Markdown
    md_file = "comparison_report.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    # Convert to HTML
    html_body = markdown.markdown(md_content, extensions=['tables'])
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        h1, h2, h3 {{ color: #333; }}
        code {{ background-color: #f9f9f9; padding: 2px 4px; border-radius: 4px; }}
    </style>
    </head>
    <body>
    {html_body}
    </body>
    </html>
    """
    
    html_file = "comparison_report.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Generated {md_file} and {html_file}")
    

if __name__ == "__main__":
    main()
