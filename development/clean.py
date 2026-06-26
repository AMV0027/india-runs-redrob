import os

def clean_artifacts(directory="."):
    files_to_remove = [
        "features.parquet",
        "skills.index",
        "exp.index",
        "bm25.pkl",
        "submission.csv"
    ]
    
    removed_count = 0
    for file_name in files_to_remove:
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
            removed_count += 1
        else:
            print(f"Not found: {file_path}")
            
    if removed_count == 0:
        print("No index files were found to clean.")
    else:
        print(f"Successfully cleaned {removed_count} index files.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean generated index and submission files.")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing the index files.")
    args = parser.parse_args()
    
    clean_artifacts(args.dir)
