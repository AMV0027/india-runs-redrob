import sys
import time
import subprocess
import csv
from datetime import datetime
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <command to run>")
        sys.exit(1)
        
    command = sys.argv[1:]
    command_str = " ".join(command)
    
    # Start timer
    start_time = time.perf_counter()
    
    # Run the command, redirecting nothing so it runs normally (doesn't affect performance)
    print(f"Running: {command_str}")
    process = subprocess.run(command)
    
    # Stop timer
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write to benchmark.csv
    csv_file = "benchmark.csv"
    file_exists = os.path.exists(csv_file)
        
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Command", "Elapsed Time (seconds)", "Elapsed Time (minutes)"])
        
        minutes = elapsed_time / 60.0
        writer.writerow([timestamp, command_str, f"{elapsed_time:.4f}", f"{minutes:.4f}"])
        
    print(f"\n--- Benchmark Results ---")
    print(f"Command: {command_str}")
    print(f"Completed at: {timestamp}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds ({minutes:.2f} minutes)")
    print(f"Results appended to {csv_file}")
    
    # Exit with the same return code as the command
    sys.exit(process.returncode)

if __name__ == "__main__":
    main()
