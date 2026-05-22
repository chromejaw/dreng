import csv
import json

def main():
    diamond_csv_path = "output/diamond.csv"
    
    seeds = []
    with open(diamond_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0] != "text":  # skip header
                seeds.append(row[0])
                
    # Get the last 13
    last_13 = seeds[-13:]
    
    with open("scratch/last_13_seeds.md", "w", encoding="utf-8") as f:
        for i, text in enumerate(last_13):
            f.write(f"## Seed {i+1}\n")
            f.write(f"```text\n{text}\n```\n\n")
            
if __name__ == "__main__":
    main()
