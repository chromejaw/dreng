import json
import csv

def main():
    mined_seeds_path = "output/mined_seeds.json"
    diamond_csv_path = "output/diamond.csv"
    
    with open(mined_seeds_path, "r") as f:
        data = json.load(f)
        
    multilingual = data.get("families", {}).get("multilingual_attacks", {})
    seeds = multilingual.get("seeds", [])
    
    if len(seeds) < 13:
        print("Not enough seeds to extract. Did this already run?")
        return
        
    # Extract the last 13 seeds (the mocked ones)
    mocked_seeds = seeds[-13:]
    # Keep the rest in mined_seeds
    multilingual["seeds"] = seeds[:-13]
    
    # Save back to mined_seeds.json
    with open(mined_seeds_path, "w") as f:
        json.dump(data, f, indent=2)
        
    # Append to diamond.csv
    added = 0
    with open(diamond_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for s in mocked_seeds:
            text = s.get("text", "")
            if text:
                # Add text and label=1
                writer.writerow([text, 1])
                added += 1
                
    print(f"Successfully removed 13 mock-processed seeds from mined_seeds.json.")
    print(f"Appended {added} valid seeds to diamond.csv.")

if __name__ == "__main__":
    main()
