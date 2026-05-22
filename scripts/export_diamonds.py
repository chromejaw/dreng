import json
import csv
import os

# The 14 junk seeds identified during triage
JUNK_SEEDS = [
    ("direct_override", 13),
    ("direct_override", 14),
    ("direct_override", 18),
    ("system_extraction", 28),
    ("system_extraction", 40),
    ("payload_smuggling", 18),
    ("multi_turn", 6),
    ("multi_turn", 24),
    ("context_manipulation", 10),
    ("encoding_wrappers", 3),
    ("encoding_wrappers", 17),
    ("math_abstraction", 2),
    ("emotional_manipulation", 5),
    ("multilingual_attacks", 1)
]

def main():
    input_file = "output/l1_review_seeds.json"
    output_file = "output/diamond.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    diamonds = []
    for item in data:
        family = item.get("family")
        index = item.get("index")
        
        # Check if it's in the junk list
        if (family, index) in JUNK_SEEDS:
            continue
            
        seed_data = item.get("seed", {})
        
        # Flatten for CSV
        row = {
            "text": seed_data.get("text", ""),
            "label": 1
        }
        diamonds.append(row)
        
    if not diamonds:
        print("No diamonds found to export.")
        return
        
    # Write to CSV
    fieldnames = ["text", "label"]
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diamonds)
        
    print(f"Successfully exported {len(diamonds)} diamond seeds to {output_file}")

if __name__ == "__main__":
    main()
