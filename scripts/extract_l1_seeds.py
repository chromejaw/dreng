import json
import re
import sys

def main():
    log_file = "/Users/karan/.gemini/antigravity/brain/f312375d-f669-4d73-9144-44c29beb08d8/.system_generated/tasks/task-316.log"
    mined_seeds_file = "output/mined_seeds.json"
    output_file = "output/l1_review_seeds.json"

    # Load existing seeds
    try:
        with open(mined_seeds_file, "r") as f:
            data = json.load(f)
            families = data.get("families", {})
    except Exception as e:
        print(f"Error loading {mined_seeds_file}: {e}")
        return

    review_seeds = []
    current_family = None

    family_header_pattern = re.compile(r"\[\d+/\d+\]\s+([a-zA-Z0-9_]+):\s+\d+\s+candidates")
    seed_line_pattern = re.compile(r"\[(\d+)/\d+\]\s+drift=.*?\s+L1=([1-9]\d*)\s+")

    try:
        with open(log_file, "r") as f:
            for line in f:
                # Check for family header
                fam_match = family_header_pattern.search(line)
                if fam_match:
                    current_family = fam_match.group(1)
                    continue
                
                # Check for L1 hits
                seed_match = seed_line_pattern.search(line)
                if seed_match and current_family:
                    idx_1based = int(seed_match.group(1))
                    l1_count = int(seed_match.group(2))
                    
                    # Fetch from JSON
                    fam_data = families.get(current_family, {})
                    seeds = fam_data.get("seeds", [])
                    if 1 <= idx_1based <= len(seeds):
                        seed_obj = seeds[idx_1based - 1]
                        review_seeds.append({
                            "family": current_family,
                            "index": idx_1based,
                            "l1_count": l1_count,
                            "seed": seed_obj
                        })
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return

    with open(output_file, "w") as f:
        json.dump(review_seeds, f, indent=2)

    print(f"Extraction complete! Found {len(review_seeds)} seeds with L1 hits.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
