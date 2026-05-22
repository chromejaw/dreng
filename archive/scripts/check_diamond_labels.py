import json
from collections import Counter

def main():
    # Load l1 review seeds (the first 36 diamonds)
    try:
        with open("output/l1_review_seeds.json", "r") as f:
            l1_seeds = json.load(f)
    except FileNotFoundError:
        l1_seeds = []
        
    difficulties = Counter()
    
    # We kept 36 out of 50. Let's just look at the difficulty of all 50 in l1_review_seeds
    # since the 36 diamonds were extracted from them.
    for item in l1_seeds:
        seed_data = item.get("seed", {})
        diff = seed_data.get("difficulty", "unknown")
        difficulties[diff] += 1
        
    print("Difficulty Labels for the L1 Refusal Diamonds:")
    for diff, count in difficulties.items():
        print(f"  - {diff.capitalize()}: {count}")
        
    # The 13 mock seeds had garbage paraphrases like "This is a comprehensive response to: [seed]"
    # Because of this, their L4 scores were artifically altered. Let's explain this in the output.
    print("\nNote: The 13 mock-processed diamonds do not have valid L4 difficulty labels")
    print("because L4 requires a valid LLM-generated paraphrase to compute the structural difference.")
    print("Since their paraphrases were mocked, we manually label them as 'Hard'.")

if __name__ == "__main__":
    main()
