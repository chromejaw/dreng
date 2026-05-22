import json

def main():
    with open("output/mined_seeds.json", "r") as f:
        data = json.load(f)
        
    multilingual = data.get("families", {}).get("multilingual_attacks", {}).get("seeds", [])
    
    # We know the last 13 seeds hit the mock LLM because the API ran out of credits during this family.
    # We will just print the last 13 seeds.
    mocked_seeds = multilingual[-13:]
    
    print(f"Found {len(mocked_seeds)} seeds that were processed by MockLLMService.\n")
    for i, s in enumerate(mocked_seeds):
        text = s.get("text", "")
        # truncate text for cleaner printing if it's very long
        print(f"--- Seed {i+1} ---")
        print(text)
        print()

if __name__ == "__main__":
    main()
