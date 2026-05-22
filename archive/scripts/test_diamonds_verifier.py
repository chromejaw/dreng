import csv
from drengr.generators.intent_verifier import IntentVerifier

def main():
    verifier = IntentVerifier()
    
    diamonds = []
    with open("output/diamond.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            diamonds.append(row["text"])
            
    print(f"Testing {len(diamonds)} diamond seeds against IntentVerifier (L1-L3)...\n")
    
    passed_count = 0
    failed_count = 0
    
    for i, text in enumerate(diamonds):
        # We pass "unknown" for family since we removed it from the CSV.
        # This will skip the Layer 2 family-specific gate and test them on Layer 1 (refusals) 
        # and Layer 3 (attack density floor).
        result = verifier.verify(text, "unknown")
        
        if result.passed:
            passed_count += 1
        else:
            failed_count += 1
            print(f"Seed #{i+1} FAILED at Layer {result.layer}: {result.reason}")
            
    print(f"\nResults: {passed_count} passed, {failed_count} failed.")

if __name__ == "__main__":
    main()
