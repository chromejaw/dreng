"""
Generate adversarial prompts using DeepSeek LLM via the full drengr pipeline.

Uses generate_prompts() — the single entry point that activates ALL pipeline
improvements automatically:
  - Fix #1: 10 concurrent LLM threads (ThreadPoolExecutor)
  - Fix #2: Temperature jitter (class-specific ranges, no cache duplicates)
  - Fix #2: CacheManager LRU (5000-entry cap, no OOM)
  - Fix #3: 16 seed pre-transforms (3392 unique LLM inputs from 212 seeds)
  - Fix #3: Parallel opener-fix
  - Family 20: multilingual_attacks (Arabic, Hindi, Hinglish, Chinese,
               Russian, Turkish, Swahili — multilingual_cot enforced)
  - Post-gen: length compliance, opener diversity, obfuscation transforms

Usage:
  python scripts/generate_50_deepseek.py [--count N]
"""

import sys
import json
import os
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from drengr.generators.semantic_paraphraser import (
    SOTASemanticParaphraseGenerator,
    _CONCURRENT_WORKERS,
    _PRE_TRANSFORM_TEMPLATES,
    LENGTH_CLASSES,
)
from drengr.generators.intent_verifier import IntentVerifier
from drengr.services.llm import ProductionLLMService
from drengr.services.embedding import MockEmbeddingService
from drengr.core.models import (
    GeneratedPrompt, PromptSpec, Category, Domain, Length, Difficulty
)

# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Generate adversarial prompts via drengr pipeline")
parser.add_argument("--count", type=int, default=50, help="Number of prompts to generate (default: 50)")
parser.add_argument("--output", type=str, default=None, help="Output file path (default: output/deepseek_<count>_elite.jsonl)")
args = parser.parse_args()

TOTAL       = args.count
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-api-key-here")
OUTPUT_FILE = Path(args.output) if args.output else (
    Path(__file__).parent.parent / "output" / f"deepseek_{TOTAL}_elite.jsonl"
)

# Hard-negative (semantic flip) rate — 15-20% of generated prompts are labelled
# as benign hard negatives (label=0). This is a generation config constant,
# not a CLI flag. Tune here for ablation experiments.
FLIP_RATE: float = 0.17

# ── Services ─────────────────────────────────────────────────────────────────
# Temperature in config is the BASE value — semantic_paraphraser.py overrides
# it per-call with class-specific jitter (short: 0.85-0.92, mid/long: 0.85-0.97).
# Setting 0.9 here keeps the base sane but jitter is what matters.
llm = ProductionLLMService({
    "api_key":       DEEPSEEK_API_KEY,
    "model":         "deepseek-chat",
    "base_url":      "https://api.deepseek.com",
    "temperature":   0.9,    # overridden by per-call jitter inside generate_prompts
    "max_tokens":    400,    # enough for long-class prompts (>200w target)
    "timeout":       45,     # multilingual CoT needs a bit more time
    "retry_attempts": 3,
})

embed = MockEmbeddingService()

# ── Generator ────────────────────────────────────────────────────────────────
gen = SOTASemanticParaphraseGenerator(
    embedding_service=embed,
    llm_service=llm,
    flip_rate=FLIP_RATE,
)

n_families = len(gen._families)
n_seeds    = sum(len(v) for v in gen._families.values())
n_unique   = n_seeds * len(_PRE_TRANSFORM_TEMPLATES)

print(f"╔══════════════════════════════════════════════════════════════╗")
print(f"║  drengr — Elite Adversarial Prompt Generation (DeepSeek)   ║")
print(f"╠══════════════════════════════════════════════════════════════╣")
print(f"║  Families:         {n_families:<3}  (incl. multilingual_attacks)        ║")
print(f"║  Raw seeds:        {n_seeds:<3}                                        ║")
print(f"║  Pre-transforms:   {len(_PRE_TRANSFORM_TEMPLATES):<3}  → {n_unique:<5} unique LLM inputs      ║")
print(f"║  Concurrent workers: {_CONCURRENT_WORKERS:<2}  (ThreadPoolExecutor)          ║")
print(f"║  Target count:     {TOTAL:<3}                                        ║")
print(f"║  LLM: deepseek-chat (temp jitter: 0.85–0.97)               ║")
print(f"╚══════════════════════════════════════════════════════════════╝")
print()

# Domain/difficulty are output metadata only — they do NOT affect generation.
# Domain.GENERAL is correct for an adversarial dataset that targets all contexts.
# Difficulty rotates hard/medium to give classifiers a balanced benchmark signal.
DIFFICULTIES = [Difficulty.HARD, Difficulty.HARD, Difficulty.MEDIUM, Difficulty.HARD]

specs = [
    PromptSpec(
        category=Category.SEMANTIC_PARAPHRASE,
        domain=Domain.GENERAL,
        length=Length.MEDIUM,
        difficulty=DIFFICULTIES[i % len(DIFFICULTIES)],
        paraphrase_family="auto",
        repeat_weight=1,
    )
    for i in range(TOTAL)
]

# ── Generate ──────────────────────────────────────────────────────────────────
# Single call — the full pipeline runs internally:
#   Phase 1  (sequential): seed assignment + pre-transform application
#   Phase 2  (parallel):   10 concurrent LLM calls via ThreadPoolExecutor
#   Phase 3  (sequential): dedup, template fallback, GeneratedPrompt assembly
#   Fix passes: length compliance, opener diversity, obfuscation transforms
print(f"Starting generation ({TOTAL} prompts, {_CONCURRENT_WORKERS} concurrent workers)...")
print()
t_start = time.time()

prompts: list[GeneratedPrompt] = gen.generate_prompts(TOTAL, specs)

elapsed = time.time() - t_start

# ── Intent drift verification ─────────────────────────────────────────────────
# Runs post-generation, before serialization. Pure regex, <1ms per sample.
# Stamps intent_drift=True on prompts that fail the verifier so the exporter
# can route them to the rejected log without re-checking.
verifier = IntentVerifier()

def _extract_family(paraphrase_family: str) -> str:
    """Extract clean family name from full paraphrase_family string."""
    return IntentVerifier._normalize_family(paraphrase_family)

drift_results = verifier.verify_batch([
    (p.prompt, _extract_family(p.paraphrase_family))
    for p in prompts
])
drift_map = {i: r for i, r in enumerate(drift_results)}

drift_report = verifier.drift_report(drift_results)
print()
print(f"Intent drift verification: {drift_report['drifted']}/{drift_report['total']} drifted "
      f"({drift_report['drift_rate']:.1%})")
if drift_report['drifted'] > 0:
    for dtype, count in drift_report['by_type'].items():
        layer = next(r.layer for r in drift_results if r.drift_type == dtype)
        print(f"  Layer {layer} — {dtype}: {count}")

# ── Print live summary ────────────────────────────────────────────────────────
# Classify length distribution
lc_counts: dict = {lc: 0 for lc in LENGTH_CLASSES}
for p in prompts:
    wc = len(p.prompt.split())
    for lc, (lo, hi) in LENGTH_CLASSES.items():
        if lo <= wc <= hi:
            lc_counts[lc] += 1
            break

families_seen: dict = {}
for p in prompts:
    fam = p.paraphrase_family.split("_")[1] if "_" in p.paraphrase_family else p.paraphrase_family
    families_seen[fam] = families_seen.get(fam, 0) + 1

for p in prompts:
    fam_label = p.paraphrase_family.split("semantic_")[1].rsplit("_", 1)[0] if "semantic_" in p.paraphrase_family else "unknown"
    wc = len(p.prompt.split())
    preview = p.prompt[:80].replace('\n', ' ')
    print(f"  [{p.id:>3}/{TOTAL}] {fam_label:28s} {wc:>3}w | {preview}...")

# ── Serialize to JSONL ────────────────────────────────────────────────────────
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, p in enumerate(prompts):
        is_flip = p.paraphrase_family.endswith("__flip")
        clean_family = p.paraphrase_family.removesuffix("__flip")
        drift_result = drift_map.get(i)
        is_drifted = drift_result is not None and not drift_result.passed
        record = {
            "id":                    p.id,
            "prompt":                p.prompt,
            "category":              p.category.value if hasattr(p.category, "value") else str(p.category),
            "paraphrase_family":     clean_family,
            "semantic_flip":         is_flip,     # True → label=0 hard negative in CSV
            "intent_drift":          is_drifted,  # True → rejected (payload neutered/refusal leaked)
            "drift_reason":          drift_result.reason if is_drifted else None,
            "safety_label":          p.safety_label.value if hasattr(p.safety_label, "value") else str(p.safety_label),
            "expected_policy_action": p.expected_policy_action.value if hasattr(p.expected_policy_action, "value") else str(p.expected_policy_action),
            "domain":                p.domain.value if hasattr(p.domain, "value") else str(p.domain),
            "difficulty":            p.difficulty.value if hasattr(p.difficulty, "value") else str(p.difficulty),
            "word_count":            len(p.prompt.split()),
            "length_class":          next(
                                       (lc for lc, (lo, hi) in LENGTH_CLASSES.items()
                                        if lo <= len(p.prompt.split()) <= hi),
                                       "unknown"
                                     ),
            "llm_generated":         True,
            "model":                 "deepseek-chat",
            "pipeline_version":      "v9",  # Fix1+Fix2+Fix3+Family20
            "generated_at":          datetime.now(timezone.utc).isoformat(),
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ── Final report ──────────────────────────────────────────────────────────────
rate = len(prompts) / elapsed if elapsed > 0 else 0

print()
print(f"╔══════════════════════════════════════════════════════════════╗")
print(f"║  Generation Complete                                        ║")
print(f"╠══════════════════════════════════════════════════════════════╣")
print(f"║  Generated:  {len(prompts):<3}  prompts                               ║")
print(f"║  Time:       {elapsed:>6.1f}s                                      ║")
print(f"║  Rate:       {rate:>5.2f}  prompts/sec                           ║")
print(f"║  Length:     short={lc_counts['short']}  medium={lc_counts['medium']}  long={lc_counts['long']}              ║")
print(f"╚══════════════════════════════════════════════════════════════╝")
print()

# Family breakdown with bar chart
print("Family breakdown:")
all_family_names = list(gen._families.keys())
family_prompt_counts: dict = {}
for p in prompts:
    for fname in all_family_names:
        if fname in p.paraphrase_family:
            family_prompt_counts[fname] = family_prompt_counts.get(fname, 0) + 1
            break

max_count = max(family_prompt_counts.values(), default=1)
for fname in all_family_names:
    count = family_prompt_counts.get(fname, 0)
    bar_filled = round(count / max_count * 10)
    bar = "█" * bar_filled + "░" * (10 - bar_filled)
    tag = " ◀ multilingual" if fname == "multilingual_attacks" else ""
    print(f"  {fname:30s} {bar} {count}{tag}")

print(f"\n📁 Output: {OUTPUT_FILE}")
print(f"   {len(prompts)} records | {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
