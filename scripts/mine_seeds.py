#!/usr/bin/env python3
"""
mine_seeds.py — Adversarial seed mining pipeline for drengr.

Mines high-quality attack seeds from real attacker datasets, deduplicates,
pre-filters, clusters to drengr's 20 attack families, and quality-ranks
by paraphrase drift rate.

Architecture:
  Phase 1 — Ingest:     Download + filter confirmed jailbreaks from HuggingFace
  Phase 2 — Dedup:      Embed-level dedup at cosine > 0.95
  Phase 3 — Pre-filter: IntentVerifier template drift check (no LLM, <1ms/sample)
  Phase 4 — Cluster:    SBERT embed → anchor-supervised k-nearest to 20 families
  Phase 5 — Select:     70% centroid-proximal + 30% outlier per family
  Phase 6 — Rank:       20-paraphrase LLM drift check, sort by drift rate ascending

Output: output/mined_seeds.json — directly loadable by semantic_paraphraser.py

Usage:
  # Phase 1-5 (no LLM needed, ~2 minutes):
  python scripts/mine_seeds.py

  # Phase 6 (needs DeepSeek API, ~30 minutes for 1000 candidates):
  python scripts/mine_seeds.py --drift-check

  # Full pipeline:
  python scripts/mine_seeds.py --drift-check --top-k 50
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ── Per-family drift rate acceptance thresholds ────────────────────────────────
# Maximum acceptable drift rate (fraction of paraphrases that can fail L1-L3)
# for a seed to qualify as "high quality" after Phase 6.
#
# Rationale:
#   direct_override: Indirect phrasing ("reassigned", "supersedes") is inherently
#     harder for any paraphraser to preserve lexically — accept up to 30% drift.
#   hybrid_attacks: Multi-vector attacks require ALL attack components to survive
#     the paraphrase chain — tighter 15% threshold ensures signal fidelity.
#   encoding_wrappers: Base64/hex encoding is frequently stripped by paraphrasers
#     — but the framing (decode + execute) must survive, so 20% cap.
#   Default: 25% across all other families.
#
# Usage: seeds with drift_rate > family threshold are excluded from top_k
# selection. If no seeds pass the threshold, the pipeline falls back to
# returning the least-drifted seeds (at least 1 per family).
FAMILY_DRIFT_THRESHOLDS: dict[str, float] = {
    "direct_override":       0.30,  # inherently hard to paraphrase faithfully
    "persona_hijack":        0.25,
    "system_extraction":     0.25,
    "payload_smuggling":     0.25,
    "social_engineering":    0.25,
    "indirect_stealth":      0.25,
    "multi_turn":            0.25,
    "context_manipulation":  0.25,
    "encoding_wrappers":     0.20,  # base64 is fragile across paraphrasers
    "competing_objectives":  0.25,
    "adversarial_poetry":    0.25,
    "math_abstraction":      0.25,
    "emotional_manipulation":0.25,
    "context_poisoning":     0.20,  # injection markers must survive verbatim
    "gray_box_deception":    0.25,
    "goal_redirection":      0.25,
    "synthetic_context":     0.25,
    "context_flooding":      0.25,
    "hybrid_attacks":        0.15,  # multi-vector: all components must survive
    "multilingual_attacks":  0.25,
}
_DEFAULT_DRIFT_THRESHOLD = 0.25

log = logging.getLogger("mine_seeds")



# ═══════════════════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RawSeed:
    """A candidate seed prompt from an external dataset."""
    text: str
    source: str                    # dataset name
    source_id: str = ""            # row ID or hash within source
    source_label: str = ""         # original label (e.g. "jailbreak", "correct=True")
    word_count: int = 0
    char_count: int = 0
    assigned_family: str = ""      # filled in Phase 4
    centroid_distance: float = 0.0 # distance from family centroid
    is_outlier: bool = False       # True = selected as outlier (Phase 5)
    drift_rate: float = -1.0       # filled in Phase 6 (-1 = not tested)
    drift_details: dict = field(default_factory=dict)
    embedding: Optional[list] = field(default=None, repr=False)

    def __post_init__(self):
        self.word_count = len(self.text.split())
        self.char_count = len(self.text)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Ingest
# ═══════════════════════════════════════════════════════════════════════════════

def _ingest_jailbreakhub() -> list[RawSeed]:
    """walledai/JailbreakHub — 15K prompts, ~1.4K confirmed jailbreaks."""
    from datasets import load_dataset
    log.info("Ingesting walledai/JailbreakHub...")
    try:
        ds = load_dataset("walledai/JailbreakHub", split="train")
        seeds = []
        for i, row in enumerate(ds):
            # Only take confirmed jailbreaks
            if not row.get("jailbreak", False):
                continue
            text = (row.get("prompt") or "").strip()
            if not text or len(text) < 15:
                continue
            seeds.append(RawSeed(
                text=text,
                source="jailbreakhub",
                source_id=f"jailbreakhub_{i}",
                source_label="jailbreak=True",
            ))
        log.info(f"  JailbreakHub: {len(seeds)} confirmed jailbreaks from {len(ds)} total")
        return seeds
    except Exception as e:
        log.warning(f"  JailbreakHub failed: {e}")
        return []


def _ingest_jackhhao() -> list[RawSeed]:
    """jackhhao/jailbreak-classification — ~666 jailbreak prompts."""
    from datasets import load_dataset
    log.info("Ingesting jackhhao/jailbreak-classification...")
    try:
        seeds = []
        for split in ["train", "test"]:
            try:
                ds = load_dataset("jackhhao/jailbreak-classification", split=split)
            except Exception:
                continue
            for i, row in enumerate(ds):
                if row.get("type", "").lower() != "jailbreak":
                    continue
                text = (row.get("prompt") or "").strip()
                if not text or len(text) < 15:
                    continue
                seeds.append(RawSeed(
                    text=text,
                    source="jackhhao",
                    source_id=f"jackhhao_{split}_{i}",
                    source_label="type=jailbreak",
                ))
        log.info(f"  jackhhao: {len(seeds)} jailbreak prompts")
        return seeds
    except Exception as e:
        log.warning(f"  jackhhao failed: {e}")
        return []


def _ingest_rubend18() -> list[RawSeed]:
    """rubend18/ChatGPT-Jailbreak-Prompts — 79 classic jailbreak personas."""
    from datasets import load_dataset
    log.info("Ingesting rubend18/ChatGPT-Jailbreak-Prompts...")
    try:
        ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split="train")
        seeds = []
        for i, row in enumerate(ds):
            text = (row.get("Prompt") or "").strip()
            if not text or len(text) < 15:
                continue
            seeds.append(RawSeed(
                text=text,
                source="rubend18",
                source_id=f"rubend18_{i}",
                source_label=f"name={row.get('Name', '')}",
            ))
        log.info(f"  rubend18: {len(seeds)} jailbreak prompts")
        return seeds
    except Exception as e:
        log.warning(f"  rubend18 failed: {e}")
        return []


def _ingest_jailbreakbench() -> list[RawSeed]:
    """JailbreakBench/JBB-Behaviors — 100 harmful behavior goals."""
    from datasets import load_dataset
    log.info("Ingesting JailbreakBench/JBB-Behaviors...")
    try:
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        seeds = []
        for i, row in enumerate(ds):
            text = (row.get("Goal") or "").strip()
            if not text or len(text) < 10:
                continue
            seeds.append(RawSeed(
                text=text,
                source="jailbreakbench",
                source_id=f"jbb_{i}",
                source_label=f"category={row.get('Category', '')}",
            ))
        log.info(f"  JailbreakBench: {len(seeds)} behavior goals")
        return seeds
    except Exception as e:
        log.warning(f"  JailbreakBench failed: {e}")
        return []


def _ingest_deepset() -> list[RawSeed]:
    """deepset/prompt-injections — labeled prompt injection dataset."""
    from datasets import load_dataset
    log.info("Ingesting deepset/prompt-injections...")
    try:
        ds = load_dataset("deepset/prompt-injections", split="train")
        seeds = []
        for i, row in enumerate(ds):
            # label 1 = injection
            if row.get("label", 0) != 1:
                continue
            text = (row.get("text") or "").strip()
            if not text or len(text) < 15:
                continue
            seeds.append(RawSeed(
                text=text,
                source="deepset",
                source_id=f"deepset_{i}",
                source_label="label=1",
            ))
        log.info(f"  deepset: {len(seeds)} prompt injections")
        return seeds
    except Exception as e:
        log.warning(f"  deepset failed: {e}")
        return []


def _ingest_jailbreakv28k() -> list[RawSeed]:
    """JailbreakV-28K/JailBreakV-28K RedTeam_2K — 2K red team prompts."""
    from datasets import load_dataset
    log.info("Ingesting JailbreakV-28K/RedTeam_2K...")
    try:
        ds = load_dataset("JailbreakV-28K/JailBreakV-28K", "RedTeam_2K", split="RedTeam_2K")
        seeds = []
        for i, row in enumerate(ds):
            # Try common field names for the prompt text
            text = ""
            for field in ["prompt", "text", "question", "query", "jailbreak_query", "redteam_query"]:
                if field in row and row[field]:
                    text = str(row[field]).strip()
                    break
            if not text:
                # Try first string column
                for k, v in row.items():
                    if isinstance(v, str) and len(v) > 15:
                        text = v.strip()
                        break
            if not text or len(text) < 15:
                continue
            seeds.append(RawSeed(
                text=text,
                source="jailbreakv28k",
                source_id=f"jbv28k_{i}",
                source_label="redteam_2k",
            ))
        log.info(f"  JailbreakV-28K: {len(seeds)} red team prompts from {len(ds)} total")
        return seeds
    except Exception as e:
        log.warning(f"  JailbreakV-28K failed: {e}")
        return []


def _ingest_trustailab() -> list[RawSeed]:
    """TrustAIRLab/in-the-wild-jailbreak-prompts — real jailbreak prompts from 2023."""
    from datasets import load_dataset
    log.info("Ingesting TrustAIRLab/in-the-wild-jailbreak-prompts...")
    try:
        seeds = []
        for config in ["jailbreak_2023_05_07", "jailbreak_2023_12_25"]:
            try:
                ds = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", config, split="train")
                for i, row in enumerate(ds):
                    text = ""
                    for field in ["prompt", "text", "content", "question"]:
                        if field in row and row[field]:
                            text = str(row[field]).strip()
                            break
                    if not text:
                        for k, v in row.items():
                            if isinstance(v, str) and len(v) > 15:
                                text = v.strip()
                                break
                    if not text or len(text) < 15:
                        continue
                    seeds.append(RawSeed(
                        text=text,
                        source="trustailab",
                        source_id=f"trustai_{config}_{i}",
                        source_label=config,
                    ))
            except Exception as e:
                log.debug(f"  TrustAIRLab config {config} failed: {e}")
        log.info(f"  TrustAIRLab: {len(seeds)} jailbreak prompts")
        return seeds
    except Exception as e:
        log.warning(f"  TrustAIRLab failed: {e}")
        return []


def phase_1_ingest() -> list[RawSeed]:
    """Download and filter all available datasets."""
    log.info("═══ Phase 1: Ingest ═══")
    all_seeds: list[RawSeed] = []

    # Try all sources — failures are non-fatal
    for ingest_fn in [
        _ingest_jailbreakhub,
        _ingest_jackhhao,
        _ingest_rubend18,
        _ingest_jailbreakbench,
        _ingest_deepset,
        _ingest_jailbreakv28k,
        _ingest_trustailab,
    ]:
        all_seeds.extend(ingest_fn())

    log.info(f"Phase 1 complete: {len(all_seeds)} raw seeds from all sources")
    return all_seeds


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Dedup (exact + embed-level)
# ═══════════════════════════════════════════════════════════════════════════════

def phase_2_dedup(seeds: list[RawSeed]) -> list[RawSeed]:
    """Remove exact duplicates and near-duplicates (cosine > 0.95)."""
    log.info("═══ Phase 2: Dedup ═══")

    # Step 1: Exact text dedup (fast, O(n))
    seen_hashes: set[str] = set()
    deduped: list[RawSeed] = []
    for s in seeds:
        h = hashlib.sha256(s.text.lower().strip().encode()).hexdigest()[:16]
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(s)
    exact_dupes = len(seeds) - len(deduped)
    log.info(f"  Exact dedup: {len(seeds)} → {len(deduped)} ({exact_dupes} removed)")

    if len(deduped) <= 100:
        log.info("  Skipping embed dedup (≤100 samples)")
        return deduped

    # Step 2: Embed-level dedup (cosine > 0.95)
    log.info("  Computing embeddings for dedup...")
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    except ImportError:
        log.warning("  sentence-transformers unavailable — skipping embed-level dedup")
        return deduped

    # Truncate very long prompts for embedding (first 512 chars is sufficient for dedup)
    texts = [s.text[:512] for s in deduped]
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=128, normalize_embeddings=True)
    if embeddings is None or len(embeddings) == 0:
        log.warning("  Embedder failed — skipping embed-level dedup")
        return deduped

    # Greedy sequential dedup: keep first occurrence, remove any later
    # prompt with cosine > 0.95 to any already-kept prompt.
    # O(n²) but n is at most ~2K after exact dedup, so <10s.
    keep_mask = np.ones(len(deduped), dtype=bool)
    for i in range(len(deduped)):
        if not keep_mask[i]:
            continue
        # Compare against all later prompts
        if i + 1 < len(deduped):
            sims = embeddings[i] @ embeddings[i + 1:].T
            for j_offset in np.where(sims > 0.95)[0]:
                keep_mask[i + 1 + j_offset] = False

    embed_deduped = [s for s, keep in zip(deduped, keep_mask) if keep]
    embed_dupes = len(deduped) - len(embed_deduped)
    log.info(f"  Embed dedup (cosine > 0.95): {len(deduped)} → {len(embed_deduped)} ({embed_dupes} removed)")

    # Store embeddings for reuse in Phase 4
    kept_idx = 0
    for i, s in enumerate(deduped):
        if keep_mask[i]:
            s.embedding = embeddings[i].tolist()
            kept_idx += 1

    return embed_deduped


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Pre-filter (intent verifier, no LLM)
# ═══════════════════════════════════════════════════════════════════════════════

def phase_3_prefilter(seeds: list[RawSeed]) -> list[RawSeed]:
    """Run IntentVerifier as a fast pre-filter. Keep seeds with ≥1 attack signal hit."""
    log.info("═══ Phase 3: Pre-filter (IntentVerifier) ═══")

    from drengr.generators.intent_verifier import IntentVerifier, _ATTACK_DENSITY_SIGNALS

    passed: list[RawSeed] = []
    failed = 0

    for s in seeds:
        text = s.text.strip()
        # For pre-filtering, we just need to know if the seed has ANY attack signal.
        # We're not checking family-specific patterns yet (family unknown at this stage).
        # Layer 1 (refusal) should reject — if the seed IS a refusal, it's not an attack.
        # Layer 3 (density) is the main filter — zero hits = not a useful seed.

        verifier = IntentVerifier()
        # Use "unknown" family to skip L2 (no family assigned yet)
        result = verifier.verify(text, "unknown")

        if result.passed:
            passed.append(s)
        elif result.drift_type == "REFUSAL_LEAK":
            # Genuine refusal text — definitely not a seed
            failed += 1
        elif result.drift_type == "SIGNAL_STARVED":
            # Zero attack signal — not useful as a seed
            failed += 1
        else:
            # L2 failure with unknown family — shouldn't happen, but keep conservatively
            passed.append(s)

    log.info(f"  Pre-filter: {len(seeds)} → {len(passed)} ({failed} rejected, "
             f"{len(seeds) - len(passed) - failed} kept despite L2)")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Embed & Cluster to 20 families
# ═══════════════════════════════════════════════════════════════════════════════

# The 20 family names, matching semantic_paraphraser.py exactly
FAMILY_NAMES = [
    "direct_override",
    "persona_hijack",
    "system_extraction",
    "payload_smuggling",
    "social_engineering",
    "indirect_stealth",
    "multi_turn",
    "context_manipulation",
    "encoding_wrappers",
    "competing_objectives",
    "adversarial_poetry",
    "math_abstraction",
    "emotional_manipulation",
    "context_poisoning",
    "gray_box_deception",
    "goal_redirection",
    "synthetic_context",
    "context_flooding",
    "hybrid_attacks",
    "multilingual_attacks",
]


def _get_existing_seeds() -> dict[str, list[str]]:
    """Load existing seeds from semantic_paraphraser.py for centroid computation."""
    from drengr.generators.semantic_paraphraser import SOTASemanticParaphraseGenerator

    # Create a minimal instance just to access the seed families
    class FakeLLM:
        def call(self, *a, **kw): return ""
        def generate(self, *a, **kw): return []
        def __getattr__(self, _): return lambda *a, **kw: ""

    p = SOTASemanticParaphraseGenerator.__new__(SOTASemanticParaphraseGenerator)
    p.llm_service = FakeLLM()
    p._flip_rate = 0.0
    p.logger = logging.getLogger("fake")
    p._load_seed_prompts()

    return {name: seeds for name, seeds in p._families.items()}


def phase_4_cluster(seeds: list[RawSeed]) -> list[RawSeed]:
    """Embed seeds with BGE-small, cluster to 20 families using existing seed centroids."""
    log.info("═══ Phase 4: Embed & Cluster ═══")

    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    except ImportError:
        log.warning("  sentence-transformers unavailable — skipping Phase 4")
        return seeds

    # Step 1: Compute centroids from existing seeds
    log.info("  Computing family centroids from existing seeds...")
    existing = _get_existing_seeds()
    centroids: dict[str, np.ndarray] = {}

    for family_name in FAMILY_NAMES:
        family_seeds = existing.get(family_name, [])
        if not family_seeds:
            log.warning(f"  No existing seeds for {family_name}, skipping centroid")
            continue
        # Truncate to 512 chars for embedding consistency
        texts = [s[:512] for s in family_seeds]
        embs = embedder.encode(texts, batch_size=64, normalize_embeddings=True)
        if embs is None or len(embs) == 0:
            log.warning(f"  Embedder unavailable for {family_name} centroid")
            continue
        centroids[family_name] = embs.mean(axis=0)
        # Re-normalize the centroid
        centroids[family_name] /= np.linalg.norm(centroids[family_name])

    log.info(f"  Computed {len(centroids)} family centroids")

    # Step 2: Embed candidate seeds (reuse if already embedded in Phase 2)
    needs_embedding = [s for s in seeds if s.embedding is None]
    if needs_embedding:
        log.info(f"  Embedding {len(needs_embedding)} seeds...")
        texts = [s.text[:512] for s in needs_embedding]
        embs = embedder.encode(texts, show_progress_bar=True, batch_size=128, normalize_embeddings=True)
        if embs is not None and len(embs) > 0:
            for s, emb in zip(needs_embedding, embs):
                s.embedding = emb.tolist()

    # Step 3: Assign each seed to nearest centroid
    log.info("  Assigning seeds to families via nearest centroid...")
    centroid_names = list(centroids.keys())
    centroid_matrix = np.array([centroids[n] for n in centroid_names])  # (20, dim)

    seed_matrix = np.array([s.embedding for s in seeds])  # (N, dim)
    # Cosine similarity (embeddings are already normalized)
    similarities = seed_matrix @ centroid_matrix.T  # (N, 20)

    for i, s in enumerate(seeds):
        best_idx = int(np.argmax(similarities[i]))
        s.assigned_family = centroid_names[best_idx]
        s.centroid_distance = float(1.0 - similarities[i, best_idx])  # distance = 1 - cosine_sim

    # Report distribution
    from collections import Counter
    dist = Counter(s.assigned_family for s in seeds)
    log.info("  Family distribution:")
    for fam in FAMILY_NAMES:
        count = dist.get(fam, 0)
        log.info(f"    {fam:30s}: {count:4d}")

    return seeds


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Select candidates (70% centroid + 30% outlier)
# ═══════════════════════════════════════════════════════════════════════════════

def phase_5_select(seeds: list[RawSeed], candidates_per_family: int = 50) -> list[RawSeed]:
    """Select top candidates per family: 70% nearest to centroid, 30% farthest (outliers)."""
    log.info("═══ Phase 5: Select Candidates ═══")

    from collections import defaultdict

    # Group by family
    by_family: dict[str, list[RawSeed]] = defaultdict(list)
    for s in seeds:
        by_family[s.assigned_family].append(s)

    selected: list[RawSeed] = []
    n_centroid = int(candidates_per_family * 0.70)
    n_outlier = candidates_per_family - n_centroid

    for fam in FAMILY_NAMES:
        family_seeds = by_family.get(fam, [])
        if not family_seeds:
            log.warning(f"  {fam}: 0 candidates (no seeds assigned)")
            continue

        # Sort by centroid distance (ascending = closest first)
        family_seeds.sort(key=lambda s: s.centroid_distance)

        # Take 70% nearest (centroid-proximal)
        centroid_picks = family_seeds[:min(n_centroid, len(family_seeds))]
        for s in centroid_picks:
            s.is_outlier = False

        # Take 30% farthest (outliers — boundary signal)
        # But exclude the centroid picks to avoid overlap
        remaining = family_seeds[len(centroid_picks):]
        remaining.sort(key=lambda s: s.centroid_distance, reverse=True)
        outlier_picks = remaining[:min(n_outlier, len(remaining))]
        for s in outlier_picks:
            s.is_outlier = True

        family_selected = centroid_picks + outlier_picks
        selected.extend(family_selected)
        log.info(f"  {fam:30s}: {len(family_selected):3d} selected "
                 f"({len(centroid_picks)} centroid + {len(outlier_picks)} outlier) "
                 f"from {len(family_seeds)} available")

    log.info(f"Phase 5 complete: {len(selected)} total candidates across all families")
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6 — Drift quality check (LLM-based, optional)
# ═══════════════════════════════════════════════════════════════════════════════

def phase_6_drift_check(
    seeds: list[RawSeed],
    paraphrases_per_seed: int = 20,
    top_k: int = 50,
    only_family: str | None = None,
    jsonl_out: "Path | None" = None,
) -> list[RawSeed]:
    """Run N paraphrases per candidate seed, measure drift rate, rank by quality.

    Args:
        seeds: Candidate seeds (from Phase 5 or loaded from JSON).
        paraphrases_per_seed: LLM paraphrases per seed for drift measurement.
        top_k: Max seeds to keep per family after ranking.
        only_family: If set, only run drift check for this one family.
                     All other seeds pass through unchanged.
        jsonl_out: If set, passing paraphrases are written here as export_to_csv-
                   compatible JSONL records. Each record carries the seed's
                   l4_score and seed_difficulty so it flows through the export
                   gate straight into dreng.csv + dreng.meta.jsonl.
                   Records from pass-through families (not scored) are omitted.
    """
    log.info("═══ Phase 6: Drift Quality Check ═══")


    from drengr.generators.intent_verifier import IntentVerifier
    from drengr.generators.semantic_paraphraser import SOTASemanticParaphraseGenerator
    from drengr.services.service_factory import ServiceFactory
    from drengr.services.llm import MockLLMService

    verifier = IntentVerifier()

    # Use the same ServiceFactory path as the production CLI.
    # Reads DRENGR_API_KEY / DRENGR_LLM_MODEL / DRENGR_LLM_BASE_URL from environment.
    llm = ServiceFactory().get_default_container().get_llm_service()

    if isinstance(llm, MockLLMService):
        log.error(
            "DRENGR_API_KEY is not set — Phase 6 requires a live LLM.\n"
            "  Set your DeepSeek key and retry:\n"
            "    export DRENGR_API_KEY=sk-...\n"
            "    python scripts/mine_seeds.py --drift-check --from-json --family direct_override"
        )
        sys.exit(1)

    log.info(f"  LLM: {type(llm).__name__} / {llm.model}")

    # Use __new__ to bypass __init__ (which requires embedding_service)
    paraphraser = SOTASemanticParaphraseGenerator.__new__(SOTASemanticParaphraseGenerator)
    paraphraser.llm_service = llm
    paraphraser._flip_rate = 0.0  # No flips for drift check
    paraphraser.logger = logging.getLogger("drift_check")
    paraphraser._load_seed_prompts()  # Needed for _llm_paraphrase_with_length internals

    # Open sidecar JSONL for passing paraphrases (if requested).
    # We open once here and pass the file handle into the loop so all families
    # share one file — no partial writes on KeyboardInterrupt (we flush per seed).
    import contextlib
    _jsonl_fh = open(jsonl_out, "w", encoding="utf-8") if jsonl_out else None
    _jsonl_ctx = contextlib.nullcontext() if _jsonl_fh is None else contextlib.nullcontext()
    _saved_total = 0  # running count of paraphrases saved to JSONL

    from collections import defaultdict
    by_family: dict[str, list[RawSeed]] = defaultdict(list)
    for s in seeds:
        by_family[s.assigned_family].append(s)

    checked_seeds: list[RawSeed] = []

    active_families = [only_family] if only_family else FAMILY_NAMES
    active_seeds = [s for s in seeds if s.assigned_family in active_families]
    log.info(f"  Families: {active_families if only_family else 'all 20'}")
    log.info(f"  {len(active_seeds)} candidates × {paraphrases_per_seed} paraphrases = "
             f"{len(active_seeds) * paraphrases_per_seed} LLM calls")

    for fam_idx, fam in enumerate(active_families):
        family_seeds = by_family.get(fam, [])
        if not family_seeds:
            continue

        log.info(f"  [{fam_idx+1}/{len(FAMILY_NAMES)}] {fam}: {len(family_seeds)} candidates")

        for seed_idx, s in enumerate(family_seeds):
            # Generate paraphrases
            drifted = 0
            drift_layers = {1: 0, 2: 0, 3: 0}
            best_paraphrase: str | None = None   # last clean paraphrase for l4_score
            passing_paraphrases: list[str] = []  # all clean paraphrases — saved to JSONL

            # Paraphrases run concurrently — 10 workers per seed.
            # Workers are pure functions (no shared writes); aggregation is
            # in the main thread after as_completed(), so no locking needed.
            # Wall clock: ~60s sequential → ~6s parallel per seed.
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _paraphrase_worker(p_idx: int):
                """Returns (text | None, passed: bool, layer: int | None)."""
                try:
                    result = paraphraser._llm_paraphrase_with_length(
                        s.text, 1, "medium", "prose"
                    )
                    if not result:
                        return None, False, None
                    para = result[0] if isinstance(result, list) else result
                    vr = verifier.verify(para, fam)
                    return para, vr.passed, (vr.layer if not vr.passed else None)
                except Exception as e:
                    log.debug(f"    Paraphrase {p_idx} failed: {e}")
                    return None, False, None

            _workers = min(10, paraphrases_per_seed)
            with ThreadPoolExecutor(max_workers=_workers) as _pool:
                _futs = {
                    _pool.submit(_paraphrase_worker, p_i): p_i
                    for p_i in range(paraphrases_per_seed)
                }
                for _fut in as_completed(_futs):
                    para, passed, layer = _fut.result()
                    if passed and para:
                        best_paraphrase = para
                        passing_paraphrases.append(para)
                    else:
                        drifted += 1
                        if layer:
                            drift_layers[layer] += 1


            s.drift_rate = drifted / paraphrases_per_seed
            s.drift_details = {
                "total_paraphrases": paraphrases_per_seed,
                "drifted": drifted,
                "passed": paraphrases_per_seed - drifted,
                "by_layer": drift_layers,
            }

            # Write passing paraphrases to sidecar JSONL.
            # Buffered per seed.
            if _jsonl_fh is not None and passing_paraphrases:
                from datetime import datetime, timezone
                fam_tag   = f"semantic_{fam}_phase6"
                for para_text in passing_paraphrases:
                    record = {
                        "prompt":                 para_text,
                        "category":               "semantic_paraphrase",
                        "paraphrase_family":       fam_tag,
                        "semantic_flip":           False,
                        "intent_drift":            False,
                        "drift_reason":            None,
                        "safety_label":            "unsafe",
                        "expected_policy_action":  "refuse",
                        "domain":                  "general",
                        "difficulty":              "hard",   # generation schedule difficulty
                        "word_count":              len(para_text.split()),
                        "length_class":            (
                            "short"  if len(para_text.split()) <= 50  else
                            "long"   if len(para_text.split()) >= 150 else
                            "medium"
                        ),
                        "seed_text":               s.text[:200],  # truncated for reference
                        # Provenance
                        "llm_generated":           True,
                        "model":                   llm.model,
                        "pipeline_version":        "v9-phase6",
                        "source":                  "phase6_drift_check",
                        "generated_at":            datetime.now(timezone.utc).isoformat(),
                    }
                    _jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                _jsonl_fh.flush()  # flush per seed — safe on KeyboardInterrupt
                _saved_total += len(passing_paraphrases)

            checked_seeds.append(s)

            log.info(f"    [{seed_idx+1}/{len(family_seeds)}] drift={s.drift_rate:.0%} "
                     f"L1={drift_layers[1]} L2={drift_layers[2]} L3={drift_layers[3]} "
                     f"pass={len(passing_paraphrases)} saved "
                     f"| {s.text[:60]}...")

    # Rank: sort by drift rate ascending within each family, take top_k
    # Seeds from families NOT in active_families pass through untouched (drift_rate=-1)
    final: list[RawSeed] = []
    by_family_checked = defaultdict(list)
    for s in checked_seeds:
        by_family_checked[s.assigned_family].append(s)

    # Pass-through: seeds from families not run in this invocation
    passthrough = [s for s in seeds if s.assigned_family not in active_families]
    final.extend(passthrough)

    for fam in active_families:
        fam_seeds = by_family_checked.get(fam, [])
        fam_seeds.sort(key=lambda s: s.drift_rate)

        # Apply per-family drift threshold: exclude seeds above the family cap.
        # Research basis: direct_override indirect phrasing is inherently harder
        # to paraphrase faithfully (30% cap); hybrid_attacks must keep all
        # attack components intact (15% cap). See FAMILY_DRIFT_THRESHOLDS.
        fam_threshold = FAMILY_DRIFT_THRESHOLDS.get(fam, _DEFAULT_DRIFT_THRESHOLD)
        under_threshold = [s for s in fam_seeds if s.drift_rate <= fam_threshold]
        over_threshold_count = len(fam_seeds) - len(under_threshold)

        if not under_threshold:
            # Fallback: if ALL seeds exceed threshold, take the best ones anyway
            # (don't starve a family entirely — dataset balance matters)
            log.warning(
                f"  {fam}: ALL {len(fam_seeds)} seeds exceed drift threshold "
                f"{fam_threshold:.0%}. Using best available (drift floor fallback)."
            )
            under_threshold = fam_seeds  # take all, already sorted by rate
        elif over_threshold_count > 0:
            log.info(
                f"  {fam}: {over_threshold_count} seed(s) cut by "
                f"{fam_threshold:.0%} drift threshold"
            )

        top = under_threshold[:top_k]
        final.extend(top)
        if top:
            best_rate = top[0].drift_rate
            worst_rate = top[-1].drift_rate if len(top) > 1 else best_rate
            log.info(f"  {fam:30s}: top {len(top)} seeds, "
                     f"drift range {best_rate:.0%}-{worst_rate:.0%} "
                     f"(threshold {fam_threshold:.0%})")


    if _jsonl_fh is not None:
        _jsonl_fh.close()
        if jsonl_out is not None:
            log.info(f"Phase 6 sidecar: {_saved_total} passing paraphrases → {jsonl_out}")

    log.info(f"Phase 6 complete: {len(final)} quality-ranked seeds")
    return final


# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════

def write_output(seeds: list[RawSeed], output_path: Path):
    """Write mined seeds to JSON, grouped by family."""
    from collections import defaultdict

    by_family: dict[str, list[dict]] = defaultdict(list)
    for s in seeds:
        by_family[s.assigned_family].append({
            "text": s.text,
            "source": s.source,
            "source_id": s.source_id,
            "source_label": s.source_label,
            "word_count": s.word_count,
            "is_outlier": s.is_outlier,
            "centroid_distance": round(s.centroid_distance, 4),
            "drift_rate": round(s.drift_rate, 3) if s.drift_rate >= 0 else None,
        })

    output = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_seeds": len(seeds),
            "families": len(by_family),
            "sources": list(set(s.source for s in seeds)),
        },
        "families": {},
    }

    for fam in FAMILY_NAMES:
        fam_seeds = by_family.get(fam, [])
        output["families"][fam] = {
            "count": len(fam_seeds),
            "seeds": fam_seeds,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info(f"Output written to {output_path}")

    # Print summary report
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  mine_seeds.py — Seed Mining Report                            ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Total mined seeds     : {len(seeds):<6}                               ║")
    print(f"║  Families covered      : {len(by_family):<6}                               ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    for fam in FAMILY_NAMES:
        fam_seeds = by_family.get(fam, [])
        if fam_seeds:
            # Show drift info if available
            drifts = [s["drift_rate"] for s in fam_seeds if s["drift_rate"] is not None]
            if drifts:
                avg_drift = sum(drifts) / len(drifts)
                drift_str = f"drift={avg_drift:.0%}"
            else:
                drift_str = "no drift check"
            print(f"║  {fam:30s}: {len(fam_seeds):3d} seeds ({drift_str:>16s}) ║")
        else:
            print(f"║  {fam:30s}:   0 seeds                        ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Output → {str(output_path):<53}║")
    print("╚══════════════════════════════════════════════════════════════════╝")


# ═══════════════════════════════════════════════════════════════════════════════
# Load from existing JSON (for --from-json mode)
# ═══════════════════════════════════════════════════════════════════════════════

def load_from_json(input_path: Path) -> list[RawSeed]:
    """Load previously mined seeds from JSON back into RawSeed objects.

    Used with --from-json to skip Phases 1-5 and run only Phase 6 drift check.
    Preserves all existing metadata (source, centroid_distance, is_outlier, etc.).
    """
    log.info(f"Loading existing seeds from {input_path}...")
    with open(input_path) as f:
        data = json.load(f)

    seeds: list[RawSeed] = []
    for fam_name, fam_info in data["families"].items():
        for entry in fam_info.get("seeds", []):
            s = RawSeed(
                text=entry["text"],
                source=entry.get("source", "unknown"),
                source_id=entry.get("source_id", ""),
                source_label=entry.get("source_label", ""),
            )
            s.assigned_family = fam_name
            s.centroid_distance = entry.get("centroid_distance", 0.0)
            s.is_outlier = entry.get("is_outlier", False)
            # Preserve existing drift_rate if already checked
            existing_drift = entry.get("drift_rate")
            s.drift_rate = existing_drift if existing_drift is not None else -1.0
            # Preserve existing l4_score / difficulty if already computed
            existing_l4 = entry.get("l4_score")
            s.l4_score = existing_l4 if existing_l4 is not None else -1.0
            s.difficulty = entry.get("difficulty") or ""
            seeds.append(s)

    total = len(seeds)
    families = len(data["families"])
    log.info(f"  Loaded {total} seeds across {families} families from {input_path.name}")

    # Show which families already have drift data
    already_checked = sum(1 for s in seeds if s.drift_rate >= 0)
    if already_checked:
        log.info(f"  {already_checked}/{total} seeds already have drift_rate from previous runs")

    return seeds


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Mine adversarial seeds from real attacker datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (Phases 1-5, no LLM):
  python scripts/mine_seeds.py

  # Phase 6 on one family (uses existing output/mined_seeds.json):
  python scripts/mine_seeds.py --drift-check --from-json --family direct_override

  # Phase 6 on ALL families from existing JSON:
  python scripts/mine_seeds.py --drift-check --from-json

  # Full pipeline + Phase 6 in one shot:
  python scripts/mine_seeds.py --drift-check
"""
    )
    parser.add_argument("--drift-check", action="store_true",
                        help="Run Phase 6: LLM drift quality check (requires DeepSeek API key)")
    parser.add_argument("--from-json", action="store_true",
                        help="Skip Phases 1-5 and load existing output JSON (use with --drift-check)")
    parser.add_argument("--family", type=str, default=None,
                        choices=FAMILY_NAMES,
                        metavar="FAMILY",
                        help=("Run Phase 6 for this family only. All other seeds pass through. "
                              f"Valid: {', '.join(FAMILY_NAMES)}"))
    parser.add_argument("--top-k", type=int, default=50,
                        help="Seeds to keep per family after drift check (default: 50)")
    parser.add_argument("--candidates", type=int, default=50,
                        help="Candidate seeds per family before drift check (default: 50)")
    parser.add_argument("-o", "--output", type=str, default="output/mined_seeds.json",
                        help="Output file path (default: output/mined_seeds.json)")
    parser.add_argument("--paraphrases", type=int, default=20,
                        help="Paraphrases per seed in drift check (default: 20)")
    parser.add_argument("--phase6-out", type=str,
                        default="output/phase6_samples.jsonl",
                        help="JSONL file to save passing paraphrases from Phase 6 "
                             "(default: output/phase6_samples.jsonl). "
                             "Set to empty string to disable saving.")
    args = parser.parse_args()

    output_path = Path(args.output)

    # Validate --from-json
    if args.from_json and not args.drift_check:
        parser.error("--from-json requires --drift-check (nothing to do without Phase 6)")
    if args.from_json and not output_path.exists():
        parser.error(f"--from-json specified but {output_path} does not exist. "
                     "Run without --from-json first to generate it.")
    if args.family and not args.drift_check:
        log.warning("--family has no effect without --drift-check")

    t0 = time.time()

    if args.from_json:
        # Skip Phases 1-5: load existing JSON directly
        log.info("═══ Skipping Phases 1-5 (--from-json) ═══")
        candidates = load_from_json(output_path)
    else:
        # Phase 1: Ingest
        raw_seeds = phase_1_ingest()
        if not raw_seeds:
            log.error("No seeds ingested. Check dataset availability and network connection.")
            sys.exit(1)

        # Phase 2: Dedup
        deduped = phase_2_dedup(raw_seeds)

        # Phase 3: Pre-filter
        filtered = phase_3_prefilter(deduped)

        # Phase 4: Cluster
        clustered = phase_4_cluster(filtered)

        # Phase 5: Select candidates
        candidates = phase_5_select(clustered, candidates_per_family=args.candidates)

    # Phase 6: Drift check (optional, expensive)
    if args.drift_check:
        if args.family:
            log.info(f"Phase 6 restricted to family: {args.family}")
        phase6_out_path = Path(args.phase6_out) if args.phase6_out else None
        if phase6_out_path:
            phase6_out_path.parent.mkdir(parents=True, exist_ok=True)
            log.info(f"Phase 6 sidecar output: {phase6_out_path}")
        final = phase_6_drift_check(
            candidates,
            paraphrases_per_seed=args.paraphrases,
            top_k=args.top_k,
            only_family=args.family,
            jsonl_out=phase6_out_path,
        )
    else:
        final = candidates
        log.info("Skipping Phase 6 (use --drift-check to enable)")

    # Write output
    write_output(final, output_path)

    elapsed = time.time() - t0
    log.info(f"Pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
