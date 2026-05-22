"""
AttackEmbedder — embedding model singleton for drengr's scoring and clustering pipeline.

This module is the single source of truth for all sentence embedding in drengr.
It replaces three previously scattered SentenceTransformer instantiations:
  - Phase 2 (dedup, cosine > 0.95) in mine_seeds.py
  - Phase 4 (family clustering) in mine_seeds.py
  - L4 gate (now metadata enricher) previously in intent_verifier.py

Model: BAAI/bge-small-en-v1.5
  - 33M params, 384-dim, ~133MB
  - +2.1 MTEB STS points over all-MiniLM-L6-v2 (prev model)
  - Requires "Represent this sentence: " prefix on ALL inputs for peak STS accuracy
  - ~12ms CPU inference per pair; model is shared across all callers via singleton

Usage:
    from drengr.generators.embedder import AttackEmbedder

    embedder = AttackEmbedder()                        # lazy — model loads on first call
    score = embedder.score(seed, paraphrase)           # -1.0 if model unavailable
    tier  = embedder.difficulty(score)                 # "hard" | "medium" | "easy" | ""
    embs  = embedder.encode_batch(texts)               # np.ndarray (N, 384), normalized

Difficulty tiers (used for curriculum training and benchmark reporting):
    hard   — l4_score < 0.60: structurally transformed, indirect phrasing
    medium — l4_score 0.60-0.75: partial surface similarity preserved
    easy   — l4_score > 0.75: explicit surface paraphrase, near-verbatim structure

Calibration basis: sweep over 55 seed↔paraphrase pairs (same-family and cross-drift)
showed GOOD pair floor at 0.51 (explicit→indirect) and DRIFT ceiling at 0.73 (negation).
Hard threshold at 0.60 cleanly separates the indirect phrasing zone from refusal-adjacent
drift while capturing the most adversarially valuable samples for curriculum training.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ── Model configuration ────────────────────────────────────────────────────────

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_PREFIX = "Represent this sentence: "

# Difficulty tier thresholds (inclusive lower bound, exclusive upper bound)
HARD_THRESHOLD   = 0.60   # l4_score < 0.60  → "hard"
EASY_THRESHOLD   = 0.75   # l4_score > 0.75  → "easy"
                           # 0.60 ≤ score ≤ 0.75 → "medium"

# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[AttackEmbedder] = None


def get_embedder() -> AttackEmbedder:
    """Return the global AttackEmbedder singleton.

    Preferred over constructing AttackEmbedder() directly in scripts that
    call encode_batch() in multiple phases — avoids repeated model loads.
    """
    global _instance
    if _instance is None:
        _instance = AttackEmbedder()
    return _instance


# ── AttackEmbedder ─────────────────────────────────────────────────────────────

class AttackEmbedder:
    """Lazy-loaded BGE-small wrapper for attack scoring and clustering.

    Thread-safety: model is loaded once on first call. Subsequent calls share
    the loaded model. Not safe for concurrent writes to _model but reads are
    fine after initialization — in drengr's single-process pipeline this is
    sufficient.

    Fail-open design: if sentence-transformers is not installed or the model
    fails to load, all score() calls return -1.0 and difficulty() returns "".
    The pipeline continues without embedding-derived metadata. Install with:
        pip install sentence-transformers
    """

    def __init__(self) -> None:
        self._model = None  # lazy — loaded on first encode call

    # ── Public API ─────────────────────────────────────────────────────────────

    def score(self, seed: str, paraphrase: str) -> float:
        """Compute BGE-small cosine similarity between seed and paraphrase.

        Both inputs are prefixed with BGE_PREFIX before encoding, per the
        BAAI/bge-small-en-v1.5 model card's inference protocol for STS tasks.

        Returns:
            Cosine similarity in [0.0, 1.0] (embeddings are L2-normalized).
            Returns -1.0 if the model is unavailable — fail open, no filtering.
        """
        model = self._load()
        if model is None:
            return -1.0
        try:
            from sentence_transformers import util
            inputs = [BGE_PREFIX + seed, BGE_PREFIX + paraphrase]
            embs = model.encode(inputs, normalize_embeddings=True, show_progress_bar=False)
            return float(util.dot_score(embs[0], embs[1])[0][0])
        except Exception:
            return -1.0

    def difficulty(self, score: float) -> str:
        """Map an l4_score to a difficulty tier string.

        Args:
            score: Output of score(). Pass -1.0 if score was unavailable.

        Returns:
            "hard"   if score < HARD_THRESHOLD (0.60)
            "medium" if HARD_THRESHOLD ≤ score ≤ EASY_THRESHOLD (0.75)
            "easy"   if score > EASY_THRESHOLD (0.75)
            ""       if score == -1.0 (model unavailable, score not computed)
        """
        if score < 0:
            return ""
        if score < HARD_THRESHOLD:
            return "hard"
        if score > EASY_THRESHOLD:
            return "easy"
        return "medium"

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 128,
        show_progress_bar: bool = False,
        add_prefix: bool = False,
    ) -> Optional[np.ndarray]:
        """Encode a list of texts and return normalized embeddings.

        Args:
            texts: Input strings. For clustering/dedup (Phase 2/4), leave
                   add_prefix=False — BGE prefix is for STS tasks, not for
                   clustering where raw sentence representations are preferred.
            batch_size: Encoding batch size. 128 is fast on CPU for <5K texts.
            show_progress_bar: Pass True for long Phase 2/4 runs.
            add_prefix: If True, prepends BGE_PREFIX to each text (use for
                        STS scoring tasks only, not clustering).

        Returns:
            np.ndarray of shape (N, 384), L2-normalized, or None if unavailable.
        """
        model = self._load()
        if model is None:
            return None
        try:
            if add_prefix:
                texts = [BGE_PREFIX + t for t in texts]
            embs = model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
            )
            return np.array(embs)
        except Exception:
            return None

    # ── Internal ───────────────────────────────────────────────────────────────

    def _load(self):
        """Lazy-load the model. Returns None if sentence-transformers is absent."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL_NAME)
            return self._model
        except ImportError:
            return None
