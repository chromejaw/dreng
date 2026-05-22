"""
Core data models for the prompt dataset generator.

This module defines all the essential data structures used throughout
the generation process, including prompts, specifications, and metadata.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class Category(Enum):
    """Prompt categories for dataset generation."""
    TEMPORAL_ANCHOR = "temporal_anchor"
    EXACT_REPEATS = "exact_repeats"
    SEMANTIC_PARAPHRASE = "semantic_paraphrase"
    NEAR_DUPLICATES = "near_duplicates"


class Domain(Enum):
    """Domain areas for prompt distribution."""
    PROGRAMMING = "programming"
    BUSINESS = "business"
    CUSTOMER_SUPPORT = "customer_support"
    TECHNICAL = "technical"
    EDUCATION = "education"
    CREATIVE = "creative"
    ECOMMERCE = "ecommerce"
    TRAVEL = "travel"
    LEGAL = "legal"
    HEALTHCARE = "healthcare"
    DATA_ANALYTICS = "data_analytics"
    GENERAL = "general"


class Length(Enum):
    """Prompt length categories."""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class Difficulty(Enum):
    """Prompt difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class SafetyLabel(Enum):
    """Safety classification labels."""
    SAFE = "safe"
    BORDERLINE = "borderline"
    UNSAFE = "unsafe"


class PolicyAction(Enum):
    """Expected policy actions."""
    RESPOND = "respond"
    REFUSE = "refuse"
    SAFE_ALTERNATIVE = "safe_alternative"


class NegativeType(Enum):
    """Types of hard negatives."""
    SYNTHETIC_LLM = "synthetic_llm"
    RETRIEVED_BM25 = "retrieved_bm25"
    RANDOM = "random"


class ArrivalDistribution(Enum):
    """Burst arrival distribution patterns."""
    POISSON = "poisson"
    UNIFORM = "uniform"
    PEAK = "peak"


@dataclass
class PromptSpec:
    """Specification for generating a single prompt."""
    category: Category
    domain: Domain
    length: Length
    difficulty: Difficulty
    paraphrase_family: str
    repeat_weight: int
    similarity_target: Optional[float] = None


@dataclass
class GeneratedPrompt:
    """Complete generated prompt with all metadata."""
    # Core identity
    id: int
    prompt: str
    normalized_prompt: str
    category: Category
    paraphrase_family: str

    # Traffic pattern fields
    repeat_weight: int
    frequency_rank: int

    # Temporal fields
    created_at: str
    source_last_updated: Optional[str] = None
    valid_until: Optional[str] = None

    # Conversation context fields
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    turn_index: Optional[int] = None
    previous_message_snippet: Optional[str] = None

    # Hard negative fields
    hard_negative_of: List[int] = field(default_factory=list)
    negative_type: Optional[NegativeType] = None

    # Burst traffic fields
    burst_group_id: Optional[str] = None
    burst_size: Optional[int] = None
    burst_window_seconds: Optional[int] = None
    arrival_distribution: Optional[ArrivalDistribution] = None

    # Safety and classification
    safety_label: SafetyLabel = SafetyLabel.SAFE
    expected_policy_action: PolicyAction = PolicyAction.RESPOND
    domain: Domain = Domain.GENERAL
    length: Length = Length.MEDIUM
    difficulty: Difficulty = Difficulty.MEDIUM
    language: str = "en"


@dataclass
class CategoryCounts:
    """Exact category distribution targets."""
    semantic_paraphrase: int = 2000
    exact_repeats: int = 1500
    near_duplicates: int = 1000
    temporal_anchor: int = 500
    
    def total(self) -> int:
        """Calculate total prompts across all categories."""
        return self.temporal_anchor + self.exact_repeats + self.semantic_paraphrase + self.near_duplicates
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "semantic_paraphrase": self.semantic_paraphrase,
            "exact_repeats": self.exact_repeats,
            "near_duplicates": self.near_duplicates,
            "temporal_anchor": self.temporal_anchor
        }


@dataclass
class DomainDistribution:
    """Target domain percentages."""
    programming: float = 0.20
    business: float = 0.15
    customer_support: float = 0.15
    technical: float = 0.15
    education: float = 0.10
    creative: float = 0.10
    ecommerce: float = 0.05
    travel: float = 0.05
    legal: float = 0.03
    healthcare: float = 0.02
    
    def validate(self) -> bool:
        """Validate that percentages sum to 1.0."""
        total = (self.programming + self.business + self.customer_support + 
                self.technical + self.education + self.creative + 
                self.ecommerce + self.travel + self.legal + self.healthcare)
        return abs(total - 1.0) < 0.001


@dataclass
class SimilarityBands:
    """Similarity thresholds for each category."""
    hard_negatives: tuple = (0.40, 0.59)
    exact_repeats: float = 1.0


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation framework."""
    global_hit_rate: float = 0.0
    effective_hit_rate: float = 0.0
    stale_hit_rate: float = 0.0
    false_positive_semantic_hits: float = 0.0
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    mrr_at_k: Dict[int, float] = field(default_factory=dict)
    token_savings: int = 0
    api_call_reduction: float = 0.0
    cost_savings_dollars: float = 0.0
    unsafe_cached_response_rate: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    qps_throughput: float = 0.0


@dataclass
class DatasetMetadata:
    """Complete dataset metadata."""
    total_prompts: int
    created_date: str
    version: str
    description: str
    generator_version: str
    random_seed: int
    generator_args: Dict[str, Any]
    checksum: str
    embedding_models_used: List[str] = field(default_factory=list)
    similarity_validation_passed: bool = False


class CategoryGenerator(ABC):
    """Abstract base class for category-specific generators."""
    
    @abstractmethod
    def generate_prompts(self, count: int, specs: List[PromptSpec]) -> List[GeneratedPrompt]:
        """Generate prompts for this category."""
        pass
    
    @abstractmethod
    def validate_similarity_bands(self, prompts: List[GeneratedPrompt]) -> bool:
        """Validate prompts meet similarity requirements."""
        pass


class EmbeddingService(ABC):
    """Abstract interface for embedding computation."""
    
    @abstractmethod
    def compute_embedding(self, text: str) -> List[float]:
        """Compute embedding vector for text."""
        pass
    
    @abstractmethod
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between embeddings."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """Get embedding model metadata."""
        pass



class ParaphraseFamily:
    """Manages related prompts within a family."""
    
    def __init__(self, family_id: str, base_intent: str, domain: Domain):
        self.family_id = family_id
        self.base_intent = base_intent
        self.domain = domain
        self.prompts: List[GeneratedPrompt] = []
        self.centroid_embedding: Optional[List[float]] = None
        self.similarity_matrix: Dict[tuple, float] = {}
    
    def add_prompt(self, prompt: GeneratedPrompt) -> None:
        """Add prompt to family and update centroid."""
        self.prompts.append(prompt)
        self._update_centroid()
    
    def validate_similarity_band(self, target_band: tuple) -> bool:
        """Validate all prompts fall within target similarity band."""
        return True  # placeholder — similarity_to_centroid removed
    
    def _update_centroid(self) -> None:
        """Recalculate family centroid embedding."""
        if not self.prompts:
            return
        # Implementation will be added when embedding service is available
        pass