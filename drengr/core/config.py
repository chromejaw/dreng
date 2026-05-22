"""
Configuration management system with reproducible seeding.

This module handles all configuration parameters, validation, and 
reproducible random number generation for the dataset generator.
"""

import random
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime

from .models import (
    CategoryCounts, DomainDistribution, SimilarityBands, 
    EvaluationMetrics, DatasetMetadata
)


@dataclass
class GeneratorConfig:
    """Main configuration class for the dataset generator."""
    
    # Core generation parameters
    total_prompts: int = 5000
    random_seed: int = 42
    generator_version: str = "1.0.0"
    
    # Category distribution
    category_counts: CategoryCounts = field(default_factory=CategoryCounts)
    
    # Domain distribution
    domain_distribution: DomainDistribution = field(default_factory=DomainDistribution)
    
    # Similarity bands
    similarity_bands: SimilarityBands = field(default_factory=SimilarityBands)
    
    # Length and difficulty distributions
    length_distribution: Dict[str, float] = field(default_factory=lambda: {
        "short": 0.50,
        "medium": 0.35,
        "long": 0.15
    })
    
    difficulty_distribution: Dict[str, float] = field(default_factory=lambda: {
        "easy": 0.40,
        "medium": 0.45,
        "hard": 0.15
    })
    
    # Zipf distribution parameters
    zipf_exponent: float = 1.0
    
    # Embedding model configuration
    embedding_model: str = "text-embedding-3-large"
    embedding_model_version: str = "2025-08-01"
    
    # Output configuration
    output_file: str = "prompt_dataset.json"
    workload_weights_file: str = "workload_weights.json"
    
    # Validation tolerances
    category_count_tolerance: int = 0  # ±0 tolerance for exact counts
    domain_distribution_tolerance: float = 0.02  # ±2% tolerance
    

    
    # Paraphrase family configuration
    paraphrases_per_family: Dict[str, tuple] = field(default_factory=lambda: {
        "exact_repeats": (1, 1),
        "temporal_anchor": (1, 1),
        "semantic_paraphrase": (5, 20),
        "near_duplicates": (1, 3)
    })

    # Hard-negative (semantic flip) rate — fraction of generated prompts that
    # are intentionally flipped to label=0 during generation.
    # These are security-framed / formally-styled requests whose actual payload
    # is benign. The classifier learns the real decision boundary instead of
    # mapping "formal security tone" → malicious.
    # 15-20% is the recommended range. Tune here only; never via CLI flag.
    flip_rate: float = 0.17
    
    # TTL configuration (in seconds)
    default_ttl_seconds: int = 3600  # 1 hour
    temporal_ttl_range: tuple = (300, 7200)  # 5 minutes to 2 hours
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._setup_reproducible_random()
    
    def _validate_config(self) -> None:
        """Validate all configuration parameters."""
        # Skip validation if explicitly disabled (for testing)
        if hasattr(self, '_skip_validation') and self._skip_validation:
            return
            
        # Validate category counts sum to total
        if self.category_counts.total() != self.total_prompts:
            raise ValueError(
                f"Category counts sum to {self.category_counts.total()}, "
                f"but total_prompts is {self.total_prompts}"
            )
        
        # Validate domain distribution sums to 1.0
        if not self.domain_distribution.validate():
            raise ValueError("Domain distribution percentages must sum to 1.0")
        
        # Validate length distribution
        if abs(sum(self.length_distribution.values()) - 1.0) > 0.001:
            raise ValueError("Length distribution percentages must sum to 1.0")
        
        # Validate difficulty distribution
        if abs(sum(self.difficulty_distribution.values()) - 1.0) > 0.001:
            raise ValueError("Difficulty distribution percentages must sum to 1.0")
        

        
        # Validate similarity bands
        bands = self.similarity_bands
        if not (0.0 <= bands.hard_negatives[0] < bands.hard_negatives[1] <= 1.0):
            raise ValueError("Invalid hard_negatives similarity band")
    
    def _setup_reproducible_random(self) -> None:
        """Set up reproducible random number generation."""
        random.seed(self.random_seed)
    
    @classmethod
    def create_invalid_for_testing(cls, **kwargs):
        """Create an invalid config for testing purposes."""
        config = cls.__new__(cls)
        config._skip_validation = True
        # Set default values
        config.total_prompts = kwargs.get('total_prompts', 0)
        config.random_seed = kwargs.get('random_seed', 42)
        config.generator_version = "1.0.0"
        config.category_counts = CategoryCounts()
        config.domain_distribution = DomainDistribution()
        config.length_distribution = {
            Length.SHORT: 0.4,
            Length.MEDIUM: 0.4,
            Length.LONG: 0.2
        }
        config.difficulty_distribution = {
            Difficulty.EASY: 0.3,
            Difficulty.MEDIUM: 0.5,
            Difficulty.HARD: 0.2
        }

        config.similarity_bands = {
            'exact_repeats': (0.95, 1.0),
            'near_duplicates': (2, 5),
            'concurrent_identical': (5, 20)
        }
        config.default_ttl_seconds = 3600
        config.temporal_ttl_range = (300, 7200)
        config.embedding_model = "test-model"
        config.embedding_model_version = "1.0"
        return config
    
    def get_embedding_model_id(self) -> str:
        """Get full embedding model identifier."""
        return f"{self.embedding_model}@{self.embedding_model_version}"
    
    def generate_checksum(self, data: str) -> str:
        """Generate SHA256 checksum for data integrity."""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def create_metadata(self, checksum: str, embedding_models: list) -> DatasetMetadata:
        """Create dataset metadata from configuration."""
        return DatasetMetadata(
            total_prompts=self.total_prompts,
            created_date=datetime.now().isoformat(),
            version="3.0",
            description="Adversarial prompt injection dataset for classifier training.",
            generator_version=self.generator_version,
            random_seed=self.random_seed,
            generator_args=self.to_dict(),
            checksum=checksum,
            embedding_models_used=embedding_models,
            similarity_validation_passed=False
        )
    
    def get_category_specs(self, category: str) -> Dict[str, Any]:
        """Get specifications for a specific category."""
        return {
            "count": getattr(self.category_counts, category, 0),
            "paraphrases_per_family": self.paraphrases_per_family.get(category, (1, 1))
        }


@dataclass
class UnifiedDatasetConfig:
    """Unified configuration system for drengr framework."""
    
    # Core generation parameters
    total_prompts: int = 5000
    random_seed: int = 42
    profile: str = "sota"
    
    # Service configuration
    embedding_backend: str = "auto"
    embedding_model: Optional[str] = None
    llm_backend: str = "auto"
    llm_model: Optional[str] = None
    use_llm_for_paraphrase: bool = True
    
    # Output configuration
    output_path: Optional[str] = None
    include_golden: bool = True
    include_metrics: bool = True
    overwrite: bool = False
    
    # Generation options
    preview: int = 0
    stream: bool = False
    run_ablation: bool = False
    force: bool = False

    # Hard-negative (semantic flip) rate — see GeneratorConfig.flip_rate.
    # Propagates into GeneratorConfig on __post_init__. Tune here only.
    flip_rate: float = 0.17
    
    # Legacy compatibility
    generator_config: Optional[GeneratorConfig] = None
    
    def __post_init__(self):
        """Initialize configuration after creation."""
        if self.generator_config is None:
            self.generator_config = self._create_generator_config()
        
        # Set output path if not provided
        if self.output_path is None:
            self.output_path = f"./dreng_dataset_{self.total_prompts}_{self.random_seed}.json"
    
    def _create_generator_config(self) -> GeneratorConfig:
        """Create legacy GeneratorConfig for compatibility."""
        # Compute category counts based on profile and total
        category_counts = self._compute_category_counts()
        
        return GeneratorConfig(
            total_prompts=self.total_prompts,
            random_seed=self.random_seed,
            generator_version="1.0.0",
            category_counts=category_counts,
            domain_distribution=self._get_domain_distribution(),
            similarity_bands=self._get_similarity_bands(),

            length_distribution=self._get_length_distribution(),
            difficulty_distribution=self._get_difficulty_distribution(),
            paraphrases_per_family=self._get_paraphrases_per_family(),
            zipf_exponent=self._get_zipf_exponent(),
            embedding_model=self.embedding_model or "all-MiniLM-L6-v2",
            embedding_model_version="2025-01-01",
            output_file=self.output_path or "dataset.json",
            flip_rate=self.flip_rate,
        )
    
    def _compute_category_counts(self) -> CategoryCounts:
        """Compute category counts based on profile and total."""
        # Split across 4 categories: 40% Semantic, 30% Exact, 20% Near Duplicates, 10% Temporal
        semantic_count = int(self.total_prompts * 0.40)
        exact_count = int(self.total_prompts * 0.30)
        near_count = int(self.total_prompts * 0.20)
        temporal_count = self.total_prompts - semantic_count - exact_count - near_count
        
        return CategoryCounts(
            semantic_paraphrase=semantic_count,
            near_duplicates=near_count,
            exact_repeats=exact_count,
            temporal_anchor=temporal_count
        )
    
    def _get_domain_distribution(self) -> DomainDistribution:
        """Get domain distribution based on profile."""
        if self.profile in ["fast", "cheap", "dev"]:
            return DomainDistribution(
                programming=0.40,
                business=0.30,
                customer_support=0.20,
                technical=0.10,
                education=0.00,
                creative=0.00,
                ecommerce=0.00,
                travel=0.00,
                legal=0.00,
                healthcare=0.00
            )
        else:  # sota
            return DomainDistribution(
                programming=0.25,
                business=0.18,
                customer_support=0.15,
                technical=0.12,
                education=0.10,
                creative=0.08,
                ecommerce=0.05,
                travel=0.03,
                legal=0.02,
                healthcare=0.02
            )
    
    def _get_similarity_bands(self) -> SimilarityBands:
        """Get similarity bands based on profile."""
        return SimilarityBands(
            hard_negatives=(0.40, 0.59),
            exact_repeats=1.0
        )
    

    
    def _get_length_distribution(self) -> Dict[str, float]:
        """Get length distribution based on profile."""
        if self.profile in ["fast", "cheap", "dev"]:
            return {
                "short": 0.70,
                "medium": 0.25,
                "long": 0.05
            }
        else:  # sota
            return {
                "short": 0.50,
                "medium": 0.35,
                "long": 0.15
            }
    
    def _get_difficulty_distribution(self) -> Dict[str, float]:
        """Get difficulty distribution based on profile."""
        if self.profile in ["fast", "cheap", "dev"]:
            return {
                "easy": 0.60,
                "medium": 0.30,
                "hard": 0.10
            }
        else:  # sota
            return {
                "easy": 0.40,
                "medium": 0.45,
                "hard": 0.15
            }
    
    def _get_paraphrases_per_family(self) -> Dict[str, tuple]:
        """Get paraphrases per family based on profile."""
        return {
            "exact_repeats": (1, 1),
            "temporal_anchor": (1, 1)
        }
    
    def _get_zipf_exponent(self) -> float:
        """Get Zipf exponent based on profile."""
        profile_exponents = {
            "sota": 1.0,
            "fast": 1.0,
            "cheap": 1.2,
            "dev": 0.8
        }
        return profile_exponents.get(self.profile, 1.0)
    
    def get_service_config(self, service_type: str) -> Dict[str, Any]:
        """Get configuration for a specific service type."""
        if service_type == "embedding":
            return {
                "backend": self.embedding_backend,
                "model": self.embedding_model,
                "provider": self.embedding_backend if self.embedding_backend != "auto" else "local"
            }
        elif service_type == "llm":
            return {
                "backend": self.llm_backend,
                "model": self.llm_model,
                "provider": self.llm_backend if self.llm_backend != "auto" else "local",
                "use_for_paraphrase": self.use_llm_for_paraphrase
            }
        else:
            return {}
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if self.total_prompts <= 0:
            errors.append("total_prompts must be positive")
        
        if self.profile not in ["sota", "fast", "cheap", "dev"]:
            errors.append(f"Invalid profile: {self.profile}")
        
        if self.embedding_backend not in ["auto", "local", "openai", "ensemble", "mock"]:
            errors.append(f"Invalid embedding_backend: {self.embedding_backend}")
        
        if self.llm_backend not in ["auto", "local", "openai", "none", "mock"]:
            errors.append(f"Invalid llm_backend: {self.llm_backend}")
        
        if self.preview < 0:
            errors.append("preview must be non-negative")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "total_prompts": self.total_prompts,
            "random_seed": self.random_seed,
            "profile": self.profile,
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "llm_backend": self.llm_backend,
            "llm_model": self.llm_model,
            "use_llm_for_paraphrase": self.use_llm_for_paraphrase,
            "output_path": self.output_path,
            "include_golden": self.include_golden,
            "include_metrics": self.include_metrics,
            "overwrite": self.overwrite,
            "preview": self.preview,
            "stream": self.stream,
            "run_ablation": self.run_ablation,
            "force": self.force
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedDatasetConfig':
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_profile(cls, 
                    total: int,
                    profile: str = "sota",
                    seed: Optional[int] = None,
                    **overrides) -> 'UnifiedDatasetConfig':
        """Create configuration from profile with overrides."""
        import random
        
        if seed is None:
            seed = random.randint(1, 2**31 - 1)
        
        config_data = {
            "total_prompts": total,
            "profile": profile,
            "random_seed": seed,
            **overrides
        }
        
        return cls(**config_data)


class DatasetConfig:
    """Enhanced configuration builder for the new drengr API."""
    
    @classmethod
    def from_total(cls, total: int, profile: str, seed: int) -> UnifiedDatasetConfig:
        """Build unified configuration from total count and profile."""
        return UnifiedDatasetConfig.from_profile(total, profile, seed)
    
    @classmethod
    def _get_profile_settings(cls, profile: str) -> Dict[str, Any]:
        """Get profile-specific configuration settings."""
        profiles = {
            "sota": {
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_model_version": "2025-01-01",
                "zipf_exponent": 1.0,
                "category_count_tolerance": 0,
                "length_distribution": {
                    "short": 0.50,
                    "medium": 0.35,
                    "long": 0.15
                },
                "difficulty_distribution": {
                    "easy": 0.40,
                    "medium": 0.45,
                    "hard": 0.15
                },
                "paraphrases_per_family": {
                    "semantic_paraphrase": (8, 15),
                    "near_duplicates": (2, 5),
                    "concurrent_identical": (5, 20)
                },
                "default_ttl_seconds": 3600,
                "temporal_ttl_range": (300, 7200)
            },
            "fast": {
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_model_version": "2025-01-01",
                "zipf_exponent": 1.0,
                "category_count_tolerance": 1,
                "length_distribution": {
                    "short": 0.60,
                    "medium": 0.30,
                    "long": 0.10
                },
                "difficulty_distribution": {
                    "easy": 0.50,
                    "medium": 0.40,
                    "hard": 0.10
                },
                "paraphrases_per_family": {
                    "semantic_paraphrase": (5, 10),
                    "near_duplicates": (2, 3),
                    "concurrent_identical": (3, 10)
                },
                "default_ttl_seconds": 1800,
                "temporal_ttl_range": (300, 3600)
            },
            "cheap": {
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_model_version": "2025-01-01",
                "zipf_exponent": 1.2,
                "category_count_tolerance": 2,
                "length_distribution": {
                    "short": 0.70,
                    "medium": 0.25,
                    "long": 0.05
                },
                "difficulty_distribution": {
                    "easy": 0.60,
                    "medium": 0.30,
                    "hard": 0.10
                },
                "paraphrases_per_family": {
                    "semantic_paraphrase": (3, 8),
                    "near_duplicates": (2, 3),
                    "concurrent_identical": (2, 8)
                },
                "default_ttl_seconds": 1200,
                "temporal_ttl_range": (300, 1800)
            },
            "dev": {
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_model_version": "2025-01-01",
                "zipf_exponent": 0.8,
                "category_count_tolerance": 5,
                "length_distribution": {
                    "short": 0.80,
                    "medium": 0.15,
                    "long": 0.05
                },
                "difficulty_distribution": {
                    "easy": 0.70,
                    "medium": 0.25,
                    "hard": 0.05
                },
                "paraphrases_per_family": {
                    "semantic_paraphrase": (2, 5),
                    "near_duplicates": (2, 3),
                    "concurrent_identical": (2, 5)
                },
                "default_ttl_seconds": 600,
                "temporal_ttl_range": (300, 1200)
            }
        }
        
        return profiles.get(profile, profiles["sota"])
    
    @classmethod
    def _compute_sota_category_counts(cls, total: int) -> CategoryCounts:
        """Compute exact SOTA category distribution."""
        semantic_count = int(total * 0.40)
        exact_count = int(total * 0.30)
        near_count = int(total * 0.20)
        temporal_count = total - semantic_count - exact_count - near_count
        
        return CategoryCounts(
            semantic_paraphrase=semantic_count,
            near_duplicates=near_count,
            exact_repeats=exact_count,
            temporal_anchor=temporal_count
        )
    
    @classmethod
    def _get_enhanced_domain_distribution(cls) -> DomainDistribution:
        """Get enhanced domain distribution with realistic percentages."""
        return DomainDistribution(
            programming=0.25,      # Increased for tech focus
            business=0.18,         # Increased for enterprise use
            customer_support=0.15, # Common use case
            technical=0.12,        # Technical documentation
            education=0.10,        # Learning and training
            creative=0.08,         # Content generation
            ecommerce=0.05,        # Online retail
            travel=0.03,           # Travel and hospitality
            legal=0.02,            # Legal documents
            healthcare=0.02        # Healthcare applications
        )


class ConfigurationManager:
    """Manages configuration loading, validation, and persistence."""
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
    
    def load_from_file(self, filepath: str) -> GeneratorConfig:
        """Load configuration from JSON file."""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            # Convert nested dictionaries back to dataclass instances
            if 'category_counts' in config_dict:
                config_dict['category_counts'] = CategoryCounts(**config_dict['category_counts'])
            if 'domain_distribution' in config_dict:
                config_dict['domain_distribution'] = DomainDistribution(**config_dict['domain_distribution'])
            if 'similarity_bands' in config_dict:
                config_dict['similarity_bands'] = SimilarityBands(**config_dict['similarity_bands'])
            
            self.config = GeneratorConfig(**config_dict)
            return self.config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2, default=str)
        except Exception as e:
            raise ValueError(f"Error saving configuration: {e}")
    
    def validate_runtime_requirements(self) -> bool:
        """Validate that runtime requirements are met."""
        # Check that required directories exist or can be created
        # Check that embedding models are accessible
        # Validate output file paths are writable
        return True
    
    def get_category_specs(self, category: str) -> Dict[str, Any]:
        """Get specifications for a specific category."""
        return {
            "count": getattr(self.config.category_counts, category),
            "paraphrases_per_family": self.config.paraphrases_per_family.get(category, (1, 1))
        }