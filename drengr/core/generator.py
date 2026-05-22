"""
Main prompt generator engine that orchestrates the entire generation process.

This module coordinates all category generators, validation, and output
generation to produce the complete 5,000 prompt dataset.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import (
    CategoryGenerator, EmbeddingService,
    PromptSpec, GeneratedPrompt, CategoryCounts, DomainDistribution,
    SimilarityBands, Category, Domain, Length, Difficulty
)
from .config import GeneratorConfig, ConfigurationManager, DatasetConfig, UnifiedDatasetConfig
from .exceptions import DrengrError, DatasetGenerationError, CategoryCountError, ValidationError, GenerationError, ConfigurationError, BackendError


class DatasetGenerator:
    """High-level dataset generator for integration tests and external API."""
    
    def __init__(self, config):
        """Initialize dataset generator with configuration."""
        self.config = config
        self.engine = PromptGeneratorEngine(config)
    
    def generate_category_prompts(self, category: Category, count: int) -> List[GeneratedPrompt]:
        """Generate prompts for a specific category."""
        # Create a spec for this category
        spec = PromptSpec(
            category=category,
            domain=Domain.GENERAL,
            length=Length.MEDIUM,
            difficulty=Difficulty.MEDIUM,
            paraphrase_family="default_family",
            repeat_weight=1
        )
        
        # Generate prompts using the engine
        return self.engine.generate_prompts_for_category(category, count, spec)
    

    
    def generate_dataset(self) -> Dict[str, Any]:
        """Generate complete dataset."""
        return self.engine.generate_dataset()


class PromptGeneratorEngine:
    """Main engine that orchestrates the prompt generation process."""
    
    def __init__(self,
                 config: Optional[GeneratorConfig] = None,
                 embedding_service: Optional[EmbeddingService] = None):
        """Initialize the generator engine."""
        self.config = config or GeneratorConfig()
        self.embedding_service = embedding_service
        
        # Initialize category generators (will be populated as they're implemented)
        self.category_generators: Dict[Category, CategoryGenerator] = {}
        
        # Track generation state
        self.generated_prompts: List[GeneratedPrompt] = []
        self.family_registry: Dict[str, List[GeneratedPrompt]] = {}
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info(f"Initialized PromptGeneratorEngine with seed {self.config.random_seed}")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def register_category_generator(self, category: Category, generator: CategoryGenerator) -> None:
        """Register a category-specific generator."""
        self.category_generators[category] = generator
        self.logger.info(f"Registered generator for category: {category.value}")
    
    def generate_prompts_for_category(self, category: Category, count: int, spec: PromptSpec) -> List[GeneratedPrompt]:
        """Generate prompts for a specific category."""
        if category not in self.category_generators:
            # Create a simple mock generator for testing
            from datetime import datetime, timezone
            prompts = []
            for i in range(count):
                prompt = GeneratedPrompt(
                    id=i + 1,
                    prompt=f"Mock {category.value} prompt {i + 1}",
                    normalized_prompt=f"mock {category.value} prompt {i + 1}",
                    category=category,
                    paraphrase_family=spec.paraphrase_family,
                    repeat_weight=spec.repeat_weight,
                    frequency_rank=i + 1,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    domain=spec.domain,
                    length=spec.length,
                    difficulty=spec.difficulty,
                    language="en"
                )
                prompts.append(prompt)
            return prompts
        
        # Use registered generator
        generator = self.category_generators[category]
        return generator.generate_prompts(spec, count)
    
    def generate_dataset(self) -> Dict[str, Any]:
        """Generate the complete dataset with all 5,000 prompts."""
        self.logger.info("Starting dataset generation...")
        
        try:
            # Validate prerequisites
            self._validate_prerequisites()
            
            # Generate prompts for each category
            self._generate_all_categories()
            
            # Validate generated dataset
            self._validate_dataset()
            
            # Assemble final dataset
            dataset = self._assemble_dataset()
            
            # Generate workload weights
            self._generate_workload_weights()
            
            self.logger.info(f"Successfully generated dataset with {len(self.generated_prompts)} prompts")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Dataset generation failed: {e}")
            raise DatasetGenerationError(f"Generation failed: {e}") from e
    
    def _validate_prerequisites(self) -> None:
        """Validate that all required components are available."""
        if not self.embedding_service:
            raise DatasetGenerationError("EmbeddingService is required but not provided")
        
        # Golden response generator is removed — not needed.
        
        # Check that all category generators are registered
        required_categories = [
            Category.SEMANTIC_PARAPHRASE,
            Category.EXACT_REPEATS,
            Category.NEAR_DUPLICATES,
            Category.TEMPORAL_ANCHOR
        ]
        
        missing_generators = [
            cat for cat in required_categories 
            if cat not in self.category_generators
        ]
        
        if missing_generators:
            raise DatasetGenerationError(
                f"Missing category generators: {[cat.value for cat in missing_generators]}"
            )
    
    def _generate_all_categories(self) -> None:
        """Generate prompts for all categories."""
        category_counts = self.config.category_counts
        
        # Generate in order of complexity (simple to complex)
        generation_order = [
            (Category.EXACT_REPEATS, category_counts.exact_repeats),
            (Category.NEAR_DUPLICATES, category_counts.near_duplicates),
            (Category.SEMANTIC_PARAPHRASE, category_counts.semantic_paraphrase),
            (Category.TEMPORAL_ANCHOR, category_counts.temporal_anchor)
        ]
        
        for category, count in generation_order:
            if count <= 0:
                self.logger.info(f"Skipping category {category.value} (count: {count})")
                continue
                
            self.logger.info(f"Generating {count} prompts for category: {category.value}")
            
            # Generate prompt specifications
            specs = self._generate_prompt_specs(category, count)
            
            # Generate prompts using category-specific generator
            generator = self.category_generators[category]
            prompts = generator.generate_prompts(count, specs)
            
            # Validate category-specific requirements
            if not generator.validate_similarity_bands(prompts):
                raise ValidationError(f"Similarity band validation failed for {category.value}")
            
            # Assign sequential IDs
            for i, prompt in enumerate(prompts):
                prompt.id = len(self.generated_prompts) + i + 1
            
            # Add to generated prompts
            self.generated_prompts.extend(prompts)
            
            # Update family registry
            self._update_family_registry(prompts)
            
            self.logger.info(f"Successfully generated {len(prompts)} prompts for {category.value}")
    
    def _generate_prompt_specs(self, category: Category, count: int) -> List[PromptSpec]:
        """Generate prompt specifications for a category."""
        specs = []
        
        # Get category-specific configuration
        category_config = self.config.get_category_specs(category.value)
        
        # Calculate domain distribution for this category
        domain_counts = self._calculate_domain_distribution(count)
        
        # Calculate length and difficulty distributions
        length_counts = self._calculate_length_distribution(count)
        difficulty_counts = self._calculate_difficulty_distribution(count)
        
        # Generate specs
        spec_id = 0
        for domain, domain_count in domain_counts.items():
            for length, length_count in length_counts.items():
                for difficulty, difficulty_count in difficulty_counts.items():
                    # Calculate how many specs for this combination
                    combo_count = int(domain_count * length_count * difficulty_count / count)
                    if combo_count == 0:
                        continue
                    
                    for _ in range(combo_count):
                        # Generate paraphrase family ID
                        family_id = f"{category.value}_{domain.value}_{spec_id}"
                        
                        # Calculate repeat weight (will be refined by Zipf distribution)
                        repeat_weight = 1
                        
                        spec = PromptSpec(
                            category=category,
                            domain=domain,
                            length=length,
                            difficulty=difficulty,
                            paraphrase_family=family_id,
                            repeat_weight=repeat_weight
                        )
                        specs.append(spec)
                        spec_id += 1
        
        # Ensure we have exactly the right count
        while len(specs) < count:
            specs.append(PromptSpec(
                category=category,
                domain=Domain.GENERAL,
                length=Length.MEDIUM,
                difficulty=Difficulty.MEDIUM,
                paraphrase_family=f"family_{category.value}_{len(specs)}",
                repeat_weight=1
            ))
        
        return specs[:count]
    
    def _calculate_domain_distribution(self, count: int) -> Dict[Domain, int]:
        """Calculate domain distribution for given count."""
        dist = self.config.domain_distribution
        return {
            Domain.PROGRAMMING: int(count * dist.programming),
            Domain.BUSINESS: int(count * dist.business),
            Domain.CUSTOMER_SUPPORT: int(count * dist.customer_support),
            Domain.TECHNICAL: int(count * dist.technical),
            Domain.EDUCATION: int(count * dist.education),
            Domain.CREATIVE: int(count * dist.creative),
            Domain.ECOMMERCE: int(count * dist.ecommerce),
            Domain.TRAVEL: int(count * dist.travel),
            Domain.LEGAL: int(count * dist.legal),
            Domain.HEALTHCARE: int(count * dist.healthcare)
        }
    
    def _calculate_length_distribution(self, count: int) -> Dict[Length, int]:
        """Calculate length distribution for given count."""
        dist = self.config.length_distribution
        return {
            Length.SHORT: int(count * dist["short"]),
            Length.MEDIUM: int(count * dist["medium"]),
            Length.LONG: int(count * dist["long"])
        }
    
    def _calculate_difficulty_distribution(self, count: int) -> Dict[Difficulty, int]:
        """Calculate difficulty distribution for given count."""
        dist = self.config.difficulty_distribution
        return {
            Difficulty.EASY: int(count * dist["easy"]),
            Difficulty.MEDIUM: int(count * dist["medium"]),
            Difficulty.HARD: int(count * dist["hard"])
        }
    

    
    def _update_family_registry(self, prompts: List[GeneratedPrompt]) -> None:
        """Update the family registry with new prompts."""
        for prompt in prompts:
            family_id = prompt.paraphrase_family
            if family_id not in self.family_registry:
                self.family_registry[family_id] = []
            self.family_registry[family_id].append(prompt)
    
    def _validate_dataset(self) -> None:
        """Validate the complete generated dataset."""
        self.logger.info("Validating generated dataset...")
        
        # Validate total count
        if len(self.generated_prompts) != self.config.total_prompts:
            raise CategoryCountError(
                f"Generated {len(self.generated_prompts)} prompts, "
                f"expected {self.config.total_prompts}"
            )
        
        # Validate category counts
        category_counts = {}
        for prompt in self.generated_prompts:
            category = prompt.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        expected_counts = self.config.category_counts.to_dict()
        for category, expected_count in expected_counts.items():
            actual_count = category_counts.get(category, 0)
            if abs(actual_count - expected_count) > self.config.category_count_tolerance:
                raise CategoryCountError(
                    f"Category {category}: expected {expected_count}, "
                    f"got {actual_count}"
                )
        
        # Validate unique IDs
        ids = [prompt.id for prompt in self.generated_prompts]
        if len(set(ids)) != len(ids):
            raise ValidationError("Duplicate prompt IDs found")
        
        # Validate ID sequence
        expected_ids = list(range(1, self.config.total_prompts + 1))
        if sorted(ids) != expected_ids:
            raise ValidationError("Prompt IDs are not sequential from 1 to 5000")
        
        self.logger.info("Dataset validation passed")
    
    def _assemble_dataset(self) -> Dict[str, Any]:
        """Assemble the final dataset structure."""
        # Sort prompts by ID
        sorted_prompts = sorted(self.generated_prompts, key=lambda p: p.id)
        
        # Convert prompts to dictionaries
        prompts_data = []
        for prompt in sorted_prompts:
            prompt_dict = {
                "id": prompt.id,
                "prompt": prompt.prompt,
                "normalized_prompt": prompt.normalized_prompt,
                "category": prompt.category.value,
                "paraphrase_family": prompt.paraphrase_family,
                "repeat_weight": prompt.repeat_weight,
                "frequency_rank": prompt.frequency_rank,
                "created_at": prompt.created_at,
                "source_last_updated": prompt.source_last_updated,
                "valid_until": prompt.valid_until,
                "session_id": prompt.session_id,
                "user_id": prompt.user_id,
                "turn_index": prompt.turn_index,
                "previous_message_snippet": prompt.previous_message_snippet,
                "hard_negative_of": prompt.hard_negative_of,
                "negative_type": prompt.negative_type.value if prompt.negative_type else None,
                "burst_group_id": prompt.burst_group_id,
                "burst_size": prompt.burst_size,
                "burst_window_seconds": prompt.burst_window_seconds,
                "arrival_distribution": prompt.arrival_distribution.value if prompt.arrival_distribution else None,
                "safety_label": prompt.safety_label.value,
                "expected_policy_action": prompt.expected_policy_action.value,
                "domain": prompt.domain.value,
                "length": prompt.length.value,
                "difficulty": prompt.difficulty.value,
                "language": prompt.language
            }
            prompts_data.append(prompt_dict)
        
        # Create dataset structure
        dataset_json = json.dumps({
            "metadata": {},  # Will be filled by create_metadata
            "categories": self.config.category_counts.to_dict(),
            "prompts": prompts_data
        }, indent=2)
        
        # Generate checksum
        checksum = self.config.generate_checksum(dataset_json)
        
        # Create metadata
        embedding_models = [self.config.get_embedding_model_id()]
        metadata = self.config.create_metadata(checksum, embedding_models)
        
        # Final dataset
        dataset = {
            "metadata": {
                "total_prompts": metadata.total_prompts,
                "created_date": metadata.created_date,
                "version": metadata.version,
                "description": metadata.description,
                "generator_version": metadata.generator_version,
                "random_seed": metadata.random_seed,
                "generator_args": metadata.generator_args,
                "checksum": metadata.checksum,
                "embedding_models_used": metadata.embedding_models_used,
                "similarity_validation_passed": metadata.similarity_validation_passed
            },
            "categories": self.config.category_counts.to_dict(),
            "prompts": prompts_data
        }
        
        return dataset
    
    def _generate_workload_weights(self) -> None:
        """Generate workload weights file for traffic simulation."""
        # This will be implemented when Zipf distribution is available
        workload_weights = {
            "zipf_exponent": self.config.zipf_exponent,
            "weights": {}
        }
        
        # Save workload weights
        with open(self.config.workload_weights_file, 'w') as f:
            json.dump(workload_weights, f, indent=2)
        
        self.logger.info(f"Generated workload weights file: {self.config.workload_weights_file}")
    
    def save_dataset(self, dataset: Dict[str, Any], filepath: Optional[str] = None) -> None:
        """Save the dataset to a JSON file."""
        output_file = filepath or self.config.output_file
        
        try:
            with open(output_file, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            self.logger.info(f"Dataset saved to: {output_file}")
            
        except Exception as e:
            raise DatasetGenerationError(f"Failed to save dataset: {e}") from e


def generate_dataset_with_new_api(
    total: int,
    output_path: Optional[str] = None,
    profile: str = "sota",
    seed: Optional[int] = None,
    embedding_backend: str = "auto",
    embedding_model: Optional[str] = None,
    use_llm_for_paraphrase: bool = True,
    llm_backend: str = "auto",
    include_golden: bool = True,
    preview: int = 0,
    stream: bool = False,
    overwrite: bool = False,
    run_ablation: bool = False,
    force: bool = False,
) -> str:
    """Generate dataset using the complete drengr API."""
    import os
    import time
    from pathlib import Path
    from datetime import datetime
    from ..services.service_factory import get_service_factory
    
    start_time = time.time()
    
    # Set up logging for progress
    if stream:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting drengr dataset generation...")
    
    # Set up configuration
    if seed is None:
        import random
        seed = random.randint(1, 2**31 - 1)
    
    try:
        # Create unified configuration with error handling
        try:
            config = UnifiedDatasetConfig.from_profile(
                total=total,
                profile=profile,
                seed=seed,
                embedding_backend=embedding_backend,
                embedding_model=embedding_model,
                llm_backend=llm_backend,
                use_llm_for_paraphrase=use_llm_for_paraphrase,
                output_path=output_path,
                include_golden=include_golden,
                preview=preview,
                stream=stream,
                overwrite=overwrite,
                run_ablation=run_ablation,
                force=force
            )
        except ConfigurationError as e:
            raise GenerationError(str(e)) from e
        except Exception as e:
            raise GenerationError(f"Failed to create configuration: {e}") from e
        
        # Validate configuration
        try:
            config_errors = config.validate()
            if config_errors and not force:
                raise ConfigurationError(f"Configuration validation failed: {'; '.join(config_errors)}")
        except ConfigurationError as e:
            raise GenerationError(str(e)) from e
        
        # Set up output path
        output_path = Path(config.output_path).resolve()
        
        # Check if output exists and handle overwrite
        if output_path.exists() and not overwrite:
            raise GenerationError(
                f"Output file {output_path} already exists. Use overwrite=True to replace it."
            )
        
        if stream:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Configuration validated (10%)")
        
        # Create service container with error handling
        try:
            service_factory = get_service_factory()
            services = service_factory.create_optimized_container(total, profile)
            
            # Validate service health
            health_status = service_factory.validate_service_health(services)
            if not all(health_status.values()) and not force:
                unhealthy_services = [k for k, v in health_status.items() if not v]
                raise BackendError(f"Unhealthy services detected: {unhealthy_services}. Use force=True to continue.")
                
        except BackendError:
            raise  # Re-raise backend errors
        except Exception as e:
            if force:
                # Create fallback services
                from ..services.embedding import MockEmbeddingService
                from ..services.llm import MockLLMService
                from ..services.service_factory import ServiceContainer
                
                services = ServiceContainer(
                    embedding_service=MockEmbeddingService(),
                    llm_service=MockLLMService()
                )
                if stream:
                    print(f"Warning: Using fallback services due to error: {e}")
            else:
                raise BackendError(f"Failed to initialize services: {e}") from e
        
        if stream:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Services initialized (20%)")
        
        # Initialize generation engine with error handling
        try:
            generator_engine = PromptGeneratorEngine(
                config=config.generator_config,
                embedding_service=services.embedding_service,
            )
            
            # Register category generators
            _register_category_generators(generator_engine, services, config)
            
        except Exception as e:
            raise GenerationError(f"Failed to initialize generation engine: {e}") from e
        
        if stream:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Generation engine ready (30%)")
        
        # Generate dataset with error handling and progress tracking
        try:
            dataset_dict = generator_engine.generate_dataset()
            
            if not generator_engine.generated_prompts:
                raise GenerationError("No prompts were generated")
                
            if len(generator_engine.generated_prompts) != total:
                if not force:
                    raise GenerationError(f"Expected {total} prompts, got {len(generator_engine.generated_prompts)}")
                elif stream:
                    print(f"Warning: Generated {len(generator_engine.generated_prompts)} prompts instead of {total}")
                    
        except GenerationError:
            raise  # Re-raise generation errors
        except Exception as e:
            raise GenerationError(f"Dataset generation failed: {e}") from e
        
        if stream:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset generated (70%)")
        

        
        # Save dataset with error handling
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Simple JSON generation as fallback
            import json
            from datetime import datetime
            
            # Convert prompts to dictionaries
            prompts_data = []
            for prompt in generator_engine.generated_prompts:
                prompt_dict = {
                    "id": prompt.id,
                    "prompt": prompt.prompt,
                    "normalized_prompt": prompt.normalized_prompt,
                    "category": prompt.category.value,
                    "paraphrase_family": prompt.paraphrase_family,
                    "repeat_weight": prompt.repeat_weight,
                    "frequency_rank": prompt.frequency_rank,
                    "created_at": prompt.created_at,
                    "domain": prompt.domain.value,
                    "length": prompt.length.value,
                    "difficulty": prompt.difficulty.value,
                    "safety_label": prompt.safety_label.value,
                    "expected_policy_action": prompt.expected_policy_action.value,
                    "language": prompt.language
                }
                prompts_data.append(prompt_dict)
            
            # Create simple dataset structure
            dataset = {
                "metadata": {
                    "total_prompts": len(generator_engine.generated_prompts),
                    "created_date": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "description": "Generated by drengr framework",
                    "generator_version": "1.0.0",
                    "random_seed": config.random_seed,
                    "profile": config.profile
                },
                "prompts": prompts_data
            }
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            # Verify file was created and is not empty
            if not output_path.exists():
                raise GenerationError(f"Output file was not created: {output_path}")
            
            if output_path.stat().st_size == 0:
                raise GenerationError(f"Output file is empty: {output_path}")
                
        except GenerationError:
            raise  # Re-raise generation errors
        except Exception as e:
            raise GenerationError(f"Failed to save dataset to {output_path}: {e}") from e
        
        if stream:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset saved (95%)")
        
        # Show preview if requested
        if preview > 0:
            prompts = generator_engine.generated_prompts
            print(f"\n=== Preview ({min(preview, len(prompts))} prompts) ===")
            for i, prompt in enumerate(prompts[:preview]):
                prompt_text = prompt.prompt
                if len(prompt_text) > 100:
                    prompt_text = prompt_text[:97] + "..."
                print(f"{i+1:2d}. [{prompt.category.value}] {prompt_text}")
            
            if len(prompts) > preview:
                print(f"... and {len(prompts) - preview} more prompts")
            print()
        
        # Show summary
        generation_time = time.time() - start_time
        prompts_count = len(generator_engine.generated_prompts)
        
        print(f"✓ Generated {prompts_count} prompts using profile '{profile}' (seed: {seed})")
        print(f"✓ Generation time: {generation_time:.1f}s ({prompts_count/generation_time:.1f} prompts/sec)")
        print(f"✓ Dataset saved to: {output_path}")
        
        if stream:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Generation complete (100%)")
        
        return str(output_path)
        
    except GenerationError:
        raise  # Re-raise GenerationError without wrapping
    except Exception as e:
        if stream:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {e}")
        raise GenerationError(f"Dataset generation failed: {e}") from e


def _register_category_generators(engine: PromptGeneratorEngine, services, config: UnifiedDatasetConfig):
    """Register all category generators with the engine."""
    import logging
    logger = logging.getLogger(__name__)
    
    registered_generators = []
    
    # Register exact repeats generator
    try:
        from ..generators.exact_repeats import EliteExactRepeatsGenerator
        exact_repeats_gen = EliteExactRepeatsGenerator(services.embedding_service)
        engine.register_category_generator(Category.EXACT_REPEATS, exact_repeats_gen)
        registered_generators.append("exact_repeats")
    except Exception as e:
        logger.error(f"Failed to register exact repeats generator: {e}")
        raise BackendError(f"Missing required generator exact_repeats: {e}")
    
    # Register near duplicates generator
    try:
        from ..generators.near_duplicates import AdversarialFuzzingGenerator
        near_dup_gen = AdversarialFuzzingGenerator()
        engine.register_category_generator(Category.NEAR_DUPLICATES, near_dup_gen)
        registered_generators.append("near_duplicates")
    except Exception as e:
        logger.error(f"Failed to register near duplicates generator: {e}")
        raise BackendError(f"Missing required generator near_duplicates: {e}")
    
    # Register semantic paraphrase generator
    try:
        from ..generators.semantic_paraphraser import SOTASemanticParaphraseGenerator
        llm_service = getattr(services, 'llm_service', None)
        semantic_gen = SOTASemanticParaphraseGenerator(services.embedding_service, llm_service)
        engine.register_category_generator(Category.SEMANTIC_PARAPHRASE, semantic_gen)
        registered_generators.append("semantic_paraphrase")
    except Exception as e:
        logger.error(f"Failed to register semantic paraphrase generator: {e}")
        raise BackendError(f"Missing required generator semantic_paraphrase: {e}")
    
    # Register temporal anchor generator
    try:
        from ..generators.temporal_anchor import TemporalFreshnessGenerator
        temporal_gen = TemporalFreshnessGenerator()
        engine.register_category_generator(Category.TEMPORAL_ANCHOR, temporal_gen)
        registered_generators.append("temporal_anchor")
    except Exception as e:
        logger.error(f"Failed to register temporal anchor generator: {e}")
        raise BackendError(f"Missing required generator temporal_anchor: {e}")
    
    logger.info(f"Successfully registered generators: {registered_generators}")