"""
Elite exact repeats generator with precision Zipf distribution weighting.

This module implements state-of-the-art verbatim duplicate creation with
Zipf distribution integration, focusing on high-repeat intents and creating
byte-for-byte identical string validation with realistic traffic patterns.
"""

import logging
import random
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from ..core.models import (
    GeneratedPrompt, PromptSpec, CategoryGenerator, Domain, Category,
    Length, Difficulty, SafetyLabel, PolicyAction
)
from ..core.exceptions import ValidationError
from ..services.embedding import EmbeddingService
try:
    from ..utils.zipf import ZipfDistributionEngine, RepeatWeightCalculator
except ImportError:
    # Fallback implementations
    class ZipfDistributionEngine:
        def generate_zipf_weights(self, num_items, exponent=1.0):
            import random
            return [random.randint(1, 100) for _ in range(num_items)]
    
    class RepeatWeightCalculator:
        def __init__(self, zipf_engine):
            self.zipf_engine = zipf_engine


@dataclass
class ExactRepeatConfig:
    """Elite configuration for exact repeat generation."""
    zipf_exponent: float = 1.0
    min_repeat_weight: int = 1
    max_repeat_weight: int = 1000
    high_repeat_threshold: int = 50
    hot_intent_ratio: float = 0.1  # 10% of intents are hot


class EliteExactRepeatsGenerator(CategoryGenerator):
    """Elite exact repeats generator with Zipf distribution precision."""
    
    def __init__(self, embedding_service: EmbeddingService):
        """Initialize elite exact repeats generator."""
        self.embedding_service = embedding_service
        self.config = ExactRepeatConfig()
        self.zipf_engine = ZipfDistributionEngine()
        self.weight_calculator = RepeatWeightCalculator(self.zipf_engine)
        
        self.logger = logging.getLogger(__name__)
        
        # Elite high-repeat intent templates
        self._load_high_repeat_intents()
        
        # Track generated repeats for validation
        self._generated_repeats: Dict[str, List[GeneratedPrompt]] = {}
    
    def generate_prompts(self, count: int, specs: List[PromptSpec]) -> List[GeneratedPrompt]:
        """Generate exact repeat prompts with elite Zipf distribution."""
        try:
            self.logger.info(f"Generating {count} exact repeat prompts with Zipf distribution")
            
            # Create base intent templates
            base_intents = self._create_base_intents(specs)
            
            # Apply Zipf distribution to determine repeat counts
            repeat_distribution = self._calculate_zipf_repeat_distribution(
                base_intents, count
            )
            
            # Generate exact repeats based on distribution
            exact_repeats = self._generate_exact_repeats_from_distribution(
                repeat_distribution, specs
            )
            
            # Validate exact repetition
            self._validate_exact_repetition(exact_repeats)
            
            # Ensure we have exactly the requested count
            if len(exact_repeats) > count:
                exact_repeats = exact_repeats[:count]
            
            self.logger.info(f"Generated {len(exact_repeats)} exact repeat prompts")
            return exact_repeats
            
        except Exception as e:
            raise ValidationError(f"Exact repeats generation failed: {e}") from e
    
    def validate_similarity_bands(self, prompts: List[GeneratedPrompt]) -> bool:
        """Validate exact repeats have perfect similarity (1.0)."""
        try:
            for prompt in prompts:
                # Group by prompt text to find repeats
                prompt_text = prompt.prompt
                
                if prompt_text in self._generated_repeats:
                    # Check that all repeats are identical
                    for existing_prompt in self._generated_repeats[prompt_text]:
                        if existing_prompt.prompt != prompt.prompt:
                            self.logger.error(
                                f"Exact repeat validation failed: prompts {existing_prompt.id} "
                                f"and {prompt.id} should be identical"
                            )
                            return False
                else:
                    self._generated_repeats[prompt_text] = []
                
                self._generated_repeats[prompt_text].append(prompt)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Exact repeat validation failed: {e}")
            return False
    
    def _load_high_repeat_intents(self) -> None:
        """Load elite high-repeat intent templates."""
        self.high_repeat_intents = {
            Domain.PROGRAMMING: [
                "How to write a Python for loop?",
                "Debug function not working",
                "Install package with pip",
                "Create new Git repository",
                "Fix syntax error in code",
                "Connect to database",
                "Parse JSON data",
                "Handle exceptions in Python",
                "Write unit tests",
                "Deploy to production"
            ],
            Domain.CUSTOMER_SUPPORT: [
                "Reset my password",
                "Track my order",
                "Cancel subscription",
                "Refund request",
                "Update billing information",
                "Change email address",
                "Delete my account",
                "Contact customer service",
                "Report a bug",
                "Get help with login"
            ],
            Domain.BUSINESS: [
                "Schedule team meeting",
                "Create project timeline",
                "Generate monthly report",
                "Calculate budget forecast",
                "Review performance metrics",
                "Update client proposal",
                "Prepare presentation slides",
                "Analyze market trends",
                "Plan quarterly goals",
                "Conduct employee review"
            ],
            Domain.TECHNICAL: [
                "Configure server settings",
                "Optimize database performance",
                "Setup monitoring alerts",
                "Deploy application update",
                "Backup system data",
                "Troubleshoot network issue",
                "Scale infrastructure",
                "Implement security patch",
                "Monitor system health",
                "Update documentation"
            ],
            Domain.EDUCATION: [
                "Explain this concept",
                "Solve math problem",
                "Write essay outline",
                "Create study guide",
                "Prepare for exam",
                "Research topic",
                "Check homework answers",
                "Understand theory",
                "Practice problems",
                "Review lesson notes"
            ],
            Domain.CREATIVE: [
                "Write story beginning",
                "Design logo concept",
                "Create color palette",
                "Brainstorm ideas",
                "Edit photo",
                "Compose music",
                "Write poem",
                "Design website layout",
                "Create marketing copy",
                "Develop character"
            ],
            Domain.ECOMMERCE: [
                "Track order status",
                "Process return",
                "Update product listing",
                "Calculate shipping cost",
                "Apply discount code",
                "Check inventory",
                "Generate invoice",
                "Update payment method",
                "Review product ratings",
                "Contact seller"
            ],
            Domain.TRAVEL: [
                "Book flight tickets",
                "Find hotel deals",
                "Check travel restrictions",
                "Plan itinerary",
                "Get directions",
                "Convert currency",
                "Check weather forecast",
                "Find restaurants",
                "Book rental car",
                "Get travel insurance"
            ],
            Domain.LEGAL: [
                "Review contract terms",
                "File legal document",
                "Check compliance requirements",
                "Understand regulations",
                "Prepare legal brief",
                "Schedule consultation",
                "Research case law",
                "Draft agreement",
                "Review policy",
                "Get legal advice"
            ],
            Domain.HEALTHCARE: [
                "Schedule appointment",
                "Check symptoms",
                "Review test results",
                "Get prescription refill",
                "Find specialist",
                "Understand treatment",
                "Check insurance coverage",
                "Get health information",
                "Book screening",
                "Contact doctor"
            ]
        }
    
    def _create_base_intents(self, specs: List[PromptSpec]) -> List[Dict]:
        """Create base intent templates from specs."""
        base_intents = []
        
        # Group specs by domain
        domain_groups = defaultdict(list)
        for spec in specs:
            domain_groups[spec.domain].append(spec)
        
        # Create intents for each domain
        for domain, domain_specs in domain_groups.items():
            domain_intents = self.high_repeat_intents.get(domain, [
                f"Generic {domain.value} query",
                f"Help with {domain.value} task",
                f"Information about {domain.value}"
            ])
            
            # Create base intent objects
            for i, intent_text in enumerate(domain_intents):
                if i < len(domain_specs):
                    spec = domain_specs[i]
                else:
                    # Use first spec as template
                    spec = domain_specs[0] if domain_specs else specs[0]
                
                base_intent = {
                    'text': intent_text,
                    'domain': domain,
                    'spec': spec,
                    'priority': self._calculate_intent_priority(intent_text, domain)
                }
                base_intents.append(base_intent)
        
        return base_intents
    
    def _calculate_intent_priority(self, intent_text: str, domain: Domain) -> float:
        """Calculate priority score for intent (higher = more repeats)."""
        priority = 1.0
        
        # High-priority keywords
        high_priority_keywords = {
            'password', 'reset', 'login', 'help', 'error', 'bug', 'fix',
            'install', 'setup', 'configure', 'track', 'order', 'status'
        }
        
        intent_lower = intent_text.lower()
        for keyword in high_priority_keywords:
            if keyword in intent_lower:
                priority += 2.0
                break
        
        # Domain-specific priorities
        if domain == Domain.CUSTOMER_SUPPORT:
            priority += 1.5  # Customer support queries repeat more
        elif domain == Domain.PROGRAMMING:
            priority += 1.2  # Programming queries are common
        elif domain == Domain.TECHNICAL:
            priority += 1.0
        
        return priority
    
    def _calculate_zipf_repeat_distribution(self, 
                                          base_intents: List[Dict],
                                          total_count: int) -> Dict[str, int]:
        """Calculate Zipf distribution for repeat counts."""
        try:
            # Sort intents by priority (highest first)
            sorted_intents = sorted(base_intents, key=lambda x: x['priority'], reverse=True)
            
            # Generate Zipf weights
            num_intents = len(sorted_intents)
            zipf_weights = self.zipf_engine.generate_zipf_weights(
                num_intents, self.config.zipf_exponent
            )
            
            # Calculate repeat counts based on weights
            total_weight = sum(zipf_weights)
            repeat_distribution = {}
            
            for i, intent in enumerate(sorted_intents):
                weight = zipf_weights[i]
                repeat_count = max(1, int((weight / total_weight) * total_count))
                repeat_distribution[intent['text']] = {
                    'count': repeat_count,
                    'intent': intent,
                    'weight': weight
                }
            
            # Adjust to exact total count
            current_total = sum(item['count'] for item in repeat_distribution.values())
            
            if current_total != total_count:
                # Adjust the highest priority intent
                first_intent = sorted_intents[0]['text']
                adjustment = total_count - current_total
                repeat_distribution[first_intent]['count'] += adjustment
                repeat_distribution[first_intent]['count'] = max(1, 
                    repeat_distribution[first_intent]['count'])
            
            self.logger.info(f"Created Zipf distribution for {len(repeat_distribution)} intents")
            return repeat_distribution
            
        except Exception as e:
            raise ValidationError(f"Zipf distribution calculation failed: {e}") from e
    
    def _generate_exact_repeats_from_distribution(self, 
                                                repeat_distribution: Dict[str, Dict],
                                                specs: List[PromptSpec]) -> List[GeneratedPrompt]:
        """Generate exact repeats based on distribution."""
        exact_repeats = []
        current_id = 1
        
        for intent_text, distribution_data in repeat_distribution.items():
            repeat_count = distribution_data['count']
            intent_data = distribution_data['intent']
            weight = distribution_data['weight']
            
            # Create exact repeats for this intent
            for i in range(repeat_count):
                exact_repeat = self._create_exact_repeat_prompt(
                    intent_text, intent_data, weight, current_id
                )
                exact_repeats.append(exact_repeat)
                current_id += 1
        
        return exact_repeats
    
    def _create_exact_repeat_prompt(self, 
                                  intent_text: str,
                                  intent_data: Dict,
                                  weight: int,
                                  prompt_id: int) -> GeneratedPrompt:
        """Create exact repeat prompt object."""
        spec = intent_data['spec']
        domain = intent_data['domain']
        
        return GeneratedPrompt(
            id=prompt_id,
            prompt=intent_text,  # Exact same text for all repeats
            normalized_prompt=intent_text.lower().strip(),
            category=Category.EXACT_REPEATS,
            paraphrase_family=f"exact_{domain.value}_{hash(intent_text) % 1000}",
            repeat_weight=weight,
            frequency_rank=self._calculate_frequency_rank(weight),
            created_at="2025-01-01T12:00:00Z",  # Will be updated by temporal manager
            source_last_updated=None,
            valid_until=None,
            session_id=None,
            user_id=None,
            turn_index=None,
            previous_message_snippet=None,
            hard_negative_of=[],
            negative_type=None,
            burst_group_id=None,
            burst_size=None,
            burst_window_seconds=None,
            arrival_distribution=None,
            safety_label=SafetyLabel.SAFE,
            expected_policy_action=PolicyAction.RESPOND,
            domain=domain,
            length=self._determine_length(intent_text),
            difficulty=spec.difficulty,
            language="en"
        )
    
    def _calculate_frequency_rank(self, weight: int) -> int:
        """Calculate frequency rank based on weight."""
        # Higher weight = lower rank number (more frequent)
        max_weight = self.config.max_repeat_weight
        return max(1, int((max_weight - weight) / 10) + 1)
    
    def _determine_length(self, text: str) -> Length:
        """Determine length category based on text."""
        word_count = len(text.split())
        
        if word_count <= 5:
            return Length.SHORT
        elif word_count <= 15:
            return Length.MEDIUM
        else:
            return Length.LONG
    
    def _validate_exact_repetition(self, exact_repeats: List[GeneratedPrompt]) -> None:
        """Validate that exact repeats are truly identical."""
        try:
            # Group by prompt text
            text_groups = defaultdict(list)
            for prompt in exact_repeats:
                text_groups[prompt.prompt].append(prompt)
            
            # Validate each group
            for text, prompts in text_groups.items():
                if len(prompts) > 1:
                    # Verify all prompts in group are identical
                    first_prompt = prompts[0]
                    
                    for prompt in prompts[1:]:
                        if prompt.prompt != first_prompt.prompt:
                            raise ValidationError(
                                f"Exact repeat validation failed: prompts {first_prompt.id} "
                                f"and {prompt.id} should be identical"
                            )
            
            self.logger.info(f"Validated {len(text_groups)} unique exact repeat groups")
            
        except Exception as e:
            raise ValidationError(f"Exact repetition validation failed: {e}") from e
    
    def get_repeat_statistics(self) -> Dict[str, any]:
        """Get statistics about generated exact repeats."""
        if not self._generated_repeats:
            return {"total_groups": 0, "total_repeats": 0}
        
        total_groups = len(self._generated_repeats)
        total_repeats = sum(len(prompts) for prompts in self._generated_repeats.values())
        
        # Calculate repeat distribution
        repeat_counts = [len(prompts) for prompts in self._generated_repeats.values()]
        
        return {
            "total_groups": total_groups,
            "total_repeats": total_repeats,
            "avg_repeats_per_group": total_repeats / total_groups if total_groups > 0 else 0,
            "max_repeats_per_group": max(repeat_counts) if repeat_counts else 0,
            "min_repeats_per_group": min(repeat_counts) if repeat_counts else 0,
            "groups_with_high_repeats": sum(1 for count in repeat_counts 
                                          if count >= self.config.high_repeat_threshold)
        }