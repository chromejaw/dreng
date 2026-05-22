"""
Golden Response Generator for creating reference responses.
Implements domain-specific response generation with quality validation.
"""

import logging
import hashlib
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class ResponseQuality(Enum):
    """Quality levels for golden responses."""
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    EXPERT = "expert"


@dataclass
class GoldenResponse:
    """A golden response with metadata."""
    content: str
    tokens: int
    quality: ResponseQuality
    domain: str
    version: str
    created_at: str
    metadata: Optional[Dict[str, Any]] = None


class GoldenResponseGenerator(ABC):
    """Abstract base class for golden response generators."""
    
    @abstractmethod
    def generate_response(self, prompt: str, domain: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate a golden response for the given prompt."""
        pass


class MockGoldenResponseGenerator(GoldenResponseGenerator):
    """Mock golden response generator for testing and development."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize mock golden response generator."""
        self.config = config or {}
        self.quality = ResponseQuality(self.config.get("quality", "standard"))
        self.version = self.config.get("version", "1.0")
        
        self._cache = {}
        self._call_count = 0
        
        # Domain-specific response templates
        self._domain_templates = {
            "general": [
                "This is a comprehensive answer to your question about {topic}.",
                "Based on the query regarding {topic}, here's what you need to know:",
                "To address your question about {topic}, consider these key points:",
            ],
            "technical": [
                "From a technical perspective, {topic} involves several key components:",
                "The technical implementation of {topic} requires understanding:",
                "When working with {topic}, the following technical considerations apply:",
            ],
            "creative": [
                "Exploring the creative aspects of {topic}, we can consider:",
                "From a creative standpoint, {topic} offers opportunities for:",
                "The artistic and creative dimensions of {topic} include:",
            ],
            "analytical": [
                "Analyzing {topic} from multiple perspectives reveals:",
                "A systematic analysis of {topic} shows:",
                "Breaking down {topic} analytically, we find:",
            ]
        }
        
        logger.info(f"MockGoldenResponseGenerator initialized with quality: {self.quality}")
    
    def generate_response(self, prompt: str, domain: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate a mock golden response."""
        self._call_count += 1
        
        # Create cache key
        cache_key = self._get_cache_key(prompt, domain, kwargs)
        
        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Generate response
        response_data = self._generate_mock_response(prompt, domain, **kwargs)
        
        # Cache the response
        self._cache[cache_key] = response_data
        
        return response_data
    
    def generate_responses_batch(self, prompts: List[str], domain: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Generate golden responses for multiple prompts."""
        return [self.generate_response(prompt, domain, **kwargs) for prompt in prompts]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "call_count": self._call_count,
            "cache_size": len(self._cache),
            "quality": self.quality.value,
            "version": self.version
        }
    
    def _generate_mock_response(self, prompt: str, domain: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate a mock golden response with realistic structure."""
        # Determine domain
        if not domain:
            domain = self._infer_domain(prompt)
        
        # Get appropriate template
        templates = self._domain_templates.get(domain, self._domain_templates["general"])
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        template_index = int(prompt_hash[:2], 16) % len(templates)
        template = templates[template_index]
        
        # Extract topic from prompt
        topic = self._extract_topic(prompt)
        
        # Generate base response
        base_response = template.format(topic=topic)
        
        # Enhance based on quality level
        enhanced_response = self._enhance_response(base_response, prompt, domain)
        
        # Calculate token count (rough estimation)
        token_count = len(enhanced_response.split()) * 1.3  # Rough word-to-token ratio
        
        # Create response data
        from datetime import datetime, timezone
        response_data = {
            "golden_response": enhanced_response,
            "tokens": int(token_count),
            "quality": self.quality.value,
            "domain": domain,
            "version": self.version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "prompt_length": len(prompt),
                "response_length": len(enhanced_response),
                "template_used": template_index,
                "topic": topic
            }
        }
        
        return response_data
    
    def _infer_domain(self, prompt: str) -> str:
        """Infer domain from prompt content."""
        prompt_lower = prompt.lower()
        
        # Technical keywords
        technical_keywords = ["code", "programming", "algorithm", "database", "api", "system", "technical"]
        if any(keyword in prompt_lower for keyword in technical_keywords):
            return "technical"
        
        # Creative keywords
        creative_keywords = ["creative", "story", "design", "art", "imagine", "creative"]
        if any(keyword in prompt_lower for keyword in creative_keywords):
            return "creative"
        
        # Analytical keywords
        analytical_keywords = ["analyze", "compare", "evaluate", "assess", "study", "research"]
        if any(keyword in prompt_lower for keyword in analytical_keywords):
            return "analytical"
        
        return "general"
    
    def _extract_topic(self, prompt: str) -> str:
        """Extract main topic from prompt."""
        # Simple topic extraction - take first few meaningful words
        words = prompt.split()
        if len(words) <= 3:
            return prompt
        
        # Remove common question words
        stop_words = {"what", "how", "why", "when", "where", "who", "is", "are", "can", "could", "would", "should"}
        meaningful_words = [word for word in words[:5] if word.lower() not in stop_words]
        
        if meaningful_words:
            return " ".join(meaningful_words[:3])
        else:
            return " ".join(words[:3])
    
    def _enhance_response(self, base_response: str, prompt: str, domain: str) -> str:
        """Enhance response based on quality level and domain."""
        enhanced = base_response
        
        if self.quality in [ResponseQuality.STANDARD, ResponseQuality.PREMIUM, ResponseQuality.EXPERT]:
            enhanced += f" Additionally, it's important to consider the broader context and implications."
        
        if self.quality in [ResponseQuality.PREMIUM, ResponseQuality.EXPERT]:
            enhanced += f" From an expert perspective, there are several advanced considerations to keep in mind."
        
        if self.quality == ResponseQuality.EXPERT:
            enhanced += f" The cutting-edge research in this area suggests innovative approaches that could be particularly relevant."
        
        # Domain-specific enhancements
        if domain == "technical":
            enhanced += " Technical implementation details and best practices should be carefully considered."
        elif domain == "creative":
            enhanced += " Creative exploration and innovative thinking can lead to unique solutions."
        elif domain == "analytical":
            enhanced += " Systematic analysis and data-driven insights provide valuable perspectives."
        
        return enhanced
    
    def _get_cache_key(self, prompt: str, domain: Optional[str], kwargs: Dict[str, Any]) -> str:
        """Generate cache key for prompt and parameters."""
        key_data = {
            "prompt": prompt,
            "domain": domain,
            "quality": self.quality.value,
            "version": self.version,
            **kwargs
        }
        import json
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


class ProductionGoldenResponseGenerator(GoldenResponseGenerator):
    """Production golden response generator with LLM integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize production golden response generator."""
        self.config = config or {}
        self.quality = ResponseQuality(self.config.get("quality", "standard"))
        self.version = self.config.get("version", "1.0")
        
        # Initialize LLM service for response generation
        from .llm import LLMServiceFactory
        llm_config = self.config.get("llm_config", {})
        self.llm_service = LLMServiceFactory.create_service(llm_config)
        
        self._cache = {}
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "errors": 0
        }
        
        logger.info(f"ProductionGoldenResponseGenerator initialized with quality: {self.quality}")
    
    def generate_response(self, prompt: str, domain: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate a golden response using LLM."""
        self._stats["total_requests"] += 1
        
        # Create cache key
        cache_key = self._get_cache_key(prompt, domain, kwargs)
        
        # Check cache
        if cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]
        
        try:
            # Generate response using LLM
            enhanced_prompt = self._create_enhanced_prompt(prompt, domain)
            llm_response = self.llm_service.generate_response(enhanced_prompt, **kwargs)
            
            # Process and structure the response
            response_data = self._process_llm_response(llm_response, prompt, domain)
            
            # Cache the response
            self._cache[cache_key] = response_data
            self._stats["llm_calls"] += 1
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating golden response: {e}")
            self._stats["errors"] += 1
            
            # Fall back to mock generator
            if not hasattr(self, '_fallback_generator'):
                self._fallback_generator = MockGoldenResponseGenerator(self.config)
            
            return self._fallback_generator.generate_response(prompt, domain, **kwargs)
    
    def _create_enhanced_prompt(self, prompt: str, domain: Optional[str]) -> str:
        """Create an enhanced prompt for LLM to generate golden response."""
        system_prompt = f"""Generate a high-quality, comprehensive response that would serve as a golden reference answer. 
        Quality level: {self.quality.value}
        Domain: {domain or 'general'}
        
        The response should be:
        - Accurate and informative
        - Well-structured and clear
        - Appropriate for the specified domain
        - Comprehensive but concise
        
        Original prompt: {prompt}
        
        Provide a response that would be considered the gold standard for this type of question."""
        
        return system_prompt
    
    def _process_llm_response(self, llm_response: str, original_prompt: str, domain: Optional[str]) -> Dict[str, Any]:
        """Process LLM response into structured golden response format."""
        from datetime import datetime, timezone
        
        # Calculate token count
        token_count = len(llm_response.split()) * 1.3  # Rough estimation
        
        response_data = {
            "golden_response": llm_response,
            "tokens": int(token_count),
            "quality": self.quality.value,
            "domain": domain or "general",
            "version": self.version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "prompt_length": len(original_prompt),
                "response_length": len(llm_response),
                "generated_by": "llm"
            }
        }
        
        return response_data
    
    def _get_cache_key(self, prompt: str, domain: Optional[str], kwargs: Dict[str, Any]) -> str:
        """Generate cache key for prompt and parameters."""
        key_data = {
            "prompt": prompt,
            "domain": domain,
            "quality": self.quality.value,
            "version": self.version,
            **kwargs
        }
        import json
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


class GoldenResponseFactory:
    """Factory for creating golden response generator instances."""
    
    @staticmethod
    def create_generator(config: Optional[Dict[str, Any]] = None) -> GoldenResponseGenerator:
        """Create a golden response generator based on configuration."""
        config = config or {}
        
        # Determine generator type
        generator_type = config.get("type", "mock")
        
        if generator_type == "production":
            return ProductionGoldenResponseGenerator(config)
        else:
            return MockGoldenResponseGenerator(config)
    
    @staticmethod
    def create_mock_generator(config: Optional[Dict[str, Any]] = None) -> MockGoldenResponseGenerator:
        """Create a mock golden response generator."""
        return MockGoldenResponseGenerator(config)
    
    @staticmethod
    def create_production_generator(config: Dict[str, Any]) -> ProductionGoldenResponseGenerator:
        """Create a production golden response generator."""
        return ProductionGoldenResponseGenerator(config)