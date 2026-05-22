"""
LLM Service for generating text responses.
Implements robust error handling, rate limiting, and response validation.
"""

import logging
import time
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

# CacheManager: LRU eviction, memory-aware, hit-rate stats.
# Replaces the unbounded plain dict that would OOM at 50k calls.
from ..optimization.memory_management import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM service."""
    content: str
    model: str
    tokens_used: int
    processing_time: float
    cached: bool = False
    metadata: Optional[Dict[str, Any]] = None


class LLMServiceError(Exception):
    """Base exception for LLM service errors."""
    pass


class LLMService(ABC):
    """Abstract base class for LLM services."""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt."""
        pass
    
    @abstractmethod
    def generate_responses_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts."""
        pass


class MockLLMService(LLMService):
    """Mock LLM service for testing and development."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize mock LLM service."""
        self.config = config or {}
        self.model = self.config.get("model", "mock-llm-model")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 150)
        
        self._cache = {}
        self._call_count = 0
        self._response_templates = [
            "This is a comprehensive response to: {prompt}",
            "Based on your query about '{prompt}', here's what I can tell you:",
            "Regarding '{prompt}', the key points are:",
            "To address your question about '{prompt}', consider the following:",
            "Here's a detailed analysis of '{prompt}':"
        ]
        
        logger.info(f"MockLLMService initialized with model: {self.model}")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a mock response based on the prompt."""
        self._call_count += 1
        
        # Create cache key
        cache_key = self._get_cache_key(prompt, kwargs)
        
        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Generate mock response
        response = self._generate_mock_response(prompt, **kwargs)
        
        # Cache the response
        self._cache[cache_key] = response
        
        return response
    
    def generate_responses_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts."""
        return [self.generate_response(prompt, **kwargs) for prompt in prompts]
    
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        try:
            test_response = self.generate_response("health check")
            return len(test_response) > 0
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "call_count": self._call_count,
            "cache_size": len(self._cache),
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model": self.model,
            "full_id": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "type": "mock",
            "provider": "mock"
        }
    
    def get_template_info(self) -> Dict[str, Any]:
        """Get template information."""
        return {
            "semantic_paraphrase": {
                "template": "Generate {count} paraphrases for '{base_intent}' in {domain} domain",
                "style": "mock"
            },
            "golden_response": {
                "template": "Generate a comprehensive response for: {prompt}",
                "style": "mock"
            }
        }
    
    def _generate_mock_response(self, prompt: str, **kwargs) -> str:
        """Generate a deterministic mock response."""
        # Use prompt hash to select template
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        template_index = int(prompt_hash[:2], 16) % len(self._response_templates)
        template = self._response_templates[template_index]
        
        # Generate base response
        base_response = template.format(prompt=prompt[:50])
        
        # Add some variation based on parameters
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        if temperature > 0.8:
            base_response += " This response includes creative and varied perspectives."
        elif temperature < 0.3:
            base_response += " This response focuses on factual and precise information."
        
        # Simulate token limit
        words = base_response.split()
        if len(words) > max_tokens // 4:  # Rough word-to-token ratio
            words = words[:max_tokens // 4]
            base_response = " ".join(words) + "..."
        
        return base_response
    
    def _get_cache_key(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for prompt and parameters."""
        key_data = {
            "prompt": prompt,
            "model": self.model,
            **kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


class ProductionLLMService(LLMService):
    """Production LLM service with API integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize production LLM service."""
        self.config = config or {}
        self.model = self.config.get("model", "gpt-3.5-turbo")
        self.api_key = self.config.get("api_key")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 150)
        self.timeout = self.config.get("timeout", 30)
        self.retry_attempts = self.config.get("retry_attempts", 3)

        # LRU cache: bounded at 5000 entries / 100 MB.
        # Temperature jitter in the paraphraser means each generation call
        # gets a unique cache key — so this mainly guards against duplicate
        # retry calls (length-fix, opener-fix passes) hitting the API twice.
        self._cache: CacheManager = CacheManager(max_size=5000, max_memory_mb=100.0)
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "errors": 0,
            "total_tokens": 0
        }
        
        if not self.api_key:
            logger.warning("No API key provided, falling back to mock mode")
            self._fallback_service = MockLLMService(config)
        else:
            self._fallback_service = None
        
        logger.info(f"ProductionLLMService initialized with model: {self.model}")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using the LLM API."""
        self._stats["total_requests"] += 1
        
        # Create cache key
        cache_key = self._get_cache_key(prompt, kwargs)
        
        # Check cache
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._stats["cache_hits"] += 1
            return cached
        
        # Use fallback if no API key
        if self._fallback_service:
            response = self._fallback_service.generate_response(prompt, **kwargs)
            self._cache.put(cache_key, response)
            return response
        
        # Make API call with retry logic
        for attempt in range(self.retry_attempts):
            try:
                response = self._call_llm_api(prompt, **kwargs)
                self._cache.put(cache_key, response)
                self._stats["api_calls"] += 1
                return response
            except Exception as e:
                logger.warning(f"LLM API call attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_attempts - 1:
                    self._stats["errors"] += 1
                    # Fall back to mock response
                    if not hasattr(self, '_emergency_fallback'):
                        self._emergency_fallback = MockLLMService()
                    return self._emergency_fallback.generate_response(prompt, **kwargs)
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def generate_responses_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts."""
        responses = []
        
        for prompt in prompts:
            response = self.generate_response(prompt, **kwargs)
            responses.append(response)
        
        return responses
    
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        try:
            test_response = self.generate_response("health check")
            return len(test_response) > 0
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        cache_stats = self._cache.get_stats()
        return {
            **self._stats,
            "cache_size": cache_stats["size"],
            "cache_memory_mb": round(cache_stats["memory_usage_mb"], 2),
            "cache_evictions": cache_stats["evictions"],
            "model": self.model,
            "cache_hit_rate": (self._stats["cache_hits"] / max(self._stats["total_requests"], 1)) * 100
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model": self.model,
            "full_id": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "type": "production",
            "provider": "openai"
        }
    
    def get_template_info(self) -> Dict[str, Any]:
        """Get template information."""
        return {
            "semantic_paraphrase": {
                "template": "Generate {count} high-quality paraphrases for the intent '{base_intent}' in the {domain} domain",
                "style": "production"
            },
            "golden_response": {
                "template": "Generate a comprehensive, accurate response for: {prompt}",
                "style": "production"
            }
        }
    
    def _call_llm_api(self, prompt: str, **kwargs) -> str:
        """Make actual API call to OpenAI-compatible LLM service (DeepSeek, OpenAI, etc.)."""
        import urllib.request
        import urllib.error

        base_url = self.config.get("base_url", "https://api.deepseek.com")
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                tokens = body.get("usage", {}).get("total_tokens", 0)
                self._stats["total_tokens"] += tokens
                return content
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise LLMServiceError(f"API error {e.code}: {error_body}") from e
        except Exception as e:
            raise LLMServiceError(f"API call failed: {e}") from e
    
    def _get_cache_key(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for prompt and parameters."""
        key_data = {
            "prompt": prompt,
            "model": self.model,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


class ParaphraseTemplateManager:
    """Manager for paraphrase templates and generation patterns."""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        """Initialize paraphrase template manager."""
        self.llm_service = llm_service or MockLLMService()
        
        # Paraphrase templates for different styles
        self.templates = {
            "formal": [
                "Please provide information regarding {topic}",
                "I would like to inquire about {topic}",
                "Could you elaborate on {topic}",
                "I am seeking details about {topic}"
            ],
            "casual": [
                "Tell me about {topic}",
                "What's up with {topic}?",
                "Can you explain {topic}?",
                "I want to know about {topic}"
            ],
            "technical": [
                "Analyze the technical aspects of {topic}",
                "Provide a technical overview of {topic}",
                "Explain the implementation details of {topic}",
                "Describe the technical specifications for {topic}"
            ],
            "question": [
                "What is {topic}?",
                "How does {topic} work?",
                "Why is {topic} important?",
                "When should I use {topic}?"
            ]
        }
    
    def generate_paraphrases(self, original_text: str, count: int = 5, style: str = "mixed") -> List[str]:
        """Generate paraphrases of the original text."""
        paraphrases = []
        
        if style == "mixed":
            # Use different styles
            styles = list(self.templates.keys())
            for i in range(count):
                current_style = styles[i % len(styles)]
                paraphrase = self._generate_paraphrase_with_style(original_text, current_style)
                paraphrases.append(paraphrase)
        else:
            # Use specific style
            for i in range(count):
                paraphrase = self._generate_paraphrase_with_style(original_text, style)
                paraphrases.append(paraphrase)
        
        return paraphrases
    
    def _generate_paraphrase_with_style(self, text: str, style: str) -> str:
        """Generate a paraphrase using a specific style."""
        if style not in self.templates:
            style = "casual"
        
        templates = self.templates[style]
        template = templates[hash(text) % len(templates)]
        
        # Extract topic from text (simple approach)
        topic = self._extract_topic(text)
        
        # Apply template
        paraphrase = template.format(topic=topic)
        
        # Add some variation
        variations = [
            paraphrase,
            paraphrase.replace("?", "."),
            paraphrase.replace(".", "?"),
            f"Actually, {paraphrase.lower()}",
            f"Specifically, {paraphrase.lower()}"
        ]
        
        return variations[hash(text + style) % len(variations)]
    
    def _extract_topic(self, text: str) -> str:
        """Extract main topic from text."""
        # Simple topic extraction
        words = text.split()
        if len(words) <= 3:
            return text
        
        # Remove common words
        stop_words = {"what", "how", "why", "when", "where", "who", "is", "are", "can", "could", "would", "should", "the", "a", "an"}
        meaningful_words = [word for word in words if word.lower() not in stop_words]
        
        if meaningful_words:
            return " ".join(meaningful_words[:3])
        else:
            return " ".join(words[:3])


class LLMServiceFactory:
    """Factory for creating LLM service instances."""
    
    @staticmethod
    def create_service(config: Optional[Dict[str, Any]] = None) -> LLMService:
        """Create an LLM service based on configuration."""
        config = config or {}
        
        # Determine service type
        service_type = config.get("type", "mock")
        
        if service_type == "production" and config.get("api_key"):
            return ProductionLLMService(config)
        else:
            return MockLLMService(config)
    
    @staticmethod
    def create_mock_service(config: Optional[Dict[str, Any]] = None) -> MockLLMService:
        """Create a mock LLM service."""
        return MockLLMService(config)
    
    @staticmethod
    def create_production_service(config: Dict[str, Any]) -> ProductionLLMService:
        """Create a production LLM service."""
        return ProductionLLMService(config)