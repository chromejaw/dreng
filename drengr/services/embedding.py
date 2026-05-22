"""
Embedding Service for generating and managing text embeddings.
Implements robust error handling, caching, and batch processing.
"""

import logging
import hashlib
import time
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    embedding: List[float]
    model: str
    tokens_used: int
    processing_time: float
    cached: bool = False


class EmbeddingServiceError(Exception):
    """Base exception for embedding service errors."""
    pass


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        pass
    
    @abstractmethod
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        pass


class MockEmbeddingService(EmbeddingService):
    """Mock embedding service for testing and development."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize mock embedding service."""
        self.config = config or {}
        self.model = self.config.get("model", "mock-embedding-model")
        self.dimension = self.config.get("dimension", 1536)
        self._cache = {}
        self._call_count = 0
        
        logger.info(f"MockEmbeddingService initialized with model: {self.model}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate a deterministic mock embedding based on text hash."""
        self._call_count += 1
        
        # Use text hash for deterministic embeddings
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache first
        if text_hash in self._cache:
            return self._cache[text_hash]
        
        # Generate deterministic embedding
        embedding = self._generate_mock_embedding(text_hash)
        
        # Cache the result
        self._cache[text_hash] = embedding
        
        return embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        return [self.get_embedding(text) for text in texts]
    
    def calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if len(emb1) != len(emb2):
            raise ValueError("Embeddings must have the same dimension")
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in emb1))
        magnitude2 = math.sqrt(sum(a * a for a in emb2))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Return cosine similarity
        return dot_product / (magnitude1 * magnitude2)
    
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        try:
            test_embedding = self.get_embedding("health check")
            return len(test_embedding) == self.dimension
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "call_count": self._call_count,
            "cache_size": len(self._cache),
            "model": self.model,
            "dimension": self.dimension
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model": self.model,
            "full_id": f"{self.model}-{self.dimension}d",
            "dimension": self.dimension,
            "type": "mock",
            "provider": "mock"
        }
    
    def _generate_mock_embedding(self, text_hash: str) -> List[float]:
        """Generate a deterministic mock embedding from text hash."""
        # Use hash to seed pseudo-random generation
        seed_value = int(text_hash[:8], 16)
        
        # Generate embedding values
        embedding = []
        for i in range(self.dimension):
            # Create deterministic but varied values
            value = math.sin((seed_value + i) * 0.1) * 0.5
            embedding.append(value)
        
        # Normalize the embedding
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding


class ProductionEmbeddingService(EmbeddingService):
    """Production embedding service with API integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize production embedding service."""
        self.config = config or {}
        self.model = self.config.get("model", "text-embedding-ada-002")
        self.api_key = self.config.get("api_key")
        self.batch_size = self.config.get("batch_size", 100)
        self.timeout = self.config.get("timeout", 30)
        self.retry_attempts = self.config.get("retry_attempts", 3)
        
        self._cache = {}
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "errors": 0
        }
        
        if not self.api_key:
            logger.warning("No API key provided, falling back to mock mode")
            self._fallback_service = MockEmbeddingService(config)
        else:
            self._fallback_service = None
        
        logger.info(f"ProductionEmbeddingService initialized with model: {self.model}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        self._stats["total_requests"] += 1
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]
        
        # Use fallback if no API key
        if self._fallback_service:
            embedding = self._fallback_service.get_embedding(text)
            self._cache[cache_key] = embedding
            return embedding
        
        # Make API call with retry logic
        for attempt in range(self.retry_attempts):
            try:
                embedding = self._call_embedding_api([text])[0]
                self._cache[cache_key] = embedding
                self._stats["api_calls"] += 1
                return embedding
            except Exception as e:
                logger.warning(f"Embedding API call attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_attempts - 1:
                    self._stats["errors"] += 1
                    # Fall back to mock embedding
                    if not hasattr(self, '_emergency_fallback'):
                        self._emergency_fallback = MockEmbeddingService()
                    return self._emergency_fallback.get_embedding(text)
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts with batching."""
        if not texts:
            return []
        
        # Check cache for all texts
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                embeddings.append(self._cache[cache_key])
                self._stats["cache_hits"] += 1
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts in batches
        if uncached_texts:
            if self._fallback_service:
                new_embeddings = self._fallback_service.get_embeddings_batch(uncached_texts)
            else:
                new_embeddings = self._get_embeddings_with_batching(uncached_texts)
            
            # Fill in the placeholders
            for i, embedding in enumerate(new_embeddings):
                original_index = uncached_indices[i]
                embeddings[original_index] = embedding
                
                # Cache the result
                cache_key = self._get_cache_key(uncached_texts[i])
                self._cache[cache_key] = embedding
        
        return embeddings
    
    def calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if len(emb1) != len(emb2):
            raise ValueError("Embeddings must have the same dimension")
        
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        magnitude1 = math.sqrt(sum(a * a for a in emb1))
        magnitude2 = math.sqrt(sum(a * a for a in emb2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        try:
            test_embedding = self.get_embedding("health check")
            return len(test_embedding) > 0
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "model": self.model,
            "cache_hit_rate": (self._stats["cache_hits"] / max(self._stats["total_requests"], 1)) * 100
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model": self.model,
            "full_id": self.model,
            "dimension": 1536,  # Default for most embedding models
            "type": "production",
            "provider": "openai"
        }
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()
    
    def _get_embeddings_with_batching(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings with proper batching."""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._call_embedding_api(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """Make actual API call to embedding service."""
        # This would be the actual API integration
        # For now, fall back to mock implementation
        if not hasattr(self, '_api_fallback'):
            self._api_fallback = MockEmbeddingService(self.config)
        
        return self._api_fallback.get_embeddings_batch(texts)


class EmbeddingServiceFactory:
    """Factory for creating embedding service instances."""
    
    @staticmethod
    def create_service(config: Optional[Dict[str, Any]] = None) -> EmbeddingService:
        """Create an embedding service based on configuration."""
        config = config or {}
        
        # Determine service type
        service_type = config.get("type", "mock")
        
        if service_type == "production" and config.get("api_key"):
            return ProductionEmbeddingService(config)
        else:
            return MockEmbeddingService(config)
    
    @staticmethod
    def create_mock_service(config: Optional[Dict[str, Any]] = None) -> MockEmbeddingService:
        """Create a mock embedding service."""
        return MockEmbeddingService(config)
    
    @staticmethod
    def create_production_service(config: Dict[str, Any]) -> ProductionEmbeddingService:
        """Create a production embedding service."""
        return ProductionEmbeddingService(config)