"""
Service Factory for managing service dependencies and lifecycle.
Implements the Factory and Service Locator patterns for robust service management.
"""

import logging
from typing import Dict, Any, Optional, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ServiceHealthCheck(Protocol):
    """Protocol for service health checking."""
    def is_healthy(self) -> bool:
        """Check if service is healthy and operational."""
        ...


@dataclass
class ServiceMetrics:
    """Metrics for service performance monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0


class ServiceContainer:
    """Container for managing service instances and their lifecycle."""
    
    def __init__(self, embedding_service=None, llm_service=None):
        """Initialize service container with optional services."""
        from .embedding import EmbeddingService
        from .llm import LLMService
        
        self.embedding_service = embedding_service or EmbeddingService()
        self.llm_service = llm_service or LLMService()
        self._metrics = ServiceMetrics()
        self._initialized = True
        
        logger.info("ServiceContainer initialized successfully")
    
    def get_embedding_service(self):
        """Get the embedding service instance."""
        if not hasattr(self, '_initialized'):
            raise RuntimeError("ServiceContainer not properly initialized")
        return self.embedding_service
    
    def get_llm_service(self):
        """Get the LLM service instance."""
        if not hasattr(self, '_initialized'):
            raise RuntimeError("ServiceContainer not properly initialized")
        return self.llm_service
    
    def get_metrics(self) -> ServiceMetrics:
        """Get service metrics."""
        return self._metrics
    
    def health_check(self) -> Dict[str, bool]:
        """Perform health check on all services."""
        health_status = {}
        
        # Check embedding service
        try:
            if hasattr(self.embedding_service, 'is_healthy'):
                health_status['embedding'] = self.embedding_service.is_healthy()
            else:
                # Basic health check - try to get a simple embedding
                test_embedding = self.embedding_service.get_embedding("test")
                health_status['embedding'] = len(test_embedding) > 0
        except Exception as e:
            logger.warning(f"Embedding service health check failed: {e}")
            health_status['embedding'] = False
        
        # Check LLM service
        try:
            if hasattr(self.llm_service, 'is_healthy'):
                health_status['llm'] = self.llm_service.is_healthy()
            else:
                # Basic health check - try to generate a simple response
                test_response = self.llm_service.generate_response("test")
                health_status['llm'] = len(test_response) > 0
        except Exception as e:
            logger.warning(f"LLM service health check failed: {e}")
            health_status['llm'] = False
        
        return health_status
    
    def is_healthy(self) -> bool:
        """Check if the service container is healthy."""
        health_status = self.health_check()
        return all(health_status.values())


class ServiceFactory:
    """Factory for creating and managing service instances."""
    
    def __init__(self):
        """Initialize the service factory."""
        self._containers = {}
        self._default_container = None
        logger.info("ServiceFactory initialized")
    
    def create_optimized_container(self, total_prompts: int, profile: str = "balanced") -> ServiceContainer:
        """
        Create an optimized service container based on workload requirements.
        
        Args:
            total_prompts: Expected number of prompts to process
            profile: Performance profile ('speed', 'balanced', 'memory')
        
        Returns:
            ServiceContainer: Configured service container
        """
        logger.info(f"Creating optimized container for {total_prompts} prompts with {profile} profile")
        
        # Import services here to avoid circular imports
        from .embedding import EmbeddingService, EmbeddingServiceFactory
        from .llm import LLMService, LLMServiceFactory
        
        # Configure services based on profile and workload
        embedding_config = self._get_embedding_config(total_prompts, profile)
        llm_config = self._get_llm_config(total_prompts, profile)
        
        # Create optimized services
        embedding_service = EmbeddingServiceFactory.create_service(embedding_config)
        llm_service = LLMServiceFactory.create_service(llm_config)
        
        # Create and cache container
        container = ServiceContainer(embedding_service, llm_service)
        container_key = f"{total_prompts}_{profile}"
        self._containers[container_key] = container
        
        if self._default_container is None:
            self._default_container = container
        
        return container
    
    def get_default_container(self) -> ServiceContainer:
        """Get the default service container."""
        if self._default_container is None:
            self._default_container = self.create_optimized_container(1000, "balanced")
        return self._default_container
    
    def validate_service_health(self, container: ServiceContainer) -> Dict[str, bool]:
        """
        Validate the health of services in a container.
        
        Args:
            container: Service container to validate
        
        Returns:
            Dict[str, bool]: Health status for each service
        """
        return container.health_check()
    
    def _get_embedding_config(self, total_prompts: int, profile: str) -> Dict[str, Any]:
        """Get embedding service configuration based on requirements."""
        base_config = {
            "model": "text-embedding-ada-002",
            "batch_size": 100,
            "timeout": 30,
            "retry_attempts": 3
        }
        
        if profile == "speed":
            base_config.update({
                "batch_size": 200,
                "timeout": 15,
                "parallel_requests": 4
            })
        elif profile == "memory":
            base_config.update({
                "batch_size": 50,
                "timeout": 60,
                "parallel_requests": 1
            })
        
        # Scale batch size based on total prompts
        if total_prompts > 10000:
            base_config["batch_size"] = min(base_config["batch_size"] * 2, 500)
        
        return base_config
    
    def _get_llm_config(self, total_prompts: int, profile: str) -> Dict[str, Any]:
        """Get LLM service configuration based on requirements."""
        import os

        # Read API key and model from environment if available
        api_key = os.environ.get("DRENGR_API_KEY")
        model = os.environ.get("DRENGR_LLM_MODEL", "deepseek-v4-flash")
        base_url = os.environ.get("DRENGR_LLM_BASE_URL", "https://api.deepseek.com")

        base_config = {
            "model": model,
            "temperature": 0.7,
            "max_tokens": 300,
            "timeout": 30,
            "retry_attempts": 3,
            "base_url": base_url,
        }

        # Activate production mode if API key is set
        if api_key:
            base_config["type"] = "production"
            base_config["api_key"] = api_key
        
        if profile == "speed":
            base_config.update({
                "temperature": 0.5,
                "max_tokens": 150,
                "timeout": 15
            })
        elif profile == "memory":
            base_config.update({
                "max_tokens": 400,
                "timeout": 60
            })
        
        return base_config
    
    def cleanup(self):
        """Clean up resources and cached containers."""
        logger.info("Cleaning up ServiceFactory resources")
        self._containers.clear()
        self._default_container = None


# Global factory instance
_service_factory = None


def get_service_factory() -> ServiceFactory:
    """Get the global service factory instance."""
    global _service_factory
    if _service_factory is None:
        _service_factory = ServiceFactory()
    return _service_factory


def reset_service_factory():
    """Reset the global service factory (mainly for testing)."""
    global _service_factory
    if _service_factory:
        _service_factory.cleanup()
    _service_factory = None

