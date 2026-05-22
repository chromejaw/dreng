"""Services package initialization - Production-ready implementations."""

# Import core services - these should now exist
from .embedding import (
    EmbeddingService,
    EmbeddingServiceFactory,
    MockEmbeddingService,
    ProductionEmbeddingService
)

from .llm import (
    LLMService,
    LLMServiceFactory,
    MockLLMService,
    ProductionLLMService,
    ParaphraseTemplateManager
)

from .golden_response import (
    GoldenResponseGenerator,
    GoldenResponseFactory,
    MockGoldenResponseGenerator,
    ProductionGoldenResponseGenerator
)

from .service_factory import (
    ServiceFactory,
    ServiceContainer,
    get_service_factory,
    reset_service_factory
)

# Make embedding module accessible for backward compatibility
class EmbeddingModule:
    """Embedding module for compatibility."""
    EmbeddingService = EmbeddingService
    MockEmbeddingService = MockEmbeddingService

embedding = EmbeddingModule()

__all__ = [
    'EmbeddingService',
    'EmbeddingServiceFactory', 
    'MockEmbeddingService',
    'ProductionEmbeddingService',
    'LLMService',
    'LLMServiceFactory',
    'MockLLMService',
    'ProductionLLMService',
    'ParaphraseTemplateManager',
    'GoldenResponseGenerator',
    'GoldenResponseFactory',
    'MockGoldenResponseGenerator',
    'ProductionGoldenResponseGenerator',
    'ServiceFactory',
    'ServiceContainer',
    'get_service_factory',
    'reset_service_factory',
    'embedding'
]