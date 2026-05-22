"""
Exception hierarchy for drengr library.

Provides comprehensive error handling with specific exception types
for different failure modes during dataset generation.
"""


class DrengrError(Exception):
    """Base exception for drengr library."""
    pass


class DatasetGenerationError(DrengrError):
    """Base exception for dataset generation errors."""
    pass


class GenerationError(DrengrError):
    """Errors during dataset generation."""
    pass


class ConfigurationError(DrengrError):
    """Configuration or setup errors."""
    pass


class ValidationError(DrengrError):
    """Validation failures."""
    pass


class BackendError(DrengrError):
    """Backend service errors."""
    pass


class EmbeddingServiceError(BackendError):
    """Embedding service specific errors."""
    pass


class LLMServiceError(BackendError):
    """LLM service specific errors."""
    pass


class SimilarityBandViolationError(DatasetGenerationError):
    """Raised when prompts don't meet similarity requirements."""
    pass


class CategoryCountError(DatasetGenerationError):
    """Raised when category counts don't match targets."""
    pass


class GoldenResponseError(DatasetGenerationError):
    """Raised when golden response generation fails."""
    pass


class TemporalDataError(DatasetGenerationError):
    """Raised when temporal data generation fails."""
    pass