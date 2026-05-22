"""
Batch operations for improved performance.

This module implements batch processing for embeddings, similarity calculations,
and other expensive operations to reduce overhead and improve throughput.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from ..core.models import GeneratedPrompt, EmbeddingService
from ..core.exceptions import GenerationError


class BatchProcessor:
    """Generic batch processor for expensive operations."""
    
    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        """Initialize batch processor."""
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
    
    def process_in_batches(self, 
                          items: List[Any],
                          processor_func: Callable[[List[Any]], List[Any]],
                          progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Any]:
        """Process items in batches with optional progress tracking."""
        try:
            total_items = len(items)
            if total_items == 0:
                return []
            
            self.logger.info(f"Processing {total_items} items in batches of {self.batch_size}")
            
            results = []
            processed_count = 0
            
            # Process in batches
            for i in range(0, total_items, self.batch_size):
                batch = items[i:i + self.batch_size]
                
                start_time = time.time()
                batch_results = processor_func(batch)
                batch_time = time.time() - start_time
                
                results.extend(batch_results)
                processed_count += len(batch)
                
                # Progress callback
                if progress_callback:
                    progress_callback(processed_count, total_items)
                
                self.logger.debug(f"Processed batch {i//self.batch_size + 1}: "
                                f"{len(batch)} items in {batch_time:.2f}s")
            
            self.logger.info(f"Batch processing completed: {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise GenerationError(f"Batch processing failed: {e}") from e
    
    def process_in_parallel_batches(self,
                                  items: List[Any],
                                  processor_func: Callable[[List[Any]], List[Any]],
                                  progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Any]:
        """Process items in parallel batches for maximum throughput."""
        try:
            total_items = len(items)
            if total_items == 0:
                return []
            
            self.logger.info(f"Processing {total_items} items in parallel batches")
            
            # Create batches
            batches = [items[i:i + self.batch_size] for i in range(0, total_items, self.batch_size)]
            
            results = []
            processed_count = 0
            
            # Process batches in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(processor_func, batch): batch 
                    for batch in batches
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                        processed_count += len(batch)
                        
                        # Progress callback
                        if progress_callback:
                            progress_callback(processed_count, total_items)
                            
                    except Exception as e:
                        self.logger.error(f"Batch processing failed: {e}")
                        raise
            
            # Sort results to maintain order (if needed)
            # Note: This assumes results have an 'id' attribute for sorting
            if results and hasattr(results[0], 'id'):
                results.sort(key=lambda x: x.id)
            
            self.logger.info(f"Parallel batch processing completed: {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Parallel batch processing failed: {e}")
            raise GenerationError(f"Parallel batch processing failed: {e}") from e


class BatchEmbeddingService:
    """Batch embedding service for improved performance."""
    
    def __init__(self, embedding_service: EmbeddingService, batch_size: int = 32):
        """Initialize batch embedding service."""
        self.embedding_service = embedding_service
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Cache for computed embeddings
        self._embedding_cache: Dict[str, List[float]] = {}
    
    def compute_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of texts."""
        try:
            if not texts:
                return []
            
            # Check cache first
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                if text in self._embedding_cache:
                    cached_embeddings.append((i, self._embedding_cache[text]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            self.logger.debug(f"Cache hit rate: {len(cached_embeddings)}/{len(texts)} "
                            f"({len(cached_embeddings)/len(texts)*100:.1f}%)")
            
            # Compute embeddings for uncached texts
            new_embeddings = []
            if uncached_texts:
                # Try batch computation if supported
                if hasattr(self.embedding_service, 'compute_embeddings_batch'):
                    new_embeddings = self.embedding_service.compute_embeddings_batch(uncached_texts)
                else:
                    # Fallback to individual computation
                    for text in uncached_texts:
                        embedding = self.embedding_service.compute_embedding(text)
                        new_embeddings.append(embedding)
                
                # Cache new embeddings
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self._embedding_cache[text] = embedding
            
            # Combine cached and new embeddings in correct order
            all_embeddings = [None] * len(texts)
            
            # Place cached embeddings
            for i, embedding in cached_embeddings:
                all_embeddings[i] = embedding
            
            # Place new embeddings
            for i, embedding in zip(uncached_indices, new_embeddings):
                all_embeddings[i] = embedding
            
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"Batch embedding computation failed: {e}")
            raise GenerationError(f"Batch embedding failed: {e}") from e
    
    def compute_similarity_matrix_batch(self, embeddings: List[List[float]]) -> np.ndarray:
        """Compute similarity matrix for a batch of embeddings."""
        try:
            if not embeddings:
                return np.array([])
            
            # Convert to numpy array for efficient computation
            embedding_matrix = np.array(embeddings, dtype=np.float32)
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
            normalized_embeddings = embedding_matrix / (norms + 1e-8)
            
            # Compute similarity matrix using matrix multiplication
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
            
            # Ensure diagonal is 1.0 (self-similarity)
            np.fill_diagonal(similarity_matrix, 1.0)
            
            return similarity_matrix
            
        except Exception as e:
            self.logger.error(f"Batch similarity computation failed: {e}")
            raise GenerationError(f"Batch similarity failed: {e}") from e
    
    def find_similar_prompts_batch(self, 
                                 prompts: List[GeneratedPrompt],
                                 similarity_threshold: float = 0.8) -> Dict[int, List[Tuple[int, float]]]:
        """Find similar prompts in batch for all prompts."""
        try:
            if not prompts:
                return {}
            
            # Extract texts and compute embeddings
            texts = [prompt.prompt for prompt in prompts]
            embeddings = self.compute_embeddings_batch(texts)
            
            # Compute similarity matrix
            similarity_matrix = self.compute_similarity_matrix_batch(embeddings)
            
            # Find similar prompts for each prompt
            similar_prompts = {}
            
            for i, prompt in enumerate(prompts):
                similarities = similarity_matrix[i]
                
                # Find prompts above threshold (excluding self)
                similar_indices = np.where((similarities >= similarity_threshold) & 
                                         (np.arange(len(similarities)) != i))[0]
                
                similar_list = []
                for j in similar_indices:
                    similar_list.append((prompts[j].id, float(similarities[j])))
                
                # Sort by similarity (highest first)
                similar_list.sort(key=lambda x: x[1], reverse=True)
                
                similar_prompts[prompt.id] = similar_list
            
            self.logger.info(f"Found similar prompts for {len(prompts)} prompts")
            return similar_prompts
            
        except Exception as e:
            self.logger.error(f"Batch similar prompt finding failed: {e}")
            raise GenerationError(f"Batch similar prompt finding failed: {e}") from e
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        cache_size = len(self._embedding_cache)
        self._embedding_cache.clear()
        self.logger.info(f"Cleared embedding cache ({cache_size} entries)")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._embedding_cache),
            "memory_usage_mb": self._estimate_cache_memory_usage()
        }
    
    def _estimate_cache_memory_usage(self) -> int:
        """Estimate cache memory usage in MB."""
        if not self._embedding_cache:
            return 0
        
        # Estimate based on first embedding
        first_embedding = next(iter(self._embedding_cache.values()))
        embedding_size = len(first_embedding) * 4  # 4 bytes per float32
        
        total_size = len(self._embedding_cache) * embedding_size
        return total_size // (1024 * 1024)  # Convert to MB


class BatchValidationProcessor:
    """Batch processor for validation operations."""
    
    def __init__(self, batch_size: int = 100):
        """Initialize batch validation processor."""
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
    
    def validate_prompts_batch(self, 
                             prompts: List[GeneratedPrompt],
                             validation_func: Callable[[List[GeneratedPrompt]], List[bool]]) -> List[bool]:
        """Validate prompts in batches."""
        try:
            if not prompts:
                return []
            
            self.logger.info(f"Validating {len(prompts)} prompts in batches")
            
            results = []
            
            for i in range(0, len(prompts), self.batch_size):
                batch = prompts[i:i + self.batch_size]
                batch_results = validation_func(batch)
                results.extend(batch_results)
            
            self.logger.info(f"Batch validation completed: {sum(results)}/{len(results)} valid")
            return results
            
        except Exception as e:
            self.logger.error(f"Batch validation failed: {e}")
            raise GenerationError(f"Batch validation failed: {e}") from e
    
    def compute_quality_metrics_batch(self, prompts: List[GeneratedPrompt]) -> Dict[str, float]:
        """Compute quality metrics for prompts in batch."""
        try:
            if not prompts:
                return {}
            
            # Batch compute various metrics
            word_counts = [len(prompt.prompt.split()) for prompt in prompts]
            char_counts = [len(prompt.prompt) for prompt in prompts]

            has_normalized = [bool(prompt.normalized_prompt and prompt.normalized_prompt.strip())
                            for prompt in prompts]

            metrics = {
                "avg_word_count": sum(word_counts) / len(word_counts),
                "avg_char_count": sum(char_counts) / len(char_counts),
                "min_word_count": min(word_counts),
                "max_word_count": max(word_counts),
                "normalized_prompt_coverage": sum(has_normalized) / len(has_normalized),
                "total_prompts": len(prompts)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Batch quality metrics computation failed: {e}")
            return {}