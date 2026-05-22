"""
Memory management and optimization utilities.

This module provides memory optimization, garbage collection management,
and efficient data structures for large dataset generation.
"""

import gc
import logging
import psutil
import weakref
from typing import Dict, Any, Optional, List, Tuple, Union
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np

from ..core.models import GeneratedPrompt
from ..core.exceptions import GenerationError


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    memory_percent: float
    process_memory_mb: float
    gc_collections: Dict[int, int]


class MemoryOptimizer:
    """Memory optimization and monitoring utilities."""
    
    def __init__(self, memory_threshold_percent: float = 80.0):
        """Initialize memory optimizer."""
        self.memory_threshold_percent = memory_threshold_percent
        self.logger = logging.getLogger(__name__)
        
        # Track memory usage over time
        self.memory_history: List[MemoryStats] = []
        self.max_history_size = 100
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        try:
            # System memory
            memory = psutil.virtual_memory()
            
            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Garbage collection stats
            gc_stats = {i: gc.get_count()[i] for i in range(3)}
            
            stats = MemoryStats(
                total_memory_mb=memory.total / (1024 * 1024),
                available_memory_mb=memory.available / (1024 * 1024),
                used_memory_mb=memory.used / (1024 * 1024),
                memory_percent=memory.percent,
                process_memory_mb=process_memory,
                gc_collections=gc_stats
            )
            
            # Add to history
            self.memory_history.append(stats)
            if len(self.memory_history) > self.max_history_size:
                self.memory_history.pop(0)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return MemoryStats(0, 0, 0, 0, 0, {})
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        stats = self.get_memory_stats()
        return stats.memory_percent > self.memory_threshold_percent
    
    def optimize_memory(self, force_gc: bool = False) -> Dict[str, Any]:
        """Optimize memory usage."""
        try:
            stats_before = self.get_memory_stats()
            
            optimization_results = {
                "memory_before_mb": stats_before.process_memory_mb,
                "memory_after_mb": 0,
                "memory_freed_mb": 0,
                "gc_collections": 0,
                "optimizations_applied": []
            }
            
            # Force garbage collection if requested or under pressure
            if force_gc or self.check_memory_pressure():
                collected = gc.collect()
                optimization_results["gc_collections"] = collected
                optimization_results["optimizations_applied"].append("garbage_collection")
                
                self.logger.info(f"Garbage collection freed {collected} objects")
            
            # Get stats after optimization
            stats_after = self.get_memory_stats()
            optimization_results["memory_after_mb"] = stats_after.process_memory_mb
            optimization_results["memory_freed_mb"] = (
                stats_before.process_memory_mb - stats_after.process_memory_mb
            )
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return {"error": str(e)}
    
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        
        if not self.memory_history:
            return recommendations
        
        current_stats = self.memory_history[-1]
        
        # High memory usage
        if current_stats.memory_percent > 90:
            recommendations.append("Critical: System memory usage above 90%")
            recommendations.append("Consider reducing batch sizes")
            recommendations.append("Enable aggressive garbage collection")
        elif current_stats.memory_percent > 80:
            recommendations.append("Warning: System memory usage above 80%")
            recommendations.append("Monitor memory usage closely")
        
        # High process memory
        if current_stats.process_memory_mb > 2000:  # 2GB
            recommendations.append("Process using over 2GB memory")
            recommendations.append("Consider processing in smaller chunks")
        
        # Memory growth trend
        if len(self.memory_history) >= 10:
            recent_usage = [s.process_memory_mb for s in self.memory_history[-10:]]
            if recent_usage[-1] > recent_usage[0] * 1.5:  # 50% growth
                recommendations.append("Memory usage growing rapidly")
                recommendations.append("Check for memory leaks")
        
        return recommendations
    
    def estimate_memory_for_prompts(self, num_prompts: int) -> float:
        """Estimate memory usage for a given number of prompts."""
        # Rough estimate: ~2KB per prompt (including embeddings)
        estimated_mb = (num_prompts * 2048) / (1024 * 1024)
        return estimated_mb
    
    def suggest_batch_size(self, total_items: int, available_memory_mb: float) -> int:
        """Suggest optimal batch size based on available memory."""
        # Reserve 20% of available memory for other operations
        usable_memory_mb = available_memory_mb * 0.8
        
        # Estimate memory per item (conservative estimate)
        memory_per_item_mb = 0.5  # 500KB per item
        
        # Calculate batch size
        suggested_batch_size = max(1, int(usable_memory_mb / memory_per_item_mb))
        
        # Cap at reasonable limits
        suggested_batch_size = min(suggested_batch_size, 1000)  # Max 1000
        suggested_batch_size = max(suggested_batch_size, 10)    # Min 10
        
        return suggested_batch_size


class CacheManager:
    """Efficient cache management with memory-aware eviction."""
    
    def __init__(self, max_size: int = 10000, max_memory_mb: float = 500.0):
        """Initialize cache manager."""
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.logger = logging.getLogger(__name__)
        
        # Use OrderedDict for LRU behavior
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._memory_usage_mb = 0.0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self._cache:
            # Move to end (most recently used)
            value = self._cache.pop(key)
            self._cache[key] = value
            self.hits += 1
            return value
        else:
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with memory-aware eviction."""
        try:
            # Estimate memory usage of new item
            item_memory = self._estimate_memory_usage(value)
            
            # Remove existing item if updating
            if key in self._cache:
                old_value = self._cache.pop(key)
                old_memory = self._estimate_memory_usage(old_value)
                self._memory_usage_mb -= old_memory
            
            # Evict items if necessary
            while (len(self._cache) >= self.max_size or 
                   self._memory_usage_mb + item_memory > self.max_memory_mb):
                if not self._cache:
                    break
                self._evict_lru()
            
            # Add new item
            self._cache[key] = value
            self._memory_usage_mb += item_memory
            
        except Exception as e:
            self.logger.error(f"Cache put failed: {e}")
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self._cache:
            key, value = self._cache.popitem(last=False)  # Remove first (oldest)
            memory_freed = self._estimate_memory_usage(value)
            self._memory_usage_mb -= memory_freed
            self.evictions += 1
            
            self.logger.debug(f"Evicted cache item: {key} (freed {memory_freed:.2f}MB)")
    
    def _estimate_memory_usage(self, value: Any) -> float:
        """Estimate memory usage of a value in MB."""
        try:
            if isinstance(value, (list, tuple)):
                if value and isinstance(value[0], (int, float)):
                    # Assume numeric list (like embeddings)
                    return len(value) * 4 / (1024 * 1024)  # 4 bytes per float
                else:
                    return len(str(value)) / (1024 * 1024)  # Rough string estimate
            elif isinstance(value, str):
                return len(value) / (1024 * 1024)
            elif isinstance(value, np.ndarray):
                return value.nbytes / (1024 * 1024)
            else:
                # Rough estimate based on string representation
                return len(str(value)) / (1024 * 1024)
        except:
            return 0.001  # 1KB default estimate
    
    def clear(self) -> None:
        """Clear all cache entries."""
        cleared_count = len(self._cache)
        self._cache.clear()
        self._memory_usage_mb = 0.0
        self.logger.info(f"Cleared cache: {cleared_count} items")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "memory_usage_mb": self._memory_usage_mb,
            "max_memory_mb": self.max_memory_mb,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "memory_utilization": self._memory_usage_mb / self.max_memory_mb
        }


class EfficientPromptStorage:
    """Memory-efficient storage for large numbers of prompts."""
    
    def __init__(self, use_compression: bool = True):
        """Initialize efficient prompt storage."""
        self.use_compression = use_compression
        self.logger = logging.getLogger(__name__)
        
        # Separate storage for different data types
        self._prompts: Dict[int, GeneratedPrompt] = {}
        self._embeddings: Dict[int, np.ndarray] = {}
        
        # Weak references for automatic cleanup
        self._weak_refs: Dict[int, weakref.ref] = {}
    
    def store_prompt(self, prompt: GeneratedPrompt) -> None:
        """Store prompt efficiently."""
        try:
            self._prompts[prompt.id] = prompt
            
            # Create weak reference for automatic cleanup
            def cleanup(ref):
                if prompt.id in self._weak_refs:
                    del self._weak_refs[prompt.id]
            
            self._weak_refs[prompt.id] = weakref.ref(prompt, cleanup)
            
        except Exception as e:
            self.logger.error(f"Failed to store prompt {prompt.id}: {e}")
    
    def store_embedding(self, prompt_id: int, embedding: List[float]) -> None:
        """Store embedding efficiently."""
        try:
            # Convert to numpy array for memory efficiency
            embedding_array = np.array(embedding, dtype=np.float32)
            
            # Optionally compress (quantize) embedding
            if self.use_compression:
                embedding_array = self._compress_embedding(embedding_array)
            
            self._embeddings[prompt_id] = embedding_array
            
        except Exception as e:
            self.logger.error(f"Failed to store embedding for prompt {prompt_id}: {e}")
    
    def get_prompt(self, prompt_id: int) -> Optional[GeneratedPrompt]:
        """Get prompt by ID."""
        return self._prompts.get(prompt_id)
    
    def get_embedding(self, prompt_id: int) -> Optional[np.ndarray]:
        """Get embedding by prompt ID."""
        embedding = self._embeddings.get(prompt_id)
        
        if embedding is not None and self.use_compression:
            # Decompress if needed
            embedding = self._decompress_embedding(embedding)
        
        return embedding
    
    def _compress_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Compress embedding using quantization."""
        # Simple int8 quantization
        # Normalize to [-1, 1] range
        norm = np.linalg.norm(embedding)
        if norm > 0:
            normalized = embedding / norm
        else:
            normalized = embedding
        
        # Quantize to int8
        quantized = np.clip(normalized * 127, -128, 127).astype(np.int8)
        return quantized
    
    def _decompress_embedding(self, compressed: np.ndarray) -> np.ndarray:
        """Decompress quantized embedding."""
        # Convert back to float32
        decompressed = compressed.astype(np.float32) / 127.0
        return decompressed
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        try:
            prompts_memory = sum(
                len(str(prompt)) for prompt in self._prompts.values()
            ) / (1024 * 1024)  # MB
            
            embeddings_memory = sum(
                embedding.nbytes for embedding in self._embeddings.values()
            ) / (1024 * 1024)  # MB
            
            return {
                "prompts_memory_mb": prompts_memory,
                "embeddings_memory_mb": embeddings_memory,
                "total_memory_mb": prompts_memory + embeddings_memory,
                "num_prompts": len(self._prompts),
                "num_embeddings": len(self._embeddings)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate memory usage: {e}")
            return {}
    
    def cleanup_orphaned_data(self) -> int:
        """Clean up orphaned data and return number of items cleaned."""
        try:
            # Find orphaned embeddings (no corresponding prompt)
            orphaned_embeddings = []
            for prompt_id in self._embeddings:
                if prompt_id not in self._prompts:
                    orphaned_embeddings.append(prompt_id)
            
            # Remove orphaned embeddings
            for prompt_id in orphaned_embeddings:
                del self._embeddings[prompt_id]
            
            # Clean up dead weak references
            dead_refs = []
            for prompt_id, ref in self._weak_refs.items():
                if ref() is None:
                    dead_refs.append(prompt_id)
            
            for prompt_id in dead_refs:
                del self._weak_refs[prompt_id]
                if prompt_id in self._prompts:
                    del self._prompts[prompt_id]
                if prompt_id in self._embeddings:
                    del self._embeddings[prompt_id]
            
            total_cleaned = len(orphaned_embeddings) + len(dead_refs)
            
            if total_cleaned > 0:
                self.logger.info(f"Cleaned up {total_cleaned} orphaned items")
            
            return total_cleaned
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return 0