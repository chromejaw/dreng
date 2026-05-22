"""
ANN (Approximate Nearest Neighbor) backend implementations.

This module provides pluggable ANN backends using faiss and hnswlib for efficient
similarity operations, with automatic backend selection based on dataset size.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

from ..core.exceptions import BackendError

# Optional imports for different ANN backends
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False


class ANNBackend(ABC):
    """Abstract base class for ANN backends."""
    
    @abstractmethod
    def build_index(self, embeddings: np.ndarray) -> None:
        """Build the ANN index from embeddings."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors.
        
        Returns:
            Tuple of (distances, indices)
        """
        pass
    
    @abstractmethod
    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """Add new embeddings to the index."""
        pass
    
    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        """Get backend information."""
        pass


class FaissBackend(ANNBackend):
    """Faiss-based ANN backend for large-scale similarity search."""
    
    def __init__(self, dimension: int, index_type: str = "IVF-PQ"):
        """Initialize Faiss backend.
        
        Args:
            dimension: Embedding dimension
            index_type: Type of Faiss index ("IVF-PQ" | "HNSW" | "Flat")
        """
        if not FAISS_AVAILABLE:
            raise BackendError("faiss not available. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.is_trained = False
        
        self.logger = logging.getLogger(__name__)
        self._create_index()
    
    def _create_index(self) -> None:
        """Create the appropriate Faiss index."""
        if self.index_type == "IVF-PQ":
            # IVF with Product Quantization - good for large datasets
            nlist = 100  # number of clusters
            m = 8        # number of subquantizers
            nbits = 8    # bits per subquantizer
            
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits)
            
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World - good balance of speed/accuracy
            M = 16  # number of connections
            self.index = faiss.IndexHNSWFlat(self.dimension, M)
            
        elif self.index_type == "Flat":
            # Exact search - good for small datasets
            self.index = faiss.IndexFlatL2(self.dimension)
            
        else:
            raise BackendError(f"Unknown Faiss index type: {self.index_type}")
        
        self.logger.info(f"Created Faiss {self.index_type} index for dimension {self.dimension}")
    
    def build_index(self, embeddings: np.ndarray) -> None:
        """Build the Faiss index from embeddings."""
        try:
            embeddings = embeddings.astype(np.float32)
            
            # Train the index if needed
            if not self.is_trained and hasattr(self.index, 'train'):
                self.logger.info(f"Training Faiss index with {len(embeddings)} embeddings...")
                self.index.train(embeddings)
                self.is_trained = True
            
            # Add embeddings to index
            self.index.add(embeddings)
            self.logger.info(f"Added {len(embeddings)} embeddings to Faiss index")
            
        except Exception as e:
            raise BackendError(f"Failed to build Faiss index: {e}") from e
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors using Faiss."""
        try:
            query = query_embedding.astype(np.float32).reshape(1, -1)
            distances, indices = self.index.search(query, k)
            return distances[0], indices[0]
            
        except Exception as e:
            raise BackendError(f"Failed to search Faiss index: {e}") from e
    
    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """Add new embeddings to the Faiss index."""
        try:
            embeddings = embeddings.astype(np.float32)
            self.index.add(embeddings)
            
        except Exception as e:
            raise BackendError(f"Failed to add embeddings to Faiss index: {e}") from e
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get Faiss backend information."""
        return {
            "backend": "faiss",
            "index_type": self.index_type,
            "dimension": self.dimension,
            "total_embeddings": self.index.ntotal if self.index else 0,
            "is_trained": self.is_trained
        }


class HNSWLibBackend(ANNBackend):
    """HNSWLib-based ANN backend for fast similarity search."""
    
    def __init__(self, dimension: int, max_elements: int = 10000, M: int = 16, ef_construction: int = 200):
        """Initialize HNSWLib backend.
        
        Args:
            dimension: Embedding dimension
            max_elements: Maximum number of elements
            M: Number of bi-directional links for each node
            ef_construction: Size of dynamic candidate list
        """
        if not HNSWLIB_AVAILABLE:
            raise BackendError("hnswlib not available. Install with: pip install hnswlib")
        
        self.dimension = dimension
        self.max_elements = max_elements
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = 64  # Default search parameter
        
        self.logger = logging.getLogger(__name__)
        self._create_index()
    
    def _create_index(self) -> None:
        """Create the HNSWLib index."""
        try:
            self.index = hnswlib.Index(space='cosine', dim=self.dimension)
            self.index.init_index(
                max_elements=self.max_elements,
                M=self.M,
                ef_construction=self.ef_construction
            )
            self.index.set_ef(self.ef_search)
            
            self.logger.info(f"Created HNSWLib index: dim={self.dimension}, M={self.M}, ef_construction={self.ef_construction}")
            
        except Exception as e:
            raise BackendError(f"Failed to create HNSWLib index: {e}") from e
    
    def build_index(self, embeddings: np.ndarray) -> None:
        """Build the HNSWLib index from embeddings."""
        try:
            embeddings = embeddings.astype(np.float32)
            ids = np.arange(len(embeddings))
            
            self.index.add_items(embeddings, ids)
            self.logger.info(f"Added {len(embeddings)} embeddings to HNSWLib index")
            
        except Exception as e:
            raise BackendError(f"Failed to build HNSWLib index: {e}") from e
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors using HNSWLib."""
        try:
            query = query_embedding.astype(np.float32)
            indices, distances = self.index.knn_query(query, k=k)
            return distances[0], indices[0]
            
        except Exception as e:
            raise BackendError(f"Failed to search HNSWLib index: {e}") from e
    
    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """Add new embeddings to the HNSWLib index."""
        try:
            embeddings = embeddings.astype(np.float32)
            current_count = self.index.get_current_count()
            ids = np.arange(current_count, current_count + len(embeddings))
            
            self.index.add_items(embeddings, ids)
            
        except Exception as e:
            raise BackendError(f"Failed to add embeddings to HNSWLib index: {e}") from e
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get HNSWLib backend information."""
        return {
            "backend": "hnswlib",
            "dimension": self.dimension,
            "max_elements": self.max_elements,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "current_count": self.index.get_current_count() if self.index else 0
        }


class ANNBackendFactory:
    """Factory for creating ANN backend instances."""
    
    @staticmethod
    def create_backend(backend_type: str, dimension: int, dataset_size: int = 1000, **kwargs) -> ANNBackend:
        """Create appropriate ANN backend.
        
        Args:
            backend_type: Backend type ("auto" | "faiss" | "hnswlib")
            dimension: Embedding dimension
            dataset_size: Expected dataset size for auto-selection
            **kwargs: Additional backend-specific parameters
            
        Returns:
            ANNBackend instance
        """
        if backend_type == "auto":
            # Auto-select based on dataset size
            if dataset_size > 10000:
                backend_type = "faiss"
            else:
                backend_type = "hnswlib"
        
        if backend_type == "faiss":
            index_type = kwargs.get("index_type", "IVF-PQ" if dataset_size > 10000 else "HNSW")
            return FaissBackend(dimension, index_type)
            
        elif backend_type == "hnswlib":
            max_elements = kwargs.get("max_elements", max(dataset_size * 2, 10000))
            M = kwargs.get("M", 16)
            ef_construction = kwargs.get("ef_construction", 200)
            return HNSWLibBackend(dimension, max_elements, M, ef_construction)
            
        else:
            raise BackendError(f"Unknown ANN backend type: {backend_type}")
    
    @staticmethod
    def get_available_backends() -> List[str]:
        """Get list of available ANN backends."""
        backends = []
        
        if FAISS_AVAILABLE:
            backends.append("faiss")
        
        if HNSWLIB_AVAILABLE:
            backends.append("hnswlib")
        
        return backends
    
    @staticmethod
    def get_recommended_backend(dataset_size: int) -> str:
        """Get recommended backend for given dataset size."""
        available = ANNBackendFactory.get_available_backends()
        
        if dataset_size > 10000 and "faiss" in available:
            return "faiss"
        elif "hnswlib" in available:
            return "hnswlib"
        elif "faiss" in available:
            return "faiss"
        else:
            raise BackendError("No ANN backends available")