"""
Ensemble embedding service and calibration utilities.

This module provides ensemble embedding services that combine multiple
embedding models for improved accuracy and robustness.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

from ..core.models import EmbeddingService
from ..core.exceptions import EmbeddingServiceError


class EnsembleEmbeddingService(EmbeddingService):
    """Ensemble embedding service combining multiple embedding models."""
    
    def __init__(self, 
                 primary_service: EmbeddingService,
                 secondary_service: EmbeddingService,
                 weight_primary: float = 0.7,
                 weight_secondary: float = 0.3):
        """
        Initialize ensemble embedding service.
        
        Args:
            primary_service: Primary embedding service
            secondary_service: Secondary embedding service  
            weight_primary: Weight for primary service (default: 0.7)
            weight_secondary: Weight for secondary service (default: 0.3)
        """
        self.primary_service = primary_service
        self.secondary_service = secondary_service
        self.weight_primary = weight_primary
        self.weight_secondary = weight_secondary
        
        # Normalize weights
        total_weight = weight_primary + weight_secondary
        self.weight_primary = weight_primary / total_weight
        self.weight_secondary = weight_secondary / total_weight
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized ensemble with weights {self.weight_primary:.2f}/{self.weight_secondary:.2f}")
    
    def compute_embedding(self, text: str) -> List[float]:
        """Compute ensemble embedding by combining multiple services."""
        try:
            # Get embeddings from both services
            primary_embedding = self.primary_service.compute_embedding(text)
            secondary_embedding = self.secondary_service.compute_embedding(text)
            
            # Ensure embeddings have same dimension
            if len(primary_embedding) != len(secondary_embedding):
                # Pad shorter embedding with zeros or truncate longer one
                target_dim = min(len(primary_embedding), len(secondary_embedding))
                primary_embedding = primary_embedding[:target_dim]
                secondary_embedding = secondary_embedding[:target_dim]
            
            # Compute weighted average
            primary_vec = np.array(primary_embedding, dtype=np.float32)
            secondary_vec = np.array(secondary_embedding, dtype=np.float32)
            
            ensemble_vec = (self.weight_primary * primary_vec + 
                          self.weight_secondary * secondary_vec)
            
            # Normalize to unit vector
            norm = np.linalg.norm(ensemble_vec)
            if norm > 0:
                ensemble_vec = ensemble_vec / norm
            
            return ensemble_vec.tolist()
            
        except Exception as e:
            self.logger.error(f"Ensemble embedding computation failed: {e}")
            # Fallback to primary service only
            try:
                return self.primary_service.compute_embedding(text)
            except Exception as fallback_error:
                raise EmbeddingServiceError(f"Ensemble and primary service failed: {e}, {fallback_error}") from e
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between ensemble embeddings."""
        try:
            vec1 = np.array(embedding1, dtype=np.float32)
            vec2 = np.array(embedding2, dtype=np.float32)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            raise EmbeddingServiceError(f"Similarity computation failed: {e}") from e
    
    def get_model_info(self) -> Dict[str, str]:
        """Get ensemble model metadata."""
        primary_info = self.primary_service.get_model_info()
        secondary_info = self.secondary_service.get_model_info()
        
        return {
            "model": f"ensemble({primary_info.get('model', 'unknown')}+{secondary_info.get('model', 'unknown')})",
            "provider": "ensemble",
            "primary_model": primary_info.get("model", "unknown"),
            "secondary_model": secondary_info.get("model", "unknown"),
            "primary_weight": str(self.weight_primary),
            "secondary_weight": str(self.weight_secondary),
            "full_id": f"ensemble-{primary_info.get('model', 'unknown')}-{secondary_info.get('model', 'unknown')}"
        }
    
    def batch_compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Compute ensemble embeddings for multiple texts."""
        try:
            # Get batch embeddings from both services
            if hasattr(self.primary_service, 'batch_compute_embeddings'):
                primary_embeddings = self.primary_service.batch_compute_embeddings(texts)
            else:
                primary_embeddings = [self.primary_service.compute_embedding(text) for text in texts]
            
            if hasattr(self.secondary_service, 'batch_compute_embeddings'):
                secondary_embeddings = self.secondary_service.batch_compute_embeddings(texts)
            else:
                secondary_embeddings = [self.secondary_service.compute_embedding(text) for text in texts]
            
            # Combine embeddings
            ensemble_embeddings = []
            for primary_emb, secondary_emb in zip(primary_embeddings, secondary_embeddings):
                # Ensure same dimension
                target_dim = min(len(primary_emb), len(secondary_emb))
                primary_vec = np.array(primary_emb[:target_dim], dtype=np.float32)
                secondary_vec = np.array(secondary_emb[:target_dim], dtype=np.float32)
                
                # Weighted average
                ensemble_vec = (self.weight_primary * primary_vec + 
                              self.weight_secondary * secondary_vec)
                
                # Normalize
                norm = np.linalg.norm(ensemble_vec)
                if norm > 0:
                    ensemble_vec = ensemble_vec / norm
                
                ensemble_embeddings.append(ensemble_vec.tolist())
            
            return ensemble_embeddings
            
        except Exception as e:
            self.logger.error(f"Batch ensemble embedding failed: {e}")
            # Fallback to primary service
            if hasattr(self.primary_service, 'batch_compute_embeddings'):
                return self.primary_service.batch_compute_embeddings(texts)
            else:
                return [self.primary_service.compute_embedding(text) for text in texts]


class EnsembleCalibrator:
    """Calibrator for creating and optimizing ensemble embedding services."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_ensemble(self, 
                       primary_backend: str,
                       secondary_backend: str,
                       primary_model: Optional[str] = None,
                       secondary_model: Optional[str] = None) -> EnsembleEmbeddingService:
        """Create ensemble embedding service from backend specifications."""
        from ..services.embedding import EmbeddingServiceFactory
        
        try:
            # Create primary service
            primary_config = {
                "provider": primary_backend,
                "model": primary_model
            }
            primary_service = EmbeddingServiceFactory.create_from_config(primary_config)
            
            # Create secondary service
            secondary_config = {
                "provider": secondary_backend,
                "model": secondary_model
            }
            secondary_service = EmbeddingServiceFactory.create_from_config(secondary_config)
            
            # Create ensemble
            ensemble = EnsembleEmbeddingService(primary_service, secondary_service)
            
            self.logger.info(f"Created ensemble: {primary_backend}+{secondary_backend}")
            return ensemble
            
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to create ensemble: {e}") from e
    
    def optimize_weights(self, 
                        ensemble: EnsembleEmbeddingService,
                        test_texts: List[str],
                        ground_truth_similarities: List[Tuple[int, int, float]]) -> Tuple[float, float]:
        """
        Optimize ensemble weights based on ground truth similarities.
        
        Args:
            ensemble: Ensemble service to optimize
            test_texts: List of test texts
            ground_truth_similarities: List of (idx1, idx2, similarity) tuples
            
        Returns:
            Optimized (primary_weight, secondary_weight) tuple
        """
        try:
            best_weights = (0.7, 0.3)
            best_error = float('inf')
            
            # Grid search over weight combinations
            for primary_weight in np.arange(0.1, 1.0, 0.1):
                secondary_weight = 1.0 - primary_weight
                
                # Update ensemble weights
                ensemble.weight_primary = primary_weight
                ensemble.weight_secondary = secondary_weight
                
                # Compute embeddings
                embeddings = ensemble.batch_compute_embeddings(test_texts)
                
                # Calculate error against ground truth
                total_error = 0.0
                for idx1, idx2, true_sim in ground_truth_similarities:
                    if idx1 < len(embeddings) and idx2 < len(embeddings):
                        pred_sim = ensemble.compute_similarity(embeddings[idx1], embeddings[idx2])
                        total_error += (pred_sim - true_sim) ** 2
                
                avg_error = total_error / len(ground_truth_similarities)
                
                if avg_error < best_error:
                    best_error = avg_error
                    best_weights = (primary_weight, secondary_weight)
            
            # Set optimal weights
            ensemble.weight_primary = best_weights[0]
            ensemble.weight_secondary = best_weights[1]
            
            self.logger.info(f"Optimized weights: {best_weights[0]:.2f}/{best_weights[1]:.2f}, error: {best_error:.4f}")
            return best_weights
            
        except Exception as e:
            self.logger.error(f"Weight optimization failed: {e}")
            return (0.7, 0.3)  # Return default weights
    
    def evaluate_ensemble_performance(self, 
                                    ensemble: EnsembleEmbeddingService,
                                    test_cases: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate ensemble performance on test cases."""
        try:
            results = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "avg_similarity_error": 0.0
            }
            
            if not test_cases:
                return results
            
            correct_predictions = 0
            total_similarity_error = 0.0
            
            for test_case in test_cases:
                text1 = test_case.get("text1", "")
                text2 = test_case.get("text2", "")
                expected_similarity = test_case.get("similarity", 0.0)
                
                # Compute embeddings and similarity
                emb1 = ensemble.compute_embedding(text1)
                emb2 = ensemble.compute_embedding(text2)
                predicted_similarity = ensemble.compute_similarity(emb1, emb2)
                
                # Calculate error
                similarity_error = abs(predicted_similarity - expected_similarity)
                total_similarity_error += similarity_error
                
                # Binary classification accuracy (threshold at 0.5)
                predicted_match = predicted_similarity > 0.5
                expected_match = expected_similarity > 0.5
                
                if predicted_match == expected_match:
                    correct_predictions += 1
            
            # Calculate metrics
            results["accuracy"] = correct_predictions / len(test_cases)
            results["avg_similarity_error"] = total_similarity_error / len(test_cases)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Performance evaluation failed: {e}")
            return {"error": str(e)}