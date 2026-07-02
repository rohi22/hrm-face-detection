"""
Similarity Computation Module

Provides face embedding similarity calculation using cosine similarity.
Optimized for L2-normalized ArcFace embeddings.
"""

import numpy as np
from typing import List
import logging
import math

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """
    Computes similarity between face embeddings using cosine similarity.
    
    For L2-normalized vectors (as produced by ArcFace), cosine similarity
    simplifies to the dot product: similarity = dot(a, b)
    
    This avoids redundant normalization computations.
    """
    
    @staticmethod
    def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Formula (for L2-normalized vectors):
            similarity = dot(a, b)
        
        Formula (general case):
            similarity = dot(a, b) / (||a|| * ||b||)
        
        Since ArcFace embeddings are already L2-normalized (||a|| = ||b|| = 1),
        we can use the simplified dot product formula.
        
        Args:
            embedding1: First 512-dimensional embedding (L2-normalized)
            embedding2: Second 512-dimensional embedding (L2-normalized)
            
        Returns:
            float: Cosine similarity score in range [-1.0, 1.0]
                  - 1.0 = identical vectors (perfect match)
                  - 0.0 = orthogonal vectors (no similarity)
                  - -1.0 = opposite vectors (completely different)
                  
        Raises:
            ValueError: If embeddings have different lengths
        """
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1, dtype=np.float64)
        vec2 = np.array(embedding2, dtype=np.float64)
        
        # Validate dimensions match
        if vec1.shape != vec2.shape:
            raise ValueError(
                f"Embedding dimensions mismatch: {vec1.shape} vs {vec2.shape}"
            )
        
        # Compute dot product (simplified cosine similarity for normalized vectors)
        similarity = np.dot(vec1, vec2)
        
        # Clamp to valid range (handle floating point precision issues)
        similarity = np.clip(similarity, -1.0, 1.0)
        
        logger.debug(f"Cosine similarity computed: {similarity:.6f}")
        
        return float(similarity)
    
    @staticmethod
    def euclidean_distance(embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute Euclidean (L2) distance between two embeddings.
        
        Formula:
            distance = sqrt(sum((a - b)^2))
        
        For L2-normalized vectors, this is related to cosine similarity:
            distance^2 = 2 * (1 - cosine_similarity)
        
        Args:
            embedding1: First 512-dimensional embedding
            embedding2: Second 512-dimensional embedding
            
        Returns:
            float: Euclidean distance (0.0 = identical, higher = more different)
            
        Note:
            Not used for verification in Phase 1, but provided for future use.
        """
        vec1 = np.array(embedding1, dtype=np.float64)
        vec2 = np.array(embedding2, dtype=np.float64)
        
        if vec1.shape != vec2.shape:
            raise ValueError(
                f"Embedding dimensions mismatch: {vec1.shape} vs {vec2.shape}"
            )
        
        distance = np.linalg.norm(vec1 - vec2)
        
        logger.debug(f"Euclidean distance computed: {distance:.6f}")
        
        return float(distance)
    
    @staticmethod
    def verify_normalization(embedding: List[float], tolerance: float = 0.01) -> bool:
        """
        Verify if embedding is L2-normalized (||embedding|| ≈ 1.0).
        
        ArcFace embeddings should be L2-normalized. This method checks
        if the norm is close to 1.0 within a tolerance.
        
        Args:
            embedding: 512-dimensional embedding to check
            tolerance: Acceptable deviation from 1.0 (default: 0.01)
            
        Returns:
            bool: True if embedding is normalized, False otherwise
        """
        vec = np.array(embedding, dtype=np.float64)
        norm = np.linalg.norm(vec)
        
        is_normalized = abs(norm - 1.0) <= tolerance
        
        if not is_normalized:
            logger.warning(
                f"Embedding not properly normalized. "
                f"L2 norm: {norm:.6f} (expected: 1.0 ± {tolerance})"
            )
        
        return is_normalized
    
    @staticmethod
    def batch_similarity(
        reference_embedding: List[float],
        candidate_embeddings: List[List[float]]
    ) -> List[float]:
        """
        Compute similarity between one reference and multiple candidates.
        
        Efficient batch computation using matrix operations.
        
        Args:
            reference_embedding: Single 512-dimensional reference embedding
            candidate_embeddings: List of 512-dimensional candidate embeddings
            
        Returns:
            List[float]: Similarity scores for each candidate
            
        Note:
            Not used in Phase 1, but useful for future 1:N matching scenarios.
        """
        ref_vec = np.array(reference_embedding, dtype=np.float64)
        candidate_matrix = np.array(candidate_embeddings, dtype=np.float64)
        
        # Matrix-vector multiplication for batch similarity
        similarities = np.dot(candidate_matrix, ref_vec)
        
        # Clamp to valid range
        similarities = np.clip(similarities, -1.0, 1.0)
        
        logger.debug(f"Batch similarity computed for {len(candidate_embeddings)} candidates")
        
        return similarities.tolist()


# ═══════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════

def compute_similarity(
    embedding1: List[float],
    embedding2: List[float],
    method: str = "cosine"
) -> float:
    """
    Convenience function to compute similarity between embeddings.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        method: Similarity method - "cosine" or "euclidean"
        
    Returns:
        float: Similarity or distance score
        
    Raises:
        ValueError: If invalid method specified
    """
    calculator = SimilarityCalculator()
    
    if method == "cosine":
        return calculator.cosine_similarity(embedding1, embedding2)
    elif method == "euclidean":
        return calculator.euclidean_distance(embedding1, embedding2)
    else:
        raise ValueError(f"Invalid similarity method: {method}. Use 'cosine' or 'euclidean'.")
