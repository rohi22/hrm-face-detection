"""
Face Verification Service Module

Handles face verification logic using ArcFace embeddings and cosine similarity.
"""

from typing import List, Dict, Any
import logging

from services.similarity import SimilarityCalculator

logger = logging.getLogger(__name__)


class FaceVerificationService:
    """
    Service for verifying face embeddings using cosine similarity.
    
    This service compares stored (enrolled) embeddings against live embeddings
    to determine if they represent the same person.
    
    Attributes:
        similarity_calculator: Instance of SimilarityCalculator
        default_threshold: Default similarity threshold for verification
    """
    
    # Default threshold for face verification (industry standard)
    DEFAULT_THRESHOLD = 0.75
    
    # Recommended threshold ranges
    THRESHOLD_LOW_SECURITY = 0.70      # More permissive, fewer false rejections
    THRESHOLD_STANDARD = 0.75          # Balanced security (recommended)
    THRESHOLD_HIGH_SECURITY = 0.80     # Stricter, more false rejections
    THRESHOLD_VERY_HIGH = 0.85         # Very strict, use with caution
    
    def __init__(self):
        """Initialize the face verification service."""
        self.similarity_calculator = SimilarityCalculator()
        logger.info("FaceVerificationService initialized")
    
    def verify(
        self,
        stored_embedding: List[float],
        live_embedding: List[float],
        threshold: float = DEFAULT_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Verify if stored and live embeddings match.
        
        Algorithm:
        1. Validate input embeddings (512 dimensions)
        2. Compute cosine similarity using dot product
        3. Compare score against threshold
        4. Return verification result
        
        Args:
            stored_embedding: Enrolled embedding (512-d, L2-normalized)
            live_embedding: Live captured embedding (512-d, L2-normalized)
            threshold: Similarity threshold for positive match (default: 0.75)
            
        Returns:
            Dict containing:
                - verified (bool): True if match, False otherwise
                - score (float): Cosine similarity score (0.0 to 1.0)
                - threshold (float): Threshold used for decision
                - details (dict): Additional debug information
                
        Raises:
            ValueError: If embeddings are invalid or have wrong dimensions
        """
        logger.info("=" * 60)
        logger.info("Face Verification Started")
        logger.info("=" * 60)
        
        # Validate inputs
        self._validate_embedding(stored_embedding, "stored_embedding")
        self._validate_embedding(live_embedding, "live_embedding")
        self._validate_threshold(threshold)
        
        logger.info(f"Stored embedding: {len(stored_embedding)} dimensions")
        logger.info(f"Live embedding: {len(live_embedding)} dimensions")
        logger.info(f"Threshold: {threshold}")
        
        # Check if embeddings are properly normalized (optional verification)
        stored_normalized = self.similarity_calculator.verify_normalization(
            stored_embedding, tolerance=0.05
        )
        live_normalized = self.similarity_calculator.verify_normalization(
            live_embedding, tolerance=0.05
        )
        
        if not stored_normalized:
            logger.warning("Stored embedding may not be properly L2-normalized")
        if not live_normalized:
            logger.warning("Live embedding may not be properly L2-normalized")
        
        # Compute cosine similarity
        logger.info("Computing cosine similarity...")
        similarity_score = self.similarity_calculator.cosine_similarity(
            stored_embedding,
            live_embedding
        )
        
        logger.info(f"Similarity Score: {similarity_score:.6f}")
        
        # Make verification decision
        verified = similarity_score >= threshold
        
        logger.info(f"Verification Decision: {'MATCH' if verified else 'NO MATCH'}")
        
        # Log confidence level
        confidence = self._calculate_confidence(similarity_score, threshold)
        logger.info(f"Confidence Level: {confidence}")
        
        logger.info("=" * 60)
        
        # Build result
        result = {
            "verified": verified,
            "score": round(similarity_score, 6),
            "threshold": threshold,
            "details": {
                "confidence": confidence,
                "stored_normalized": stored_normalized,
                "live_normalized": live_normalized,
                "margin": round(similarity_score - threshold, 6),
                "recommendation": self._get_recommendation(similarity_score, threshold)
            }
        }
        
        return result
    
    def _validate_embedding(self, embedding: List[float], name: str) -> None:
        """
        Validate embedding format and dimensions.
        
        Args:
            embedding: Embedding to validate
            name: Name of the embedding (for error messages)
            
        Raises:
            ValueError: If embedding is invalid
        """
        if not isinstance(embedding, list):
            raise ValueError(f"{name} must be a list")
        
        if len(embedding) != 512:
            raise ValueError(
                f"{name} must have exactly 512 dimensions, got {len(embedding)}"
            )
        
        # Check if all values are numeric
        for i, value in enumerate(embedding):
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{name}[{i}] is not a number: {value} (type: {type(value).__name__})"
                )
            
            # Sanity check: values should typically be in [-2, 2] for normalized embeddings
            if not (-5.0 <= value <= 5.0):
                logger.warning(
                    f"{name}[{i}] = {value} is outside typical range [-2, 2]. "
                    "This may indicate an issue with the embedding."
                )
    
    def _validate_threshold(self, threshold: float) -> None:
        """
        Validate threshold value.
        
        Args:
            threshold: Threshold to validate
            
        Raises:
            ValueError: If threshold is invalid
        """
        if not isinstance(threshold, (int, float)):
            raise ValueError(f"Threshold must be a number, got {type(threshold).__name__}")
        
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        # Warn if threshold is outside recommended range
        if threshold < 0.65:
            logger.warning(
                f"Threshold {threshold} is very low. "
                "This may result in false positives (security risk)."
            )
        elif threshold > 0.90:
            logger.warning(
                f"Threshold {threshold} is very high. "
                "This may result in false negatives (usability issues)."
            )
    
    def _calculate_confidence(self, score: float, threshold: float) -> str:
        """
        Calculate confidence level for verification result.
        
        Args:
            score: Similarity score
            threshold: Threshold used
            
        Returns:
            str: Confidence level description
        """
        margin = score - threshold
        
        if score >= threshold:
            # Positive match
            if margin >= 0.15:
                return "Very High (strong match)"
            elif margin >= 0.10:
                return "High (clear match)"
            elif margin >= 0.05:
                return "Medium (acceptable match)"
            else:
                return "Low (borderline match)"
        else:
            # Negative match
            if margin <= -0.15:
                return "Very High (clear non-match)"
            elif margin <= -0.10:
                return "High (strong non-match)"
            elif margin <= -0.05:
                return "Medium (probable non-match)"
            else:
                return "Low (borderline non-match)"
    
    def _get_recommendation(self, score: float, threshold: float) -> str:
        """
        Get recommendation based on verification result.
        
        Args:
            score: Similarity score
            threshold: Threshold used
            
        Returns:
            str: Recommendation message
        """
        margin = score - threshold
        
        if score >= threshold:
            if margin >= 0.10:
                return "Strong match - proceed with confidence"
            elif margin >= 0.05:
                return "Good match - acceptable for most use cases"
            else:
                return "Borderline match - consider requiring re-verification or additional authentication"
        else:
            if margin >= -0.05:
                return "Near miss - consider adjusting threshold or allowing retry"
            else:
                return "Clear non-match - deny access"
    
    def get_threshold_recommendation(
        self,
        security_level: str = "standard"
    ) -> float:
        """
        Get recommended threshold for specified security level.
        
        Args:
            security_level: One of "low", "standard", "high", "very_high"
            
        Returns:
            float: Recommended threshold value
            
        Raises:
            ValueError: If invalid security level specified
        """
        thresholds = {
            "low": self.THRESHOLD_LOW_SECURITY,
            "standard": self.THRESHOLD_STANDARD,
            "high": self.THRESHOLD_HIGH_SECURITY,
            "very_high": self.THRESHOLD_VERY_HIGH
        }
        
        if security_level not in thresholds:
            raise ValueError(
                f"Invalid security level: {security_level}. "
                f"Must be one of: {', '.join(thresholds.keys())}"
            )
        
        return thresholds[security_level]
