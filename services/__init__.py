"""
Services Package

Contains core business logic for face verification and embedding extraction.
"""

from services.verify import FaceVerificationService
from services.similarity import SimilarityCalculator, compute_similarity
from services.embedding_buffalo import BuffaloLEmbeddingExtractor
from services.insightface_aligner import InsightFaceAligner

__all__ = [
    'FaceVerificationService',
    'SimilarityCalculator',
    'compute_similarity',
    'BuffaloLEmbeddingExtractor',
    'InsightFaceAligner',
]
