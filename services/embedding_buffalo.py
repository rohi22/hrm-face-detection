"""
Buffalo_L Embedding Extraction Service

Generates 512-dimensional face embeddings using InsightFace Buffalo_L (W600K_R50).
This is a PARALLEL implementation alongside GhostFaceNet for comparison testing.
"""

import onnxruntime as ort
import numpy as np
import cv2
from typing import List
import logging
import os

logger = logging.getLogger(__name__)


class BuffaloLEmbeddingExtractor:
    """
    Extracts face embeddings using InsightFace Buffalo_L ONNX model.
    
    Model: buffalo_l_w600k_r50 (official InsightFace)
    Handles model loading, preprocessing with NCHW transpose, inference, and L2 normalization.
    """
    
    # Model expects 112x112 RGB images in NCHW format
    INPUT_SIZE = (112, 112)
    EMBEDDING_SIZE = 512
    
    def __init__(self, model_path: str = "models/buffalo_l_w600k_r50.onnx"):
        """
        Initialize Buffalo_L embedding extractor.
        
        Args:
            model_path: Path to Buffalo_L ONNX model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Buffalo_L model not found at: {model_path}")
        
        logger.info(f"Loading Buffalo_L model from: {model_path}")
        
        # Load ONNX model with ONNX Runtime
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Get input and output details
            self.input_info = self.session.get_inputs()[0]
            self.output_info = self.session.get_outputs()[0]
            
            # Log model details
            input_shape = self.input_info.shape
            output_shape = self.output_info.shape
            input_dtype = self.input_info.type
            
            logger.info(f"Buffalo_L model loaded successfully")
            logger.info(f"Input name: {self.input_info.name}")
            logger.info(f"Output name: {self.output_info.name}")
            logger.info(f"Input shape: {input_shape}, dtype: {input_dtype}")
            logger.info(f"Output shape: {output_shape}")
            logger.info(f"Tensor layout: NCHW (batch, channels, height, width)")
            
            # Buffalo_L preprocessing: image / 127.5 - 1.0 (range [-1, 1]), RGB, NCHW
            self.preprocessing_method = "normalize_minus1_1_nchw"
            logger.info("Preprocessing: (pixel / 127.5) - 1.0 → [-1, 1] + NHWC→NCHW transpose (Buffalo_L)")
            
        except Exception as e:
            logger.error(f"Failed to load Buffalo_L model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def preprocess_face(self, face_image: np.ndarray, save_debug: bool = False, debug_path: str = None) -> np.ndarray:
        """
        Preprocess face image for Buffalo_L model.
        
        Steps:
        1. Resize to 112x112
        2. Convert to RGB if needed
        3. Normalize: (pixel / 127.5) - 1.0 → [-1, 1]
        4. Transpose from NHWC to NCHW (CRITICAL for Buffalo_L)
        
        Args:
            face_image: Input face image (any size, any color format)
            save_debug: If True, save the final tensor as image
            debug_path: Path to save debug image
            
        Returns:
            Preprocessed image ready for model inference (NCHW format)
        """
        # Resize to 112x112
        resized = cv2.resize(face_image, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # Save the resized image before normalization (this is the actual input to the model)
        if save_debug and debug_path:
            import time
            timestamp = int(time.time() * 1000)
            cv2.imwrite(
                f"{debug_path}/{timestamp}_6_final_tensor_input.jpg",
                cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            )
            logger.info(f"✓ Saved final tensor input image: {debug_path}/{timestamp}_6_final_tensor_input.jpg")
        
        # Ensure RGB format
        if len(resized.shape) == 2:
            # Grayscale to RGB
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            # RGBA to RGB
            resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
        
        # Buffalo_L preprocessing: (pixel / 127.5) - 1.0
        # This scales [0, 255] to [-1, 1]
        preprocessed = (resized.astype(np.float32) / 127.5) - 1.0
        
        # Add batch dimension: (112, 112, 3) -> (1, 112, 112, 3)
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        # CRITICAL: Transpose from NHWC to NCHW for Buffalo_L
        # (1, 112, 112, 3) -> (1, 3, 112, 112)
        preprocessed = np.transpose(preprocessed, (0, 3, 1, 2))
        
        logger.debug(f"Preprocessed face: shape={preprocessed.shape}, dtype={preprocessed.dtype}, "
                    f"min={preprocessed.min():.3f}, max={preprocessed.max():.3f}")
        
        return preprocessed
    
    def extract_embedding(self, face_image: np.ndarray, save_debug: bool = False, debug_path: str = None) -> np.ndarray:
        """
        Extract 512-dimensional embedding from face image.
        
        Args:
            face_image: Cropped face image (any size, will be resized to 112x112)
            save_debug: If True, save debug tensor image
            debug_path: Path to save debug images
            
        Returns:
            512-dimensional L2-normalized embedding as numpy array
        """
        logger.info("─" * 80)
        logger.info("🔴 USING ALIGNED IMAGE FOR BUFFALO_L INFERENCE 🔴")
        logger.info(f"Input image shape: {face_image.shape}")
        logger.info(f"Input image dtype: {face_image.dtype}")
        logger.info(f"Input min/max: [{face_image.min()}, {face_image.max()}]")
        logger.info("─" * 80)
        
        # Preprocess image (includes NHWC→NCHW transpose)
        preprocessed = self.preprocess_face(face_image, save_debug=save_debug, debug_path=debug_path)
        
        logger.info(f"Preprocessed shape: {preprocessed.shape} (NCHW format)")
        logger.info(f"Preprocessed dtype: {preprocessed.dtype}")
        logger.info(f"Preprocessed min/max: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
        
        # Run inference
        import time
        start_time = time.time()
        
        outputs = self.session.run(
            [self.output_info.name],
            {self.input_info.name: preprocessed}
        )
        
        inference_time = (time.time() - start_time) * 1000  # ms
        logger.info(f"Buffalo_L inference time: {inference_time:.2f} ms")
        
        # Get output embedding
        embedding = outputs[0]
        
        # Remove batch dimension: (1, 512) -> (512,)
        embedding = embedding.squeeze()
        
        # DEBUG: Log RAW embedding before normalization
        logger.info("─" * 80)
        logger.info("RAW EMBEDDING (before L2 normalization)")
        logger.info(f"  Shape: {embedding.shape}")
        logger.info(f"  Min: {embedding.min():.6f}")
        logger.info(f"  Max: {embedding.max():.6f}")
        logger.info(f"  Mean: {embedding.mean():.6f}")
        logger.info(f"  Std: {embedding.std():.6f}")
        logger.info(f"  L2 Norm: {np.linalg.norm(embedding):.6f}")
        logger.info(f"  First 10 values: {embedding[:10]}")
        logger.info("─" * 80)
        
        # L2 normalization (Buffalo_L does NOT normalize by default)
        embedding = self._l2_normalize(embedding)
        
        logger.info("NORMALIZED EMBEDDING (after L2 normalization)")
        logger.info(f"  L2 Norm: {np.linalg.norm(embedding):.6f}")
        logger.info(f"  First 10 values: {embedding[:10]}")
        logger.info("─" * 80)
        
        return embedding
    
    def _l2_normalize(self, embedding: np.ndarray) -> np.ndarray:
        """
        L2 normalize embedding vector.
        
        Formula: normalized = embedding / ||embedding||
        
        Args:
            embedding: Raw embedding vector
            
        Returns:
            L2-normalized embedding (norm = 1.0)
        """
        norm = np.linalg.norm(embedding)
        
        if norm == 0:
            logger.warning("Zero norm detected, returning original embedding")
            return embedding
        
        normalized = embedding / norm
        
        # Verify normalization
        final_norm = np.linalg.norm(normalized)
        logger.debug(f"L2 normalization: original_norm={norm:.6f}, final_norm={final_norm:.6f}")
        
        return normalized
    
    def embedding_to_list(self, embedding: np.ndarray) -> List[float]:
        """
        Convert numpy embedding to Python list of floats.
        
        Args:
            embedding: Numpy array embedding
            
        Returns:
            List of 512 float values
        """
        return embedding.astype(float).tolist()
    
    def get_embedding_from_image(self, image: np.ndarray, save_debug: bool = False, debug_path: str = None) -> List[float]:
        """
        Complete pipeline: image -> embedding list.
        
        Args:
            image: Input face image (cropped and aligned)
            save_debug: If True, save debug images
            debug_path: Path to save debug images
            
        Returns:
            512-dimensional embedding as list of floats
        """
        embedding_array = self.extract_embedding(image, save_debug=save_debug, debug_path=debug_path)
        return self.embedding_to_list(embedding_array)
    
    def compute_image_quality(self, image_rgb: np.ndarray) -> dict:
        """
        Compute quality metrics for an image.
        
        Args:
            image_rgb: RGB image (H, W, 3)
        
        Returns:
            Dict with quality metrics:
                - blur_score: Laplacian variance (higher = sharper)
                - brightness: Mean luminance [0-255]
                - image_size: Total pixels
                - quality_score: Overall quality [0-1]
                - is_acceptable: Whether image meets minimum standards
                - warnings: List of quality issues
        """
        # Convert to grayscale for blur detection
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Blur detection using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(laplacian.var())
        
        # Brightness (mean luminance)
        brightness = float(gray.mean())
        
        # Image size
        image_size = image_rgb.shape[0] * image_rgb.shape[1]
        
        # Quality score (normalized [0-1])
        blur_norm = min(blur_score / 500.0, 1.0)  # 500+ is very sharp
        brightness_norm = 1.0 - abs(brightness - 128.0) / 128.0  # Best around 128
        
        quality_score = 0.6 * blur_norm + 0.4 * brightness_norm
        
        # Quality assessment
        is_acceptable = (
            blur_score >= 100.0 and  # Not too blurry
            40.0 <= brightness <= 220.0 and  # Not too dark/bright
            image_size >= 10000  # Not too small
        )
        
        warnings = []
        if blur_score < 100.0:
            warnings.append('Image too blurry')
        if brightness < 40.0:
            warnings.append('Image too dark')
        if brightness > 220.0:
            warnings.append('Image too bright')
        if image_size < 10000:
            warnings.append('Image too small')
        
        return {
            'blur_score': blur_score,
            'brightness': brightness,
            'image_size': image_size,
            'quality_score': quality_score,
            'is_acceptable': is_acceptable,
            'warnings': warnings
        }
    
    @staticmethod
    def fuse_embeddings(embeddings: List[np.ndarray], 
                       quality_scores: List[float] = None,
                       method: str = 'average') -> np.ndarray:
        """
        Fuse multiple embeddings into single template.
        
        This is the RECOMMENDED approach for production systems.
        Based on InsightFace/ArcFace best practices.
        
        Args:
            embeddings: List of normalized embeddings (each 512-D)
            quality_scores: Optional quality scores [0-1] for weighted fusion
            method: 'average', 'weighted', or 'median'
                - 'average': Simple average (recommended)
                - 'weighted': Quality-weighted average
                - 'median': Median (robust to outliers)
        
        Returns:
            Fused template (normalized to unit length)
        """
        if not embeddings:
            raise ValueError("No embeddings provided for fusion")
        
        embeddings_array = np.array(embeddings)  # Shape: (N, 512)
        
        if method == 'average':
            # Simple average (standard practice)
            template = embeddings_array.mean(axis=0)
        
        elif method == 'weighted' and quality_scores is not None and len(quality_scores) == len(embeddings):
            # Quality-weighted average
            weights = np.array(quality_scores)
            if weights.sum() > 0:
                weights = weights / weights.sum()  # Normalize weights
                template = (embeddings_array.T @ weights)  # Shape: (512,)
            else:
                logger.warning("All quality scores are zero, falling back to average")
                template = embeddings_array.mean(axis=0)
        
        elif method == 'median':
            # Median (robust to outliers)
            template = np.median(embeddings_array, axis=0)
        
        else:
            # Fallback to average
            logger.warning(f"Unknown method '{method}' or invalid quality_scores, falling back to average")
            template = embeddings_array.mean(axis=0)
        
        # L2 normalize
        norm = np.linalg.norm(template)
        if norm > 0:
            template = template / norm
        else:
            logger.warning("Zero norm template, returning unnormalized")
        
        return template
