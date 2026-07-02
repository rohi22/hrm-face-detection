"""
Official InsightFace Alignment Pipeline

Uses InsightFace's official face analysis and norm_crop for Buffalo_L preprocessing.
This replaces the custom MediaPipe-based alignment with the exact preprocessing
that Buffalo_L was trained on.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging
import insightface
from insightface.app import FaceAnalysis

import config

logger = logging.getLogger(__name__)


class InsightFaceAligner:
    """
    Official InsightFace alignment using built-in FaceAnalysis and norm_crop.
    
    This is the CORRECT preprocessing pipeline for Buffalo_L:
    - Uses SCRFD detector (part of Buffalo_L package)
    - Uses official 5-point landmarks
    - Uses official norm_crop alignment
    - Produces 112x112 aligned faces exactly as Buffalo_L expects
    """
    
    def __init__(self, det_size=(640, 640), det_thresh=0.5):
        """
        Initialize InsightFace face analyzer.
        
        Args:
            det_size: Detection size (width, height)
            det_thresh: Detection confidence threshold
        """
        logger.info("Initializing Official InsightFace Face Analyzer...")

        # Point InsightFace at the model pack BUNDLED in this repo so there is NO
        # ~330 MB runtime download on first run (Railway's FS is ephemeral, and a
        # cold download would blow the health-check timeout). We default
        # INSIGHTFACE_HOME to  <repo>/face-service/insightface_home  (which
        # contains  models/buffalo_l/*.onnx) whenever the env var is not already
        # set, guaranteeing zero download. An explicit INSIGHTFACE_HOME still wins.
        import os
        insightface_root = os.getenv("INSIGHTFACE_HOME")
        if not insightface_root:
            # <this file> = services/insightface_aligner.py  ->  parent.parent = face-service/
            _bundled_home = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "insightface_home",
            )
            os.environ["INSIGHTFACE_HOME"] = _bundled_home
            insightface_root = _bundled_home
            logger.info(f"  INSIGHTFACE_HOME defaulted to bundled pack: {insightface_root}")

        # Initialize FaceAnalysis with Buffalo_L detector. Pass root explicitly too
        # (FaceAnalysis resolves models at  root/models/buffalo_l/*.onnx).
        fa_kwargs = {"name": "buffalo_l", "providers": ["CPUExecutionProvider"],
                     "root": insightface_root}
        logger.info(f"  Using bundled InsightFace models at: {insightface_root}")
        self.app = FaceAnalysis(**fa_kwargs)
        
        self.app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
        
        logger.info("✓ Official InsightFace analyzer initialized")
        logger.info(f"  Detector: SCRFD (Buffalo_L)")
        logger.info(f"  Detection size: {det_size}")
        logger.info(f"  Detection threshold: {det_thresh}")
    
    def analyze(self, image_rgb: np.ndarray) -> Dict:
        """
        Single-pass detection + alignment that exposes everything the quality gate
        needs. Unlike detect_and_align(), this does NOT reject multi-face up front —
        it reports the count so the caller can decide.

        Returns a dict:
            {
              "face_count": int,
              "bbox": [x1,y1,x2,y2] | None,     # largest face
              "kps": [[x,y]*5] | None,          # 5 landmarks of largest face
              "det_score": float,
              "aligned_face_rgb": np.ndarray | None,   # 112x112 RGB norm_crop
              "pose": [pitch, yaw, roll] | None,        # 3D head angles in degrees
            }
        """
        from insightface.utils import face_align

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        faces = self.app.get(image_bgr)

        if not faces:
            return {"face_count": 0, "bbox": None, "kps": None,
                    "det_score": 0.0, "aligned_face_rgb": None, "pose": None}

        # Choose the largest face (most likely the subject) when several appear.
        def _area(f):
            b = f.bbox
            return (b[2] - b[0]) * (b[3] - b[1])
        face = max(faces, key=_area)
        largest_area = max(_area(f) for f in faces)

        # Count only SIGNIFICANT faces as "people present". SCRFD sometimes fires
        # on face-like patterns (wallpaper, printed logos) or on a tiny bystander
        # in the background; those boxes are low-confidence or much smaller than
        # the subject. Counting them produced false "more than one face" rejections
        # on clearly single-person photos. A second face only counts when it is
        # both confident AND comparable in size to the largest face.
        significant = [
            f for f in faces
            if float(f.det_score) >= config.MIN_DETECTION_SCORE
            and _area(f) >= config.MULTI_FACE_MIN_AREA_RATIO * largest_area
        ]
        face_count = max(len(significant), 1)  # the subject always counts

        aligned_bgr = face_align.norm_crop(image_bgr, landmark=face.kps, image_size=112)
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

        # 3D head pose from the bundled 1k3d68 model (pitch, yaw, roll in degrees).
        # Far more reliable than estimating from 5 points; used by the quality gate.
        pose = getattr(face, "pose", None)
        pose = pose.tolist() if pose is not None else None

        return {
            "face_count": face_count,
            "bbox": face.bbox.astype(int).tolist(),
            "kps": face.kps.tolist(),
            "det_score": float(face.det_score),
            "aligned_face_rgb": aligned_rgb,
            "pose": pose,
        }

    def detect_and_align(
        self,
        image: np.ndarray,
        save_debug: bool = False,
        debug_path: Optional[str] = None
    ) -> Tuple[bool, Optional[np.ndarray], Dict, str]:
        """
        Detect face and align using official InsightFace pipeline.
        
        Args:
            image: Input image (RGB format)
            save_debug: If True, save debug images
            debug_path: Path to save debug images
            
        Returns:
            Tuple containing:
                - success (bool): True if alignment succeeded
                - aligned_face (np.ndarray): Aligned 112x112 face or None
                - debug_info (dict): Debugging information
                - message (str): Status message
        """
        debug_info = {}
        
        logger.info("=" * 80)
        logger.info("OFFICIAL INSIGHTFACE PREPROCESSING PIPELINE")
        logger.info("=" * 80)
        
        # Convert RGB to BGR for InsightFace
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detect faces using official InsightFace detector
        faces = self.app.get(image_bgr)
        
        if len(faces) == 0:
            logger.warning("No face detected by InsightFace SCRFD")
            return False, None, debug_info, "No face detected"
        
        if len(faces) > 1:
            logger.warning(f"Multiple faces detected: {len(faces)}")
            return False, None, debug_info, "Multiple faces detected"
        
        # Get the detected face
        face = faces[0]
        
        # Extract information
        bbox = face.bbox.astype(int)
        kps = face.kps  # Official 5-point landmarks
        det_score = face.det_score
        
        logger.info(f"✓ Face detected by SCRFD")
        logger.info(f"  Confidence: {det_score:.3f}")
        logger.info(f"  Bounding box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        
        # Log official landmark coordinates
        logger.info("=" * 80)
        logger.info("OFFICIAL INSIGHTFACE LANDMARKS (5 points)")
        logger.info("=" * 80)
        logger.info(f"1. Left Eye:          [{kps[0][0]:.2f}, {kps[0][1]:.2f}]")
        logger.info(f"2. Right Eye:         [{kps[1][0]:.2f}, {kps[1][1]:.2f}]")
        logger.info(f"3. Nose Tip:          [{kps[2][0]:.2f}, {kps[2][1]:.2f}]")
        logger.info(f"4. Left Mouth Corner: [{kps[3][0]:.2f}, {kps[3][1]:.2f}]")
        logger.info(f"5. Right Mouth Corner:[{kps[4][0]:.2f}, {kps[4][1]:.2f}]")
        logger.info("=" * 80)
        
        # Use official norm_crop for alignment
        # This is the EXACT function used in InsightFace training
        from insightface.utils import face_align
        
        aligned_face = face_align.norm_crop(image_bgr, landmark=kps, image_size=112)
        
        # Convert back to RGB
        aligned_face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        
        logger.info(f"✓ Official norm_crop alignment completed")
        logger.info(f"  Aligned face shape: {aligned_face_rgb.shape}")
        
        # Store debug info
        debug_info["bbox"] = bbox.tolist()
        debug_info["landmarks"] = kps.tolist()
        debug_info["confidence"] = float(det_score)
        debug_info["aligned_shape"] = aligned_face_rgb.shape
        
        # Save debug images if requested
        if save_debug and debug_path:
            self._save_debug_images(
                image,
                bbox,
                kps,
                aligned_face_rgb,
                debug_path,
                det_score
            )
        
        return True, aligned_face_rgb, debug_info, "Face aligned with official InsightFace"
    
    def _save_debug_images(
        self,
        original: np.ndarray,
        bbox: np.ndarray,
        landmarks: np.ndarray,
        aligned: np.ndarray,
        output_path: str,
        confidence: float
    ):
        """
        Save debug visualization images.
        """
        import os
        import time
        os.makedirs(output_path, exist_ok=True)
        
        timestamp = int(time.time() * 1000)
        
        # 1. Save original
        cv2.imwrite(
            os.path.join(output_path, f"{timestamp}_1_original.jpg"),
            cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        )
        
        # 2. Original with annotations
        img_annotated = original.copy()
        
        # Draw bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            img_annotated,
            f"SCRFD: {confidence:.3f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
        # Draw landmarks
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        labels = ["L_eye", "R_eye", "Nose", "L_mouth", "R_mouth"]
        
        for i, (lm, color, label) in enumerate(zip(landmarks, colors, labels)):
            x, y = int(lm[0]), int(lm[1])
            cv2.circle(img_annotated, (x, y), 8, color, -1)
            cv2.circle(img_annotated, (x, y), 9, (255, 255, 255), 2)
            cv2.putText(
                img_annotated,
                label,
                (x + 15, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        cv2.imwrite(
            os.path.join(output_path, f"{timestamp}_2_scrfd_detection.jpg"),
            cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR)
        )
        
        # 3. Save aligned face (official norm_crop result)
        cv2.imwrite(
            os.path.join(output_path, f"{timestamp}_3_official_aligned_112x112.jpg"),
            cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
        )
        
        logger.info(f"✓ Debug images saved: {output_path} (timestamp: {timestamp})")
