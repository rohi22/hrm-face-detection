"""
Passive presentation-attack detection (anti-spoofing) — Layer 2 of the design in
LARAVEL_INTEGRATION_CONTRACT.md §9.

Wraps the minivision "Silent-Face" MiniFASNet models (ONNX) to classify a captured
face as a LIVE person vs a PRINT / SCREEN-REPLAY / MASK presentation attack. This
runs SERVER-SIDE so a tampered app cannot bypass it and the model can be swapped
without an app release.

────────────────────────────────────────────────────────────────────────────
VALIDATED PREPROCESSING (do NOT "normalize" this — that breaks it)
────────────────────────────────────────────────────────────────────────────
The reference ONNX inference (yakhyo/face-anti-spoofing, a faithful port of the
minivision weights) feeds RAW pixels, not [0,1]:
  1. crop the face bbox expanded by a model-specific scale (2.7 for V2, 4.0 for
     V1SE) around its centre, clamped to the image, resized to 80x80
  2. RAW BGR pixels [0,255], channel-first NCHW, float32 — NO /255, NO mean/std
  3. 3-class softmax; index 1 = REAL, indices 0 and 2 = attack
  4. ensemble = mean of the two models' softmax; p_real = ensemble[1]

Feeding /255 makes both models collapse to a constant class (verified empirically)
— hence the explicit warning. Calibrated on labelled minivision samples + 39
genuine captures: genuine quality-passing faces score p_real >= 0.53; print/replay
score p_real <= 0.07. Default decision threshold 0.5 (config.LIVENESS_THRESHOLD).
"""

import os
import logging

import cv2
import numpy as np
import onnxruntime as ort

import config

logger = logging.getLogger(__name__)


class LivenessDetector:
    """Ensemble MiniFASNet passive anti-spoofing over a burst of live frames."""

    def __init__(self, models_dir: str = "models"):
        self.sessions = []  # list of (session, input_name, scale)
        for fname, scale in config.LIVENESS_MODELS:
            path = os.path.join(models_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Liveness model missing: {path}. Download the MiniFASNet ONNX "
                    f"models into {models_dir}/ (see LARAVEL_INTEGRATION_CONTRACT.md §9)."
                )
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.sessions.append((sess, sess.get_inputs()[0].name, float(scale)))
            logger.info(f"  ✓ Liveness model loaded: {fname} (crop scale {scale})")
        logger.info(f"✓ Anti-spoofing ready — {len(self.sessions)}-model MiniFASNet "
                    f"ensemble, threshold p_real>={config.LIVENESS_THRESHOLD}")

    # ── preprocessing ────────────────────────────────────────────────────────
    @staticmethod
    def _crop_face(image_bgr: np.ndarray, bbox_xywh, scale: float, out: int = 80) -> np.ndarray:
        """Expand the bbox by `scale` around its centre, clamp, crop, resize."""
        src_h, src_w = image_bgr.shape[:2]
        x, y, box_w, box_h = bbox_xywh
        # Never enlarge past the image; this mirrors the reference implementation.
        scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
        new_w, new_h = box_w * scale, box_h * scale
        cx, cy = x + box_w / 2.0, y + box_h / 2.0
        x1 = max(0, int(cx - new_w / 2))
        y1 = max(0, int(cy - new_h / 2))
        x2 = min(src_w - 1, int(cx + new_w / 2))
        y2 = min(src_h - 1, int(cy + new_h / 2))
        return cv2.resize(image_bgr[y1:y2 + 1, x1:x2 + 1], (out, out))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    # ── scoring ──────────────────────────────────────────────────────────────
    def score_frame(self, image_bgr: np.ndarray, bbox_xyxy) -> float:
        """Return p_real (0..1) for a single frame given its face bbox [x1,y1,x2,y2]."""
        x1, y1, x2, y2 = bbox_xyxy
        bbox_xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        ensemble = np.zeros(3, dtype=np.float64)
        for sess, iname, scale in self.sessions:
            crop = self._crop_face(image_bgr, bbox_xywh, scale)
            # RAW BGR [0,255], NCHW, float32 — intentionally NOT normalized.
            inp = crop.astype(np.float32).transpose(2, 0, 1)[None]
            logits = sess.run(None, {iname: inp})[0][0]
            ensemble += self._softmax(logits)
        ensemble /= len(self.sessions)
        return float(ensemble[1])  # index 1 = REAL

    def check(self, frames_with_bbox) -> dict:
        """
        Decide liveness over a burst.

        Args:
            frames_with_bbox: list of (image_bgr, bbox_xyxy) for every VALID frame
                (exactly one detected face). Empty list -> not live (no usable frame).

        Returns a dict matching the /check-liveness contract.
        """
        per_frame = [round(self.score_frame(img, bb), 4) for img, bb in frames_with_bbox]

        if len(per_frame) < config.LIVENESS_MIN_VALID_FRAMES:
            return {
                "live": False,
                "spoof_score": 1.0,
                "label": "spoof",
                "threshold": config.LIVENESS_THRESHOLD,
                "frames_used": len(per_frame),
                "per_frame_p_real": per_frame,
                "message": "Could not read a clear live face. Hold the camera steady on your face.",
            }

        # Median p_real — robust to one bad frame in the burst.
        agg = float(np.median(per_frame))
        live = agg >= config.LIVENESS_THRESHOLD
        return {
            "live": live,
            "spoof_score": round(1.0 - agg, 4),
            "label": "real" if live else "spoof",
            "threshold": config.LIVENESS_THRESHOLD,
            "frames_used": len(per_frame),
            "per_frame_p_real": per_frame,
            "message": "Liveness OK" if live
                       else "Spoof detected. Use your real face — a photo or screen is not allowed.",
        }
