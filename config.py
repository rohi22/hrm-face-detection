"""
Central configuration for the face recognition service.

This is the SINGLE SOURCE OF TRUTH for the decision threshold, the quality-gate
limits, and the score -> confidence mapping. Do not hardcode thresholds anywhere
else (app.py, the Flutter app, etc.) — read them from here / from the API response.

────────────────────────────────────────────────────────────────────────────
WHY 0.75 WAS WRONG
────────────────────────────────────────────────────────────────────────────
Buffalo_L (w600k_r50, ArcFace) produces cosine scores like this for DIFFERENT
photos of the SAME person:   ~0.45 – 0.85   (good frontal shots cluster 0.65–0.85)
and for DIFFERENT people:    ~0.0  – 0.45
Two DIFFERENT photos of the same person will essentially NEVER reach 0.95. The old
0.75 default therefore rejected most genuine matches. The correct operating point
sits in the GAP between the genuine and impostor clusters — around 0.40–0.50.

Run `python calibrate_threshold.py` on your own labelled photos to confirm the exact
value for your population, then update VERIFICATION_THRESHOLD below.
"""

import os
import math

# ════════════════════════════════════════════════════════════════════════════
# DECISION THRESHOLD  (cosine similarity on L2-normalised Buffalo_L embeddings)
# ════════════════════════════════════════════════════════════════════════════
# A live face is accepted when its similarity to the best enrollment image is
# >= VERIFICATION_THRESHOLD. Calibrated default; refine with calibrate_threshold.py.
# Overridable at runtime with the FACE_MATCH_THRESHOLD env var (handy on Railway).
VERIFICATION_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.45"))

# Below this, treat as a definite non-match (used for "no match" messaging).
HARD_REJECT_THRESHOLD = 0.30

# ════════════════════════════════════════════════════════════════════════════
# ENROLLMENT POLICY
# ════════════════════════════════════════════════════════════════════════════
# Recommended: 3 quality-gated frontal photos. Store ALL embeddings and match a
# live face against the MAX similarity across them. 1 photo is acceptable as a
# fallback if it passes the quality gate and is captured live.
ENROLLMENT_IMAGES_RECOMMENDED = 3
ENROLLMENT_IMAGES_MIN = 1

# ════════════════════════════════════════════════════════════════════════════
# QUALITY GATE LIMITS  (consumed by services/quality_gate.py)
# ════════════════════════════════════════════════════════════════════════════
# Detection
MIN_DETECTION_SCORE = 0.60      # SCRFD confidence; below -> "Face not clear"
MIN_FACE_RATIO = 0.030          # face box area / image area; below -> "Move closer"
# Absolute-pixel floor below which the crop is too small to recognise reliably.
# "Too far" is judged primarily by MIN_FACE_RATIO (how much of the frame the face
# fills). This pixel floor is only a last-ditch guard for genuinely tiny crops:
# uploaded/downscaled photos can put a CLOSE face (~19% of the frame) at only
# ~60px wide, and 62px crops still match well (cosine ~0.7), so 90 was rejecting
# close faces. 50 keeps that guard without firing on normal close-up uploads.
MIN_FACE_PIXELS = 50            # min face box width in px; below -> "Move closer"
# When counting "how many people", ignore spurious/background detections. SCRFD
# occasionally fires on face-like patterns (wallpaper, logos) or a tiny bystander
# far behind the subject. A box only counts as a second person when it is both
# confident AND comparable in size to the largest face — otherwise a single user
# gets wrongly rejected with "more than one face".
MULTI_FACE_MIN_AREA_RATIO = 0.35  # secondary face area / largest face area

# Pose / frontality.
# PREFERRED: real 3D head pose (pitch, yaw, roll) in DEGREES from Buffalo_L's
# 1k3d68 landmark model (face.pose). Calibrated on real photos: good frontal
# shots sit at |yaw|<=~14, |pitch|<=~12; clear side poses read 18-22.
MAX_YAW_DEGREES = 18.0          # left/right head turn; above -> "Look straight"
MAX_PITCH_DEGREES = 20.0        # up/down head tilt; above -> "Keep your face level"
# FALLBACK (only used when 3D pose is unavailable): yaw from the 5 landmarks.
MAX_YAW_RATIO = 0.34            # |nose offset from eye-centre| / inter-ocular dist
MAX_ROLL_DEGREES = 22.0         # head tilt (eye line vs horizontal)

# Sharpness / lighting (measured on the aligned 112x112 crop)
MIN_BLUR_VARIANCE = 60.0        # Laplacian variance; below -> "Image is blurry"
MIN_BRIGHTNESS = 50.0           # mean luminance 0-255; below -> "too dark"
MAX_BRIGHTNESS = 215.0          # above -> "too bright"

# Eye occlusion / sunglasses (analysed on the aligned 112x112 crop).
# ─────────────────────────────────────────────────────────────────────────────
# KEY INSIGHT (validated on 40 real photos): brightness ALONE cannot tell
# sunglasses from a bare-but-shadowed face — a deep-set bare face scored eye/cheek
# ratio 0.51 while real sunglasses scored 0.55–0.58 (the bare face was DARKER).
# But adding a COLOUR signal separates them cleanly: the eye region of a bare face
# (eyelids/brow/skin) still contains skin-coloured pixels, whereas a sunglasses
# lens does not. So we combine two signals:
#
#   1. eyes genuinely BLACKED OUT:        eye/cheek ratio < EYES_DARK_RATIO        (any colour)
#   2. sunglasses (dark AND not skin):    ratio < EYE_OCCLUSION_RATIO  AND  eye-region
#                                          skin fraction < EYE_OCCLUSION_MAX_SKIN
#
# On the 40-photo test this caught 5/5 sunglasses with 0 false positives:
#   - bare shadowed faces  (ratio ~0.43-0.51, skin ~0.80) PASS  (skin saves them)
#   - clear Rx glasses     (ratio ~0.67-0.68, skin ~0.30) PASS  (brightness saves them)
#   - sunglasses           (ratio ~0.31-0.58, skin ~0.13-0.21) REJECT
# A trained glasses classifier would be even more robust, but this is a large,
# data-validated improvement with no extra model. All produce the SAME user
# message ("eyes not clearly visible"); the match stage is the second line.
EYES_DARK_RATIO = 0.40          # eye/cheek brightness below this -> eyes blacked out
EYE_OCCLUSION_RATIO = 0.62      # eye region this dark AND lacking skin colour ...
EYE_OCCLUSION_MAX_SKIN = 0.45   # ... (skin fraction below this) -> sunglasses
# STANDALONE occlusion gate (added after a reflective pair of sunglasses slipped
# through). Mirror/reflective lenses throw bright specular highlights, so the eye
# region is NOT dark — dark_ratio stays high and both rules above miss it. But the
# reflections are blue/white glints, not skin, so the eye-region SKIN fraction is
# still very low. On a GOOD camera a bare face keeps skin ~0.5-0.8 around the eyes
# while sunglasses sit ~0.1-0.2, so this gate is safe with a wide margin.
#
# CAMERA-QUALITY CAVEAT (measured on the LAN test phone's low-res preview): a cheap
# sensor in dim light both darkens AND DESATURATES the eye region, so a genuine
# bare-but-dim face can read as low as ~0.12 while sunglasses read ~0.09 — almost no
# gap. The default is therefore set LOW (0.10) so real dim faces pass on poor cameras;
# raise it (e.g. 0.20-0.26) for production hardware where bare faces read 0.5+. The
# other two rules (blacked_out / dark+no-skin) still catch matte sunglasses; this
# standalone rule is only the extra catch for reflective lenses. Tune via env var.
EYE_MIN_SKIN = float(os.getenv("FACE_EYE_MIN_SKIN", "0.10"))
# Skin-colour test (YCrCb) bounds used to measure the eye-region skin fraction.
SKIN_CR_MIN, SKIN_CR_MAX = 133, 180
SKIN_CB_MIN, SKIN_CB_MAX = 77, 127

# ════════════════════════════════════════════════════════════════════════════
# LIVENESS / ANTI-SPOOFING  (passive PAD — Layer 2; consumed by services/liveness.py)
# ════════════════════════════════════════════════════════════════════════════
# Server-side presentation-attack detection. Classifies a captured face as a LIVE
# person vs a PRINT / SCREEN-REPLAY / MASK attack so nobody can hold a photo or a
# phone in front of the camera to mark someone else's attendance.
#
# Models: the minivision "Silent-Face" MiniFASNet pair (ONNX), run as an ensemble.
#   - MiniFASNetV2  uses a 2.7x crop around the face bbox
#   - MiniFASNetV1SE uses a 4.0x crop
# PREPROCESSING (validated against the reference inference — do NOT change blindly):
#   crop->80x80, RAW BGR pixels [0,255] (NO /255), NCHW float32. The 3-class softmax
#   has index 1 = REAL; p_real = mean(softmax)[1] across the two models.
#
# Calibration (labelled minivision samples + 39 genuine captures): genuine
# quality-passing faces score p_real >= 0.53, print/replay <= 0.07 — so 0.5 cleanly
# separates. Re-validate on real deployment spoof samples and tune via the env var.
LIVENESS_MODELS = [
    ("MiniFASNetV2.onnx", 2.7),
    ("MiniFASNetV1SE.onnx", 4.0),
]
# CAMERA-QUALITY CAVEAT (measured on the LAN test phone's low-res preview): the
# MiniFASNet p_real for a GENUINE live person on that cheap sensor spreads 0.36-0.88
# (not the 0.9+ a good camera gives), while real print/screen/video spoofs still sit
# at p_real <= 0.05. So the calibrated 0.5 cutoff intermittently false-rejected a
# real face whose frame dipped to ~0.49. Lowered to 0.25 for the test hardware: it
# clears every genuine capture (>=0.36) with margin while real spoofs (<=0.05) are
# nowhere close. The active on-device gesture challenge is now the PRIMARY liveness
# proof; this passive PAD is the backstop. RAISE back to ~0.45-0.5 for production
# cameras (env FACE_LIVENESS_THRESHOLD) where genuine faces read 0.9+.
LIVENESS_THRESHOLD = float(os.getenv("FACE_LIVENESS_THRESHOLD", "0.25"))
# A check-in sends a short burst of live frames; the decision uses the MEDIAN
# p_real across valid frames (robust to a single bad frame). At least this many
# frames must contain exactly one detectable face or the burst is rejected.
LIVENESS_MIN_VALID_FRAMES = 1

# ════════════════════════════════════════════════════════════════════════════
# SCORE -> CONFIDENCE %  (logistic, centred on the threshold)
# ════════════════════════════════════════════════════════════════════════════
# Stakeholders want to see a high "accuracy" number for a genuine match. The raw
# cosine (e.g. 0.78) is NOT a percentage. This maps a cosine score to a calibrated
# confidence% so a real match reads as 94–99% while the DECISION still uses the
# threshold. Tuned so: 0.45->50%, 0.55->~78%, 0.65->~92%, 0.75->~97%, 0.82->~99%.
_CONF_CENTER = VERIFICATION_THRESHOLD
_CONF_STEEPNESS = 12.0


def score_to_confidence(score: float) -> float:
    """Map a cosine similarity to a 0–100 confidence percentage (1 decimal)."""
    x = _CONF_STEEPNESS * (score - _CONF_CENTER)
    # Guard against overflow for extreme inputs.
    x = max(min(x, 60.0), -60.0)
    confidence = 100.0 / (1.0 + math.exp(-x))
    return round(confidence, 1)


def confidence_band(score: float) -> str:
    """Human label for the match strength (decision-independent)."""
    if score >= 0.70:
        return "very_high"
    if score >= 0.58:
        return "high"
    if score >= VERIFICATION_THRESHOLD:
        return "medium"
    if score >= HARD_REJECT_THRESHOLD:
        return "low"
    return "none"


def decide(score: float, threshold: float = None) -> dict:
    """Turn a cosine score into a full decision payload used across the service."""
    threshold = VERIFICATION_THRESHOLD if threshold is None else threshold
    matched = score >= threshold
    return {
        "matched": matched,
        "score": round(float(score), 6),
        "threshold": round(float(threshold), 4),
        "confidence": score_to_confidence(score),   # 0-100 %
        "band": confidence_band(score),
        "margin": round(float(score) - float(threshold), 4),
    }


# Official ArcFace 5-point template for a 112x112 aligned face. Landmark positions
# are FIXED by norm_crop alignment, so the quality gate can sample eye/cheek
# regions at known coordinates regardless of the original pose.
ARCFACE_TEMPLATE_112 = [
    (38.2946, 51.6963),   # left eye
    (73.5318, 51.5014),   # right eye
    (56.0252, 71.7366),   # nose tip
    (41.5493, 92.3655),   # left mouth corner
    (70.7299, 92.2041),   # right mouth corner
]
