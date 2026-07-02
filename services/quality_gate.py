"""
Production Face Quality Gate
────────────────────────────────────────────────────────────────────────────
The single biggest accuracy lever in a face-attendance system is NOT the model —
it is refusing to run recognition on bad input. Commercial face SDKs feel
accurate mostly because they reject sunglasses, side faces, blur, and dark frames
BEFORE matching. This module reproduces that behaviour with the data we already
have from SCRFD detection (bbox, 5 landmarks, detection score) plus pixel analysis
on the aligned 112x112 crop.

Every failure returns a short, user-facing message (the kind the stakeholder asked
for: "Please remove sunglasses", "Look straight at the camera", ...).

Usage:
    from services.quality_gate import evaluate_quality
    result = evaluate_quality(
        face_count=2, bbox=[x1,y1,x2,y2], kps=kps5x2, det_score=0.85,
        image_shape=(h, w), aligned_face_rgb=aligned112,
    )
    if not result["passed"]:
        return 400 with result["message"]
"""

import math
import cv2
import numpy as np

import config


# Severity order — the first failing check (most important) drives the headline
# message shown to the user. Lower number = reported first.
# Geometry/pose (reliable) is reported BEFORE the eye-darkness heuristics
# (sunglasses/eyes_dark), so a turned/tilted/down face never mis-reports as
# "remove your sunglasses" just because the tilt shadowed the eyes.
_SEVERITY = {
    "no_face": 0,
    "multiple_faces": 1,
    "low_detection_confidence": 2,
    "face_too_small": 3,
    "not_frontal_yaw": 4,
    "not_level_pitch": 5,
    "head_tilted": 6,
    "eyes_too_dark": 7,
    "image_blurry": 8,
    "too_dark": 9,
    "too_bright": 10,
}

_MESSAGES = {
    "no_face": "No face detected. Position your face inside the frame.",
    "multiple_faces": "More than one face detected. Only one person at a time.",
    "low_detection_confidence": "Face not clear. Hold steady and try again.",
    "face_too_small": "Face is too far. Move closer to the camera.",
    "eyes_too_dark": "Your eyes are not clearly visible. Remove sunglasses/cap and face the light.",
    "not_frontal_yaw": "Look straight at the camera.",
    "not_level_pitch": "Keep your face level — do not look up or down.",
    "head_tilted": "Keep your head straight (do not tilt).",
    "image_blurry": "Image is blurry. Hold the phone steady.",
    "too_dark": "Too dark. Move to better lighting.",
    "too_bright": "Too bright. Reduce direct light/glare.",
}


def _patch_stats(gray: np.ndarray, cx: float, cy: float, half: int):
    """Mean brightness and std (texture) of a square patch around (cx, cy)."""
    h, w = gray.shape[:2]
    x0 = max(0, int(cx - half));  x1 = min(w, int(cx + half))
    y0 = max(0, int(cy - half));  y1 = min(h, int(cy + half))
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0
    patch = gray[y0:y1, x0:x1]
    return float(patch.mean()), float(patch.std())


def _skin_fraction(rgb: np.ndarray, cx: float, cy: float, half: int) -> float:
    """Fraction of pixels in a patch that look like skin (YCrCb rule).

    This is the signal that separates sunglasses from a bare-but-shadowed face:
    a bare eye region (eyelids/brow) still has skin-coloured pixels; a sunglasses
    lens does not. See config.py for the bounds and the validation notes.
    """
    h, w = rgb.shape[:2]
    x0 = max(0, int(cx - half));  x1 = min(w, int(cx + half))
    y0 = max(0, int(cy - half));  y1 = min(h, int(cy + half))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = rgb[y0:y1, x0:x1]
    ycrcb = cv2.cvtColor(patch, cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[:, :, 1].astype(np.int32)
    cb = ycrcb[:, :, 2].astype(np.int32)
    mask = ((cr >= config.SKIN_CR_MIN) & (cr <= config.SKIN_CR_MAX) &
            (cb >= config.SKIN_CB_MIN) & (cb <= config.SKIN_CB_MAX))
    return float(mask.mean())


def _estimate_pose(kps):
    """
    Rough yaw/roll from the 5 SCRFD landmarks.
      kps = [L_eye, R_eye, nose, L_mouth, R_mouth], each (x, y).
    Returns (yaw_ratio, roll_degrees). yaw_ratio is the nose's horizontal offset
    from the eye midpoint, normalised by inter-ocular distance (0 = perfectly
    centred / frontal). roll is the tilt of the eye line vs horizontal.
    """
    l_eye, r_eye, nose = kps[0], kps[1], kps[2]
    eye_cx = (l_eye[0] + r_eye[0]) / 2.0
    eye_cy = (l_eye[1] + r_eye[1]) / 2.0
    interocular = math.hypot(r_eye[0] - l_eye[0], r_eye[1] - l_eye[1])
    if interocular < 1e-3:
        return 1.0, 90.0, eye_cy
    yaw_ratio = (nose[0] - eye_cx) / interocular
    # vertical component of nose offset hints at pitch, but yaw is the main risk
    roll_degrees = math.degrees(math.atan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))
    return yaw_ratio, roll_degrees, eye_cy


def evaluate_quality(
    face_count: int,
    bbox,
    kps,
    det_score: float,
    image_shape,
    aligned_face_rgb: np.ndarray,
    pose=None,
) -> dict:
    """
    Run all quality checks. Returns:
      {
        "passed": bool,
        "message": str | None,          # headline message (most severe failure)
        "failures": [ {code, message, value, limit}, ... ],
        "metrics": { ... raw numbers for debugging ... },
      }
    """
    failures = []
    metrics = {}

    # ── 1. Face presence / uniqueness ────────────────────────────────────────
    if face_count == 0 or bbox is None or kps is None:
        return _result([{"code": "no_face", "message": _MESSAGES["no_face"]}], {})
    if face_count > 1:
        failures.append({"code": "multiple_faces", "message": _MESSAGES["multiple_faces"],
                         "value": face_count, "limit": 1})

    # ── 2. Detection confidence ──────────────────────────────────────────────
    det_score = float(det_score)
    metrics["det_score"] = round(det_score, 4)
    if det_score < config.MIN_DETECTION_SCORE:
        failures.append({"code": "low_detection_confidence",
                         "message": _MESSAGES["low_detection_confidence"],
                         "value": round(det_score, 3), "limit": config.MIN_DETECTION_SCORE})

    # ── 3. Face size (too far) ───────────────────────────────────────────────
    img_h, img_w = image_shape[0], image_shape[1]
    bw = float(bbox[2] - bbox[0])
    bh = float(bbox[3] - bbox[1])
    face_ratio = (bw * bh) / float(max(img_w * img_h, 1))
    metrics["face_box_width_px"] = round(bw, 1)
    metrics["face_area_ratio"] = round(face_ratio, 4)
    if face_ratio < config.MIN_FACE_RATIO or bw < config.MIN_FACE_PIXELS:
        failures.append({"code": "face_too_small", "message": _MESSAGES["face_too_small"],
                         "value": round(face_ratio, 4), "limit": config.MIN_FACE_RATIO})

    # ── 4. Pose: yaw (turn) + pitch (up/down) + roll (tilt) ──────────────────
    # Prefer the real 3D head pose from Buffalo_L's 1k3d68 model (degrees); it is
    # far more reliable than estimating yaw from 5 points and additionally gives
    # pitch (looking up/down), which the 5-point fallback cannot measure.
    if pose is not None and len(pose) == 3:
        pitch_deg, yaw_deg, roll_deg = float(pose[0]), float(pose[1]), float(pose[2])
        metrics["pitch_degrees"] = round(pitch_deg, 1)
        metrics["yaw_degrees"] = round(yaw_deg, 1)
        metrics["roll_degrees"] = round(roll_deg, 1)
        if abs(yaw_deg) > config.MAX_YAW_DEGREES:
            failures.append({"code": "not_frontal_yaw", "message": _MESSAGES["not_frontal_yaw"],
                             "value": round(abs(yaw_deg), 1), "limit": config.MAX_YAW_DEGREES})
        if abs(pitch_deg) > config.MAX_PITCH_DEGREES:
            failures.append({"code": "not_level_pitch", "message": _MESSAGES["not_level_pitch"],
                             "value": round(abs(pitch_deg), 1), "limit": config.MAX_PITCH_DEGREES})
        if abs(roll_deg) > config.MAX_ROLL_DEGREES:
            failures.append({"code": "head_tilted", "message": _MESSAGES["head_tilted"],
                             "value": round(abs(roll_deg), 1), "limit": config.MAX_ROLL_DEGREES})
    else:
        # Fallback: rough yaw/roll from the 5 landmarks (no pitch available).
        yaw_ratio, roll_deg, _eye_cy = _estimate_pose(kps)
        metrics["yaw_ratio"] = round(yaw_ratio, 3)
        metrics["roll_degrees"] = round(roll_deg, 2)
        if abs(yaw_ratio) > config.MAX_YAW_RATIO:
            failures.append({"code": "not_frontal_yaw", "message": _MESSAGES["not_frontal_yaw"],
                             "value": round(abs(yaw_ratio), 3), "limit": config.MAX_YAW_RATIO})
        if abs(roll_deg) > config.MAX_ROLL_DEGREES:
            failures.append({"code": "head_tilted", "message": _MESSAGES["head_tilted"],
                             "value": round(abs(roll_deg), 1), "limit": config.MAX_ROLL_DEGREES})

    # ── 5–7. Pixel checks on the ALIGNED crop (geometry is normalised here) ──
    if aligned_face_rgb is not None and aligned_face_rgb.size > 0:
        gray = cv2.cvtColor(aligned_face_rgb, cv2.COLOR_RGB2GRAY)

        # Blur (Laplacian variance)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        metrics["blur_variance"] = round(blur, 1)
        if blur < config.MIN_BLUR_VARIANCE:
            failures.append({"code": "image_blurry", "message": _MESSAGES["image_blurry"],
                             "value": round(blur, 1), "limit": config.MIN_BLUR_VARIANCE})

        # Brightness
        brightness = float(gray.mean())
        metrics["brightness"] = round(brightness, 1)
        if brightness < config.MIN_BRIGHTNESS:
            failures.append({"code": "too_dark", "message": _MESSAGES["too_dark"],
                             "value": round(brightness, 1), "limit": config.MIN_BRIGHTNESS})
        elif brightness > config.MAX_BRIGHTNESS:
            failures.append({"code": "too_bright", "message": _MESSAGES["too_bright"],
                             "value": round(brightness, 1), "limit": config.MAX_BRIGHTNESS})

        # Sunglasses / eye occlusion — sample eye region vs cheek region at the
        # fixed ArcFace template positions (works because the crop is aligned).
        tmpl = config.ARCFACE_TEMPLATE_112
        le_b, le_s = _patch_stats(gray, tmpl[0][0], tmpl[0][1], half=12)
        re_b, re_s = _patch_stats(gray, tmpl[1][0], tmpl[1][1], half=12)
        # cheeks: just below and outside the mouth corners / under the eyes
        lc_b, _ = _patch_stats(gray, tmpl[3][0] - 4, (tmpl[0][1] + tmpl[3][1]) / 2, half=10)
        rc_b, _ = _patch_stats(gray, tmpl[4][0] + 4, (tmpl[1][1] + tmpl[4][1]) / 2, half=10)
        eye_brightness = (le_b + re_b) / 2.0
        eye_std = (le_s + re_s) / 2.0
        cheek_brightness = max((lc_b + rc_b) / 2.0, 1.0)
        dark_ratio = eye_brightness / cheek_brightness
        # Skin-colour fraction in the eye region (the sunglasses discriminator).
        le_skin = _skin_fraction(aligned_face_rgb, tmpl[0][0], tmpl[0][1], half=12)
        re_skin = _skin_fraction(aligned_face_rgb, tmpl[1][0], tmpl[1][1], half=12)
        eye_skin = (le_skin + re_skin) / 2.0
        metrics["eye_brightness"] = round(eye_brightness, 1)
        metrics["cheek_brightness"] = round(cheek_brightness, 1)
        metrics["eye_cheek_ratio"] = round(dark_ratio, 3)
        metrics["eye_texture_std"] = round(eye_std, 1)
        metrics["eye_skin_fraction"] = round(eye_skin, 3)

        # Occlusion gate. Brightness alone can't tell sunglasses from a shadowed
        # bare face, so we combine darkness with a skin-colour test (see config.py):
        #   (a) eyes genuinely blacked out (very dark vs cheek), OR
        #   (b) eyes dark AND the eye region has almost no skin colour -> sunglasses, OR
        #   (c) eye region has almost NO skin colour on its own -> covered lens.
        # A bare-but-dark face keeps skin colour around the eyes, so it passes (a)+(b);
        # clear/Rx glasses keep the eyes bright, so they pass (a) and the ratio test.
        # (c) is the catch for REFLECTIVE/mirror sunglasses: their bright glints keep
        # the eyes from reading "dark", so (a) and (b) miss them, but the glints are
        # not skin-coloured, so the skin fraction is still near zero.
        blacked_out = dark_ratio < config.EYES_DARK_RATIO
        sunglasses = (dark_ratio < config.EYE_OCCLUSION_RATIO
                      and eye_skin < config.EYE_OCCLUSION_MAX_SKIN)
        no_skin = eye_skin < config.EYE_MIN_SKIN
        if blacked_out or sunglasses or no_skin:
            failures.append({"code": "eyes_too_dark", "message": _MESSAGES["eyes_too_dark"],
                             "value": round(eye_skin, 3), "limit": config.EYE_MIN_SKIN})

    return _result(failures, metrics)


def _result(failures, metrics):
    if not failures:
        return {"passed": True, "message": None, "failures": [], "metrics": metrics}
    failures.sort(key=lambda f: _SEVERITY.get(f["code"], 99))
    return {
        "passed": False,
        "message": failures[0]["message"],
        "failures": failures,
        "metrics": metrics,
    }
