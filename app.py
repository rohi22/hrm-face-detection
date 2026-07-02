"""
FastAPI Face Verification Service
Phase 1: Local verification service for ArcFace embeddings

Provides REST API endpoint for verifying face embeddings using cosine similarity.
No database, no enrollment logic, just verification computation.
"""

from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import logging
import numpy as np
import cv2
from PIL import Image, ImageOps
import io

import config
from services.verify import FaceVerificationService
from services.embedding_buffalo import BuffaloLEmbeddingExtractor
from services.insightface_aligner import InsightFaceAligner
from services.liveness import LivenessDetector
from services.quality_gate import evaluate_quality
from services.similarity import SimilarityCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Verification Service",
    description="Buffalo_L (InsightFace/ONNX) face embedding verification service using cosine similarity",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize verification service
verification_service = FaceVerificationService()

# Initialize the PRODUCTION embedding extractor.
# ─────────────────────────────────────────────────────────────────────────────
# SINGLE MODEL = Buffalo_L. Every production endpoint Laravel/mobile relies on
# (/enroll, /verify, /extract-embedding-buffalo*, /verify-images) uses Buffalo_L
# ONLY. Embeddings from two different models live in different vector spaces, so
# mixing models would produce a meaningless cosine — that is the "embedding
# conflict" we must avoid. Detection + alignment come from InsightFace's own
# SCRFD/norm_crop pipeline (see insightface_aligner below), so there is no
# separate face detector to load.
embedding_extractor_buffalo = BuffaloLEmbeddingExtractor(model_path="models/buffalo_l_w600k_r50.onnx")

# Initialize OFFICIAL InsightFace aligner for Buffalo_L (replaces custom alignment)
insightface_aligner = InsightFaceAligner(det_size=(640, 640), det_thresh=0.5)

# Passive anti-spoofing (Layer 2). Tiny MiniFASNet ONNX ensemble (~3.5 MB total).
# If the model files are absent the service still boots; /check-liveness then 503s.
try:
    liveness_detector = LivenessDetector(models_dir="models")
except Exception as _live_e:
    liveness_detector = None
    logger.warning(f"Liveness/anti-spoofing DISABLED — could not load models: {_live_e}")


# ═══════════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════════

class EnrollmentRequest(BaseModel):
    """Request model for face enrollment endpoint"""
    
    emp_id: int = Field(
        ...,
        description="Employee ID from the HRM system",
        gt=0
    )
    embedding: List[float] = Field(
        ...,
        description="512-dimensional ArcFace embedding from enrollment (L2 normalized)",
        min_length=512,
        max_length=512
    )
    
    @field_validator('embedding')
    @classmethod
    def validate_embedding_values(cls, v: List[float]) -> List[float]:
        """Validate that all embedding values are valid floats"""
        for i, value in enumerate(v):
            if not isinstance(value, (int, float)):
                raise ValueError(f"Element at index {i} is not a number: {value}")
            if not (-2.0 <= value <= 2.0):
                logger.warning(f"Embedding value at index {i} outside expected range: {value}")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "emp_id": 451,
                    "embedding": [0.123, -0.456, 0.789] + [0.0] * 509
                }
            ]
        }
    }


class EnrollmentResponse(BaseModel):
    """Response model for face enrollment endpoint"""
    
    success: bool = Field(
        ...,
        description="TRUE if enrollment payload received and validated successfully"
    )
    message: str = Field(
        ...,
        description="Human-readable result message"
    )
    emp_id: int = Field(
        ...,
        description="Employee ID from request"
    )
    embedding_size: int = Field(
        ...,
        description="Number of dimensions in the embedding (should be 512)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "message": "Enrollment payload received",
                    "emp_id": 451,
                    "embedding_size": 512
                }
            ]
        }
    }


class VerificationRequest(BaseModel):
    """Request model for face verification endpoint"""
    
    emp_id: int = Field(
        ...,
        description="Employee ID from HRM system",
        gt=0
    )
    stored_embedding: List[float] = Field(
        ...,
        description="512-dimensional ArcFace embedding from enrollment (L2 normalized)",
        min_length=512,
        max_length=512
    )
    live_embedding: List[float] = Field(
        ...,
        description="512-dimensional ArcFace embedding from live capture (L2 normalized)",
        min_length=512,
        max_length=512
    )
    threshold: float = Field(
        default=config.VERIFICATION_THRESHOLD,
        description="Cosine similarity threshold (calibrated default ~0.45). "
                    "ArcFace genuine pairs sit ~0.45-0.85; do NOT set this to 0.75+.",
        ge=0.20,
        le=0.95
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "emp_id": 451,
                    "stored_embedding": [0.123, -0.456, 0.789] + [0.0] * 509,
                    "live_embedding": [0.121, -0.454, 0.791] + [0.0] * 509,
                    "threshold": 0.45
                }
            ]
        }
    }


class VerificationResponse(BaseModel):
    """Response model for face verification endpoint"""
    
    verified: bool = Field(
        ...,
        description="TRUE if faces match (score >= threshold), FALSE otherwise"
    )
    emp_id: int = Field(
        ...,
        description="Employee ID from request"
    )
    score: float = Field(
        ...,
        description="Cosine similarity score (0.0 to 1.0, higher = more similar)"
    )
    threshold: float = Field(
        ...,
        description="Threshold used for verification decision"
    )
    confidence: str = Field(
        ...,
        description="Confidence level: very_high, high, medium, low, failed"
    )
    message: str = Field(
        ...,
        description="Human-readable verification result message"
    )
    details: dict = Field(
        ...,
        description="Additional details including margin from threshold"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "verified": True,
                    "emp_id": 451,
                    "score": 0.69,
                    "threshold": 0.45,
                    "confidence": "very_high",
                    "message": "Face verified successfully",
                    "details": {
                        "margin": 0.16
                    }
                }
            ]
        }
    }


class FuseTemplateRequest(BaseModel):
    """Request body for /fuse-enrollment-template"""
    embeddings: List[List[float]] = Field(..., description="List of enrollment embeddings (512-D each)")
    quality_scores: Optional[List[float]] = Field(None, description="Optional quality scores [0-1] for weighted fusion")
    method: str = Field("average", description="Fusion method: 'average', 'weighted', or 'median'")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str


class ExtractEmbeddingResponse(BaseModel):
    """Response model for extract-embedding endpoint"""
    
    success: bool = Field(
        ...,
        description="TRUE if embedding extracted successfully"
    )
    embedding: List[float] = Field(
        ...,
        description="512-dimensional L2-normalized ArcFace embedding"
    )
    embedding_size: int = Field(
        ...,
        description="Number of dimensions in the embedding (should be 512)"
    )
    message: str = Field(
        default="Embedding extracted successfully",
        description="Human-readable result message"
    )
    quality: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Quality-gate result: passed, message, failures[], metrics{}"
    )
    detection: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detection info: face_count, bbox, det_score"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "embedding": [0.123, -0.456, 0.789] + [0.0] * 509,
                    "embedding_size": 512,
                    "message": "Embedding extracted successfully"
                }
            ]
        }
    }


# ═══════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Root endpoint - service health check"""
    return HealthResponse(
        status="online",
        service="Face Verification Service",
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return HealthResponse(
        status="healthy",
        service="Face Verification Service",
        version="1.0.0"
    )


@app.post(
    "/enroll",
    response_model=EnrollmentResponse,
    status_code=status.HTTP_200_OK,
    tags=["Enrollment"],
    summary="Enroll face embedding",
    description="""
    Receives and validates a face enrollment payload for an employee.
    
    **Phase 2A - Validation Only:**
    - Receives employee ID and face embedding
    - Validates embedding dimensions (must be 512)
    - Validates embedding values (must be floats)
    - Logs enrollment attempt
    - Returns success confirmation
    
    **NOT Implemented Yet (Future Phases):**
    - Database storage
    - Duplicate enrollment checking
    - Embedding quality validation
    - Face image storage
    
    **Expected Input:**
    - emp_id: Positive integer (employee ID from HRM system)
    - embedding: Exactly 512 float values (L2-normalized from ArcFace)
    
    **Use Case:**
    This endpoint will be called by Laravel when an employee completes
    face enrollment in the Flutter app. The embedding will later be stored
    in the database for verification during attendance check-in.
    """
)
async def enroll_face(request: EnrollmentRequest):
    """
    Enroll face embedding for an employee.
    
    Args:
        request: EnrollmentRequest containing emp_id and embedding
        
    Returns:
        EnrollmentResponse with success confirmation
        
    Raises:
        HTTPException: If validation fails
    """
    try:
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("Face Enrollment Request Received")
        logger.info(f"Employee ID: {request.emp_id}")
        logger.info(f"Embedding Dimensions: {len(request.embedding)}")
        
        # Verify embedding normalization (optional check)
        embedding_array = np.array(request.embedding)
        norm = np.linalg.norm(embedding_array)
        logger.info(f"Embedding L2 Norm: {norm:.6f}")
        
        if abs(norm - 1.0) > 0.05:
            logger.warning(
                f"Embedding may not be properly L2-normalized. "
                f"Norm: {norm:.6f} (expected: ~1.0)"
            )
        
        logger.info("✓ Validation passed")
        logger.info("✓ Enrollment payload received successfully")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        return EnrollmentResponse(
            success=True,
            message="Enrollment payload received",
            emp_id=request.emp_id,
            embedding_size=len(request.embedding)
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Enrollment error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during enrollment: {str(e)}"
        )


@app.post(
    "/verify",
    response_model=VerificationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Verification"],
    summary="Verify face embeddings",
    description="""
    Verifies if two ArcFace embeddings match using cosine similarity.
    
    **1:1 embedding match (Laravel calls this for attendance verify).**
    - Accepts employee ID for audit logging
    - No database access (Laravel handles that)
    - Single source of truth for the threshold + confidence is config.py

    **Algorithm:**
    1. Receives two 512-d L2-normalized Buffalo_L embeddings (stored + live)
    2. Computes cosine similarity: score = dot(stored, live)
    3. Compares against the calibrated threshold (default config.VERIFICATION_THRESHOLD = 0.45)
    4. Returns the decision + calibrated confidence% (config.score_to_confidence)

    **IMPORTANT — do NOT pass threshold=0.75.** Buffalo_L genuine pairs (two photos
    of the SAME person) sit ~0.45-0.85 and almost never reach 0.75, so 0.75 would
    reject real employees. Leave `threshold` unset to use the calibrated 0.45, or
    override only via the FACE_MATCH_THRESHOLD env var after running calibrate_threshold.py.

    **Both embeddings MUST come from this service's Buffalo_L extractor**
    (/extract-embedding-buffalo). Mixing models (e.g. on-device GhostFaceNet) makes
    the embeddings incomparable and the score meaningless.

    **Confidence band** (config.confidence_band, decision-independent):
    very_high >=0.70 · high >=0.58 · medium >=0.45 · low >=0.30 · none <0.30
    """
)
async def verify_face(request: VerificationRequest):
    """
    Verify if two face embeddings match.
    
    Args:
        request: VerificationRequest containing emp_id, stored and live embeddings
        
    Returns:
        VerificationResponse with verification result and similarity score
        
    Raises:
        HTTPException: If validation fails or computation error occurs
    """
    try:
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("Face Verification Request Received")
        logger.info(f"[VERIFY] emp_id={request.emp_id} threshold={request.threshold}")
        
        # Perform verification
        result = verification_service.verify(
            stored_embedding=request.stored_embedding,
            live_embedding=request.live_embedding,
            threshold=request.threshold
        )
        
        # Decision + calibrated confidence% (single source of truth: config.py)
        score = result['score']
        decision = config.decide(score, request.threshold)
        verified = decision["matched"]
        confidence = decision["band"]

        logger.info(f"[VERIFY] emp_id={request.emp_id} score={score:.4f} "
                    f"threshold={decision['threshold']} confidence%={decision['confidence']} "
                    f"band={confidence} verified={verified}")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        message = "Face verified successfully" if verified else "Face does not match the enrolled employee"

        return VerificationResponse(
            verified=verified,
            emp_id=request.emp_id,
            score=round(score, 6),
            threshold=decision["threshold"],
            confidence=confidence,
            message=message,
            details={
                "margin": decision["margin"],
                "confidence_percent": decision["confidence"],
                "band": decision["band"],
            }
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Verification error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during verification: {str(e)}"
        )


@app.post(
    "/extract-embedding-buffalo",
    response_model=ExtractEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    tags=["Embedding"],
    summary="Extract face embedding from image using Buffalo_L model",
    description="""
    Extracts a 512-dimensional embedding from an uploaded face image using InsightFace Buffalo_L model.
    
    **⚠️ PARALLEL MODEL FOR COMPARISON TESTING**
    This endpoint uses Buffalo_L (W600K_R50) model instead of GhostFaceNet.
    Use this endpoint from the debug screen to compare model performance.
    
    **Workflow:**
    1. Upload image (JPEG, PNG supported)
    2. Detect face using MediaPipe
    3. Validate exactly one face present
    4. Crop and align face
    5. Resize to 112x112
    6. Apply NCHW transpose (required for Buffalo_L)
    7. Run Buffalo_L ONNX inference
    8. L2 normalize embedding
    9. Return 512-dimensional embedding
    
    **Requirements:**
    - Image must contain exactly ONE face
    - Supported formats: JPEG, PNG, BMP
    - Face must be clearly visible
    - Good lighting recommended
    
    **Model Details:**
    - Model: InsightFace Buffalo_L (W600K_R50)
    - Input: 112x112 RGB in NCHW format
    - Output: 512-dimensional L2-normalized embedding
    - Preprocessing: (pixel / 127.5) - 1.0, then NHWC→NCHW transpose
    
    **Use Case:**
    This endpoint is for comparing Buffalo_L against GhostFaceNet.
    Use the debug screen to test both models with the same images.
    
    **Error Cases:**
    - No face detected → HTTP 400
    - Multiple faces detected → HTTP 400
    - Invalid image format → HTTP 400
    - Processing error → HTTP 500
    """
)
async def extract_embedding_buffalo(
    image: UploadFile = File(
        ...,
        description="Face image file (JPEG, PNG, BMP)"
    ),
    enforce_quality: bool = False,
    save_debug: bool = True,
):
    """
    Extract a 512-d Buffalo_L embedding AND run the production quality gate.

    Query params:
        enforce_quality: when TRUE, a face that fails the quality gate is rejected
            with HTTP 400 and the user-facing message (production attendance flow).
            When FALSE (default), the embedding is still returned together with the
            quality verdict so the debug screen can show scores for any image.
        save_debug: write annotated debug images to debug_output/ (default True).

    Returns ExtractEmbeddingResponse with `quality` and `detection` blocks.
    """
    try:
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("Extract Embedding Request (Buffalo_L + Quality Gate)")
        logger.info(f"Filename: {image.filename}  enforce_quality={enforce_quality}")

        # Read + decode image
        contents = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Invalid image format: {str(e)}")

        image_array = np.array(pil_image)
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

        # ── Detect + align (single pass, exposes detection data) ──────────────
        info = insightface_aligner.analyze(image_array)

        if info["face_count"] == 0 or info["aligned_face_rgb"] is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="No face detected. Position your face inside the frame.")

        aligned_face = info["aligned_face_rgb"]

        # ── Quality gate ──────────────────────────────────────────────────────
        quality = evaluate_quality(
            face_count=info["face_count"],
            bbox=info["bbox"],
            kps=info["kps"],
            det_score=info["det_score"],
            image_shape=image_array.shape,
            aligned_face_rgb=aligned_face,
            pose=info.get("pose"),
        )
        logger.info(f"[QUALITY] passed={quality['passed']} "
                    f"msg={quality['message']} metrics={quality['metrics']}")

        detection = {
            "face_count": info["face_count"],
            "bbox": info["bbox"],
            "det_score": round(info["det_score"], 4),
        }

        # Production: hard-reject bad input before it ever reaches matching.
        if enforce_quality and not quality["passed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": quality["message"],
                        "failures": quality["failures"],
                        "metrics": quality["metrics"]},
            )

        # Optional annotated debug images (original, SCRFD landmarks, aligned crop)
        if save_debug:
            try:
                insightface_aligner._save_debug_images(
                    image_array,
                    np.array(info["bbox"]),
                    np.array(info["kps"]),
                    aligned_face,
                    "debug_output/insightface_official",
                    info["det_score"],
                )
            except Exception as dbg_e:
                logger.warning(f"Debug image save skipped: {dbg_e}")

        # ── Embedding extraction ──────────────────────────────────────────────
        embedding = embedding_extractor_buffalo.get_embedding_from_image(aligned_face)

        if len(embedding) != 512:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Model produced invalid embedding size: {len(embedding)}")

        norm = float(np.linalg.norm(np.array(embedding)))
        if abs(norm - 1.0) > 0.05:
            logger.warning(f"Embedding norm deviates from 1.0: {norm:.6f}")

        msg = "Embedding extracted (Buffalo_L)"
        if not quality["passed"]:
            msg = f"Embedding extracted, but quality warning: {quality['message']}"

        logger.info("✓ Buffalo_L embedding extracted. " + msg)
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return ExtractEmbeddingResponse(
            success=True,
            embedding=embedding,
            embedding_size=len(embedding),
            message=msg,
            quality=quality,
            detection=detection,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding extraction error (Buffalo_L): {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during embedding extraction (Buffalo_L): {str(e)}"
        )


# ═══════════════════════════════════════════════════════════════
# Production Recognition Endpoints (Template Fusion & Quality)
# ═══════════════════════════════════════════════════════════════

@app.post(
    "/analyze-face",
    status_code=status.HTTP_200_OK,
    tags=["Production"],
    summary="Quality gate only (no decision) — for capture-time guidance",
    description="""
    Runs face detection + the production quality gate on a single image and returns
    a pass/fail verdict with a user-facing message (sunglasses, not frontal, blurry,
    too dark, too far, multiple faces, ...). No embedding, no matching.

    Use this at capture time (enrollment or attendance) to tell the user how to fix
    their photo BEFORE running recognition. This is what makes the system feel like
    a commercial SDK.
    """
)
async def analyze_face(image: UploadFile = File(..., description="Face image to check")):
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        image_array = np.array(pil_image)
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

        info = insightface_aligner.analyze(image_array)
        if info["face_count"] == 0 or info["aligned_face_rgb"] is None:
            return {
                "success": True,
                "passed": False,
                "message": "No face detected. Position your face inside the frame.",
                "failures": [{"code": "no_face"}],
                "detection": {"face_count": 0},
                "metrics": {},
            }

        quality = evaluate_quality(
            face_count=info["face_count"],
            bbox=info["bbox"],
            kps=info["kps"],
            det_score=info["det_score"],
            image_shape=image_array.shape,
            aligned_face_rgb=info["aligned_face_rgb"],
            pose=info.get("pose"),
        )
        return {
            "success": True,
            "passed": quality["passed"],
            "message": quality["message"] or "Face looks good.",
            "failures": quality["failures"],
            "detection": {
                "face_count": info["face_count"],
                "bbox": info["bbox"],
                "det_score": round(info["det_score"], 4),
            },
            "metrics": quality["metrics"],
        }
    except Exception as e:
        logger.error(f"analyze-face error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post(
    "/check-liveness",
    status_code=status.HTTP_200_OK,
    tags=["Production"],
    summary="Passive anti-spoofing — is this a LIVE face or a photo/screen replay?",
    description="""
    Layer 2 of the anti-spoofing design (see LARAVEL_INTEGRATION_CONTRACT.md §9).
    Send a short BURST of live frames (3–5 recommended) captured during check-in.
    The server runs a MiniFASNet PAD ensemble on each frame and returns a single
    liveness verdict (median across frames, robust to one bad frame).

    Blocks a fraudster holding a printed photo, a phone/laptop screen showing a
    photo/video, or a paper mask in front of the camera.

    Recommended check-in order (Laravel orchestrates):
      /check-liveness (block if not live) -> /verify-images -> mark attendance only
      if BOTH live AND verified AND quality-passed.

    Each frame must contain exactly ONE detectable face; frames with no face or
    multiple faces are skipped (and a multi-face frame fails the burst).
    """
)
async def check_liveness(
    frames: List[UploadFile] = File(..., description="Live frames burst (3–5 recommended)")
):
    if liveness_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Anti-spoofing model not loaded on the server.",
        )
    try:
        logger.info(f"[LIVENESS] burst of {len(frames)} frame(s)")
        frames_with_bbox = []
        multi_face = False
        for idx, upload in enumerate(frames):
            contents = await upload.read()
            try:
                pil_image = ImageOps.exif_transpose(Image.open(io.BytesIO(contents)))
            except Exception:
                logger.warning(f"[LIVENESS] frame {idx} unreadable — skipped")
                continue
            image_array = np.array(pil_image)  # RGB
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

            info = insightface_aligner.analyze(image_array)
            if info["face_count"] == 0 or info["bbox"] is None:
                logger.info(f"[LIVENESS] frame {idx}: no face — skipped")
                continue
            if info["face_count"] > 1:
                multi_face = True
                logger.info(f"[LIVENESS] frame {idx}: multiple faces")
                continue
            # Liveness models expect BGR; analyze() worked in RGB space.
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            frames_with_bbox.append((image_bgr, info["bbox"]))

        if not frames_with_bbox and multi_face:
            return {
                "live": False, "spoof_score": 1.0, "label": "spoof",
                "threshold": config.LIVENESS_THRESHOLD, "frames_used": 0,
                "per_frame_p_real": [],
                "message": "More than one person in frame. Only the employee should be visible.",
            }

        result = liveness_detector.check(frames_with_bbox)
        logger.info(f"[LIVENESS] live={result['live']} spoof_score={result['spoof_score']} "
                    f"frames_used={result['frames_used']} per_frame={result['per_frame_p_real']}")
        return {"success": True, **result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"check-liveness error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post(
    "/verify-images",
    status_code=status.HTTP_200_OK,
    tags=["Production"],
    summary="Full 1:1 verification — enrollment image(s) vs one live image",
    description="""
    The complete production verification flow in ONE call:
      1. Quality-gate the live (verification) image — reject with a clear message
         if it fails (sunglasses / not frontal / blurry / too dark / too far).
      2. Extract its Buffalo_L embedding.
      3. Match against EVERY enrollment image's embedding for this employee.
      4. Decide with the calibrated threshold; return MAX similarity, the best
         matching enrollment index, a confidence%, and the decision.

    This is 1:1 (tied to one employee's enrollments), which is the accurate,
    recommended mode. Send 1–N enrollment images (3 recommended) and 1 live image.
    """
)
async def verify_images(
    enrollment_images: List[UploadFile] = File(..., description="1-N enrollment photos (3 recommended)"),
    verification_image: UploadFile = File(..., description="Live/verification photo"),
    emp_id: Optional[str] = None,
    enforce_quality: bool = True,
):
    async def _embed(upload: UploadFile, run_gate: bool, skip_on_gate_fail: bool = False):
        contents = await upload.read()
        arr = np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(contents))))
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        info = insightface_aligner.analyze(arr)
        if info["face_count"] == 0 or info["aligned_face_rgb"] is None:
            return None, {"passed": False, "message": "No face detected.", "failures": [], "metrics": {}}
        gate = evaluate_quality(
            face_count=info["face_count"], bbox=info["bbox"], kps=info["kps"],
            det_score=info["det_score"], image_shape=arr.shape,
            aligned_face_rgb=info["aligned_face_rgb"], pose=info.get("pose"),
        ) if run_gate else {"passed": True, "message": None, "failures": [], "metrics": {}}
        # "Reject on the spot, before matching": if this face is going to be
        # rejected on quality anyway, do NOT spend a forward pass extracting the
        # embedding. Layer 3 (quality) short-circuits Layer 4 (match).
        if run_gate and skip_on_gate_fail and not gate["passed"]:
            return None, gate
        emb = embedding_extractor_buffalo.get_embedding_from_image(info["aligned_face_rgb"])
        return emb, gate

    try:
        # 1-2. Live image: gate first; embed only if it passes (when enforcing).
        live_emb, live_gate = await _embed(
            verification_image, run_gate=True, skip_on_gate_fail=enforce_quality)
        _m = live_gate.get("metrics", {})
        logger.info(
            "[VERIFY-IMAGES] quality passed=%s msg=%s | eye_skin=%s eye_cheek_ratio=%s "
            "blur=%s brightness=%s yaw=%s pitch=%s",
            live_gate.get("passed"), live_gate.get("message"),
            _m.get("eye_skin_fraction"), _m.get("eye_cheek_ratio"),
            _m.get("blur_variance"), _m.get("brightness"),
            _m.get("yaw_degrees"), _m.get("pitch_degrees"),
        )
        if live_emb is None or (enforce_quality and not live_gate["passed"]):
            return {
                "success": True, "verified": False, "emp_id": emp_id,
                "message": live_gate["message"] or "Live image rejected by quality gate.",
                "rejected_by_quality": True, "quality": live_gate,
            }

        # 3. Enrollment images: embed (no hard gate — already enrolled, but report)
        enroll_embs = []
        for up in enrollment_images:
            emb, _ = await _embed(up, run_gate=False)
            if emb is not None:
                enroll_embs.append(emb)
        if not enroll_embs:
            raise HTTPException(status_code=400, detail="No face found in any enrollment image.")

        # 4. Match: MAX similarity across enrollments
        sims = [SimilarityCalculator.cosine_similarity(e, live_emb) for e in enroll_embs]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        decision = config.decide(best_score)

        logger.info(f"[VERIFY-IMAGES] emp_id={emp_id} sims={[round(s,4) for s in sims]} "
                    f"best={best_score:.4f} matched={decision['matched']}")

        return {
            "success": True,
            "verified": decision["matched"],
            "emp_id": emp_id,
            "score": decision["score"],
            "threshold": decision["threshold"],
            "confidence_percent": decision["confidence"],
            "band": decision["band"],
            "best_enrollment_index": best_idx,
            "per_enrollment_scores": [round(s, 4) for s in sims],
            "rejected_by_quality": False,
            "quality": live_gate,
            "message": "Face verified successfully" if decision["matched"]
                       else "Face does not match the enrolled employee.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"verify-images error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post(
    "/check-quality",
    status_code=status.HTTP_200_OK,
    tags=["Production"],
    summary="Fast image-quality pre-check (no embedding, no match)",
    description="""
    Runs ONLY face detection + the quality gate (pose, sunglasses/eyes, blur,
    lighting) on a single frame. No embedding is extracted and nothing is matched,
    so it is cheap enough to call live — the client uses it to reject sunglasses /
    dark / off-pose faces ON THE SPOT, before asking the user for the liveness
    gestures. Returns {passed, message, failures, metrics}.
    """,
)
async def check_quality(image: UploadFile = File(..., description="One live frame")):
    try:
        contents = await image.read()
        arr = np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(contents))))
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        info = insightface_aligner.analyze(arr)
        if info["face_count"] == 0 or info["aligned_face_rgb"] is None:
            return {"passed": False,
                    "message": "No face detected. Position your face in the frame.",
                    "failures": [], "metrics": {}}
        gate = evaluate_quality(
            face_count=info["face_count"], bbox=info["bbox"], kps=info["kps"],
            det_score=info["det_score"], image_shape=arr.shape,
            aligned_face_rgb=info["aligned_face_rgb"], pose=info.get("pose"),
        )
        _m = gate.get("metrics", {})
        logger.info(
            "[CHECK-QUALITY] passed=%s msg=%s | eye_skin=%s eye_cheek=%s blur=%s "
            "brightness=%s yaw=%s pitch=%s",
            gate.get("passed"), gate.get("message"),
            _m.get("eye_skin_fraction"), _m.get("eye_cheek_ratio"),
            _m.get("blur_variance"), _m.get("brightness"),
            _m.get("yaw_degrees"), _m.get("pitch_degrees"),
        )
        return {
            "passed": gate["passed"], "message": gate["message"],
            "failures": gate["failures"], "metrics": gate["metrics"],
        }
    except Exception as e:
        logger.error(f"check-quality error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post(
    "/extract-embedding-buffalo-with-quality",
    status_code=status.HTTP_200_OK,
    tags=["Production"],
    summary="Extract Buffalo_L embedding with quality assessment",
    description="""
    Extract face embedding using Buffalo_L with quality assessment.
    
    **Pipeline:**
    1. SCRFD face detection
    2. Official InsightFace alignment (norm_crop)
    3. Buffalo_L embedding extraction
    4. Quality assessment (blur, brightness)
    
    **Quality Metrics:**
    - blur_score: Laplacian variance (>100 = sharp)
    - brightness: Mean luminance [0-255] (best: 100-180)
    - quality_score: Overall quality [0-1]
    - is_acceptable: Boolean quality pass/fail
    
    Returns embedding with quality metrics.
    """
)
async def extract_embedding_buffalo_with_quality(
    image: UploadFile = File(..., description="Face image for embedding extraction")
):
    """
    Extract Buffalo_L embedding with quality assessment.
    """
    try:
        logger.info("="*80)
        logger.info("🔍 Buffalo_L Embedding Extraction with Quality Assessment")
        logger.info("="*80)
        
        # Read and decode image (respect EXIF orientation from phone cameras)
        contents = await image.read()
        pil_image = ImageOps.exif_transpose(Image.open(io.BytesIO(contents)))
        image_array = np.array(pil_image)

        # Convert to RGB if needed
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        logger.info(f"Image loaded: shape={image_array.shape}, dtype={image_array.dtype}")

        # Detect + align (single pass) and run the production quality gate
        info = insightface_aligner.analyze(image_array)
        if info["face_count"] == 0 or info["aligned_face_rgb"] is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in image"
            )

        aligned_face_rgb = info["aligned_face_rgb"]
        quality = evaluate_quality(
            face_count=info["face_count"],
            bbox=info["bbox"],
            kps=info["kps"],
            det_score=info["det_score"],
            image_shape=image_array.shape,
            aligned_face_rgb=aligned_face_rgb,
            pose=info.get("pose"),
        )
        logger.info(f"Quality gate: passed={quality['passed']} message={quality['message']}")

        # Extract embedding
        embedding = embedding_extractor_buffalo.get_embedding_from_image(aligned_face_rgb)

        logger.info(f"✓ Embedding extracted with quality assessment")
        logger.info("="*80)

        return {
            "success": True,
            "embedding": embedding,
            "embedding_size": len(embedding),
            "quality": quality,
            "detection": {
                "face_count": info["face_count"],
                "bbox": info["bbox"],
                "det_score": round(info["det_score"], 4),
            },
            "message": "Embedding extracted with quality assessment"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding extraction with quality error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post(
    "/fuse-enrollment-template",
    status_code=status.HTTP_200_OK,
    tags=["Production"],
    summary="Fuse multiple enrollment embeddings into single template",
    description="""
    Create fused enrollment template from multiple embeddings.
    
    **Purpose:**
    This implements the RECOMMENDED production approach for enrollment:
    - Collect multiple enrollment images (3-10)
    - Extract embedding from each
    - Fuse into single representative template
    - Use template for all verifications
    
    **Benefits:**
    - Reduces variance in recognition
    - More robust to outliers
    - Better generalization
    - Standard practice in commercial systems
    
    **Methods:**
    - average: Simple average (recommended, default)
    - weighted: Quality-weighted average
    - median: Median (robust to outliers)
    
    **Input:**
    - embeddings: List of 512-D embeddings
    - quality_scores: Optional quality scores for weighted fusion
    - method: Fusion method
    
    **Output:**
    - template: Fused 512-D template (L2 normalized)
    """
)
async def fuse_enrollment_template(request: FuseTemplateRequest):
    """
    Fuse multiple enrollment embeddings into single template.
    """
    embeddings = request.embeddings
    quality_scores = request.quality_scores
    method = request.method
    try:
        logger.info("="*80)
        logger.info("🔗 Enrollment Template Fusion")
        logger.info("="*80)
        logger.info(f"Number of embeddings: {len(embeddings)}")
        logger.info(f"Fusion method: {method}")
        logger.info(f"Quality scores provided: {quality_scores is not None}")
        
        # Validate embeddings
        if not embeddings:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No embeddings provided"
            )
        
        for i, emb in enumerate(embeddings):
            if len(emb) != 512:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Embedding {i} has invalid size: {len(emb)} (expected 512)"
                )
        
        # Convert to numpy arrays
        embeddings_np = [np.array(emb) for emb in embeddings]
        
        # Validate quality scores if provided
        if quality_scores is not None:
            if len(quality_scores) != len(embeddings):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Number of quality scores ({len(quality_scores)}) != number of embeddings ({len(embeddings)})"
                )
        
        # Fuse embeddings
        template_array = BuffaloLEmbeddingExtractor.fuse_embeddings(
            embeddings_np,
            quality_scores=quality_scores,
            method=method
        )
        
        template = template_array.tolist()
        
        logger.info(f"✓ Template fused successfully")
        logger.info(f"Template norm: {np.linalg.norm(template_array):.6f}")
        logger.info("="*80)
        
        return {
            "success": True,
            "template": template,
            "template_size": len(template),
            "num_embeddings_fused": len(embeddings),
            "fusion_method": method,
            "message": "Enrollment template fused successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template fusion error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# ═══════════════════════════════════════════════════════════════
# Error Handlers
# ═══════════════════════════════════════════════════════════════

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "message": "An unexpected error occurred",
            "status_code": 500
        }
    )


# ═══════════════════════════════════════════════════════════════
# Startup/Shutdown Events
# ═══════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Execute on application startup"""
    logger.info("=" * 60)
    logger.info("Face Verification Service Starting")
    logger.info("=" * 60)
    logger.info("Service: Face Verification API")
    logger.info("Version: 1.0.0")
    logger.info("Phase: Production (Buffalo_L — SINGLE MODEL)")
    logger.info("=" * 60)
    logger.info("PRODUCTION endpoints (Laravel integrates these — Buffalo_L only):")
    logger.info("  - POST /extract-embedding-buffalo              : image -> Buffalo_L 512-d embedding + quality gate")
    logger.info("  - POST /extract-embedding-buffalo-with-quality : image -> Buffalo_L 512-d embedding + quality gate")
    logger.info("  - POST /enroll                                 : validate+receive a Buffalo_L embedding (Laravel stores it)")
    logger.info("  - POST /verify                                 : match stored vs live Buffalo_L embeddings")
    logger.info("  - POST /verify-images                          : verify enrollment image(s) vs a live image directly")
    logger.info("  - POST /analyze-face, /check-quality           : quality gate only (capture-time guidance)")
    logger.info("  - POST /check-liveness                         : passive anti-spoofing (PAD) on a burst of frames")
    logger.info("  - POST /fuse-enrollment-template               : fuse multiple enrollment embeddings")
    logger.info("  - GET  /health                                 : Health check")
    logger.info("=" * 60)
    logger.info("Buffalo_L Pipeline (Official InsightFace):")
    logger.info("  - Detection: SCRFD (Buffalo_L detector)")
    logger.info("  - Landmarks: Official 5-point from SCRFD")
    logger.info("  - Alignment: Official face_align.norm_crop()")
    logger.info("  - Model: Buffalo_L W600K_R50 ONNX")
    logger.info("  - Embedding: 512-D L2-normalized")
    logger.info("  - Pose: real 3D head angles from 1k3d68 (pitch/yaw/roll)")
    logger.info("=" * 60)
    logger.info("Verification:")
    logger.info("  - Algorithm: Cosine similarity (Buffalo_L, single model)")
    logger.info(f"  - Threshold: {config.VERIFICATION_THRESHOLD} (calibrated; override via FACE_MATCH_THRESHOLD)")
    logger.info("  - Confidence%: calibrated logistic centred on the threshold (config.score_to_confidence)")
    logger.info("=" * 60)
    logger.info("API Documentation: http://127.0.0.1:8000/docs")
    logger.info("=" * 60)



@app.on_event("shutdown")
async def shutdown_event():
    """Execute on application shutdown"""
    logger.info("=" * 60)
    logger.info("Face Verification Service Shutting Down")
    logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn
    import os
    
    # Railway-compatible startup: use PORT env var if available
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Disable reload in production (Railway sets PORT env var)
    reload = port == 8000  # Only reload on default local port
    
    logger.info(f"Starting server on {host}:{port} (reload={reload})")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
