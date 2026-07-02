# Face verification service (Buffalo_L) — Railway/Docker image.
FROM python:3.11-slim

# System libraries:
#  - build-essential: insightface compiles a small Cython extension at install time.
#  - libgl1 + libglib2.0-0: required by opencv-python-headless.
#  - libgomp1: required by onnxruntime.
#  - curl + ca-certificates: to fetch the LFS model files at build time (see below).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps, then force the HEADLESS OpenCV.
# insightface declares a dependency on the full `opencv-python` (GUI build), which
# needs libxcb/X11 libraries that aren't present in a slim server image and crashes
# `import cv2` with "libxcb.so.1: cannot open shared object file". We install
# everything, remove the GUI build, and force-reinstall the headless build so cv2
# resolves to a server-safe OpenCV with no X11 requirement.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && (pip uninstall -y opencv-python opencv-contrib-python || true) \
    && pip install --no-cache-dir --force-reinstall "opencv-python-headless>=4.10,<5"

# App code (the .onnx files copied here are Git LFS *pointer stubs* — Railway does
# not resolve LFS into the build context — so we re-download the real bytes below).
COPY . .

# Fetch the real model files from GitHub's LFS media endpoint (public repo, no auth).
# This replaces the LFS pointer stubs with the actual ONNX weights so onnxruntime /
# insightface can load them. If you fork/rename the repo, update MODELS_BASE.
ENV MODELS_BASE="https://media.githubusercontent.com/media/rohi22/hrm-face-detection/main"
RUN set -eu; \
    for f in \
        models/buffalo_l_w600k_r50.onnx \
        models/MiniFASNetV1SE.onnx \
        models/MiniFASNetV2.onnx \
        insightface_home/models/buffalo_l/1k3d68.onnx \
        insightface_home/models/buffalo_l/2d106det.onnx \
        insightface_home/models/buffalo_l/det_10g.onnx \
        insightface_home/models/buffalo_l/genderage.onnx \
        insightface_home/models/buffalo_l/w600k_r50.onnx ; do \
        echo "Fetching $f"; \
        curl -fSL --retry 3 --retry-delay 2 -o "$f" "$MODELS_BASE/$f"; \
        sz=$(wc -c < "$f"); \
        if [ "$sz" -lt 100000 ]; then echo "ERROR: $f is only $sz bytes (LFS fetch failed)"; exit 1; fi; \
    done; \
    echo "All model files fetched."

ENV PYTHONUNBUFFERED=1

# Railway injects $PORT at runtime.
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
