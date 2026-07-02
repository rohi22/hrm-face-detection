# Face Verification Service (Buffalo_L)

Stateless FastAPI microservice for face **embedding extraction**, **quality gating**,
**1:1 matching**, and **passive liveness** — powered by a single model,
**InsightFace Buffalo_L** (SCRFD detector + ArcFace R50, 512-d embeddings).

It holds **no database**. The Laravel backend is its only client and owns all storage.
See [`BACKEND_INTEGRATION.md`](BACKEND_INTEGRATION.md) for the enroll/verify flows
and the DB schema.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/` , `/health` | Health check |
| POST | `/extract-embedding-buffalo-with-quality` | image → 512-d embedding **+ quality gate** (primary) |
| POST | `/extract-embedding-buffalo` | image → embedding + quality |
| POST | `/verify` | 1:1 cosine match of `stored_embedding` vs `live_embedding` |
| POST | `/verify-images` | enrollment image(s) vs a live image (quality gate + match, all-in-one) |
| POST | `/check-quality` | fast quality-only pre-check (no embedding) |
| POST | `/check-liveness` | passive anti-spoof over a burst of frames (MiniFASNet) |
| POST | `/analyze-face` | quality gate only, capture-time guidance |
| POST | `/enroll` | validate a 512-d embedding payload (no storage) |
| POST | `/fuse-enrollment-template` | fuse N enrollment embeddings into one template |

Interactive docs at `/docs` when running.

---

## Configuration (environment variables)

All thresholds live here — this is the single source of truth. Override on Railway if needed.

| Var | Default | Meaning |
|---|---|---|
| `FACE_MATCH_THRESHOLD` | `0.45` | Cosine match threshold (ArcFace genuine pairs ~0.45–0.85; **never** 0.75+). |
| `FACE_LIVENESS_THRESHOLD` | `0.25` | Passive PAD `p_real` cutoff. Tuned low for cheap test webcams; **raise to ~0.45–0.5 for production cameras**. |
| `FACE_EYE_MIN_SKIN` | `0.10` | Sunglasses/occlusion gate (eye-region skin fraction). Raise to ~0.20–0.26 for good cameras. |
| `INSIGHTFACE_HOME` | *(bundled)* | Defaults to the in-repo `insightface_home/` pack so nothing downloads at runtime. |
| `PORT` | `8000` | Injected by Railway. |

---

## Models & Git LFS  ⚠️ read before committing

Model files are large and are tracked with **Git LFS** (`.gitattributes` at the repo root:
`*.onnx filter=lfs`). What ships in the repo:

- `models/buffalo_l_w600k_r50.onnx` (~166 MB) — recognition (used by `embedding_buffalo.py`).
- `models/MiniFASNetV2.onnx`, `models/MiniFASNetV1SE.onnx` (~1.7 MB each) — liveness.
- `insightface_home/models/buffalo_l/*.onnx` (~340 MB) — the InsightFace pack (detection,
  3D pose, etc.). Bundled so **nothing is downloaded at cold start** (`INSIGHTFACE_HOME`
  points here by default) — important on Railway's ephemeral filesystem.

**One-time LFS setup before your first push:**
```bash
git lfs install
git add .gitattributes
git add face-service/           # .onnx files are stored as LFS pointers
git commit -m "Add face-service (models via LFS)"
git push
```

> **LFS bandwidth note:** the models total ~**500 MB**. GitHub's free LFS tier is 1 GB
> storage + 1 GB bandwidth/month, and **every Railway deploy pulls the LFS files**, so you
> may exhaust free bandwidth after a couple of deploys. Options: buy a GitHub LFS data pack,
> or (leaner) delete the redundant standalone `models/buffalo_l_w600k_r50.onnx` and point
> `embedding_buffalo.py` at the identical `insightface_home/models/buffalo_l/w600k_r50.onnx`
> (saves ~166 MB) — ask the developer to do this if bandwidth is tight.

**Housekeeping left for you (a dev server had it locked):** delete the now-unused
`models/face_detector.tflite` before committing — stop any running `python app.py`, then
`rm face-service/models/face_detector.tflite`. It is not referenced by any code.

---

## Deploy to Railway

The service ships with `Procfile`, `railway.toml`, `nixpacks.toml`, `runtime.txt`.

1. Push the repo to GitHub (with LFS, per above).
2. On Railway: **New Project → Deploy from GitHub repo** → select `rohi22/hrm-face-detection`.
3. Leave **Root Directory** as the repository root — this repo *is* the service
   (`app.py` is at the root). Make sure Railway has Git LFS enabled so the models resolve.
4. Railway auto-detects Nixpacks (Python 3.11) and runs
   `uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1`.
5. Health check path `/health` (timeout 300s to cover first boot / model load).
6. (Optional) set `FACE_MATCH_THRESHOLD`, `FACE_LIVENESS_THRESHOLD`, `FACE_EYE_MIN_SKIN`
   in the Railway service variables.
7. After deploy, verify: `curl https://<your-service>.up.railway.app/health`.

Give the resulting URL to the Laravel developer (`PYTHON_FACE_SERVICE_URL`).

---

## Run locally

```bash
cd face-service
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
# docs: http://127.0.0.1:8000/docs
```

Requires the model files present (via LFS checkout). Python 3.11.
