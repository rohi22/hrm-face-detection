# Backend Integration Guide — Face Enrollment & Verification (Laravel)

This is the hand-off spec for the **Laravel backend developer**. It describes the
two pieces you need to build — **enrollment** and **verification** — how they call
the Python face-service, and the database you need.

> **Architecture in one line:**
> **Mobile app → Laravel → Python face-service (Buffalo_L)**.
> The mobile app NEVER talks to Python directly. Laravel is the only client of the
> Python service. Python is **stateless** — it holds no database; **Laravel owns all
> storage** (employee ↔ face template).

```
┌─────────┐   HTTPS (Bearer)   ┌──────────┐   HTTP (internal)   ┌───────────────────┐
│ Flutter │ ─────────────────▶ │  Laravel │ ──────────────────▶ │ Python face-service│
│  app    │  POST /face/verify │  backend │  /extract… /verify  │ (Railway, Buffalo_L)│
└─────────┘   (live image)     └──────────┘                     └───────────────────┘
                                     │
                                     ▼
                                ┌─────────┐
                                │   DB    │  face_templates (512-d embedding per employee)
                                └─────────┘
```

Single model everywhere: **InsightFace Buffalo_L** (ArcFace R50, 512-d embeddings,
cosine match). The match **threshold and all quality/liveness thresholds live in the
Python service config** — do **not** hardcode `0.45`/`0.75`/etc. in Laravel. Trust the
Python response.

---

## 0. The Python service (what Laravel calls)

Base URL = the Railway deployment URL of `face-service/` (e.g.
`https://face-service-production.up.railway.app`). Health check: `GET /health`.
Full deploy instructions are in [`README.md`](README.md).

Endpoints you will use:

| Endpoint | Body | Returns |
|---|---|---|
| `POST /extract-embedding-buffalo-with-quality` | multipart `image` (file) | `embedding[512]` + `quality{passed,message,failures,metrics}` + `detection{face_count,bbox,det_score}` |
| `POST /verify` | JSON `{emp_id, stored_embedding[512], live_embedding[512], threshold?}` | `{verified, score, threshold, confidence, message, details{margin}}` |
| `POST /check-quality` | multipart `image` | `{passed, message, failures[], metrics{}}` (quality only, no embedding) |
| `POST /fuse-enrollment-template` | JSON `{embeddings[[512]…], method}` | one fused `embedding[512]` (use if enrolling from several photos) |
| `GET /health` | — | `{status, service, version}` |

You store the `embedding[512]` array (from extraction) in your DB. On verification you
send the stored one + the freshly-extracted live one to `/verify`.

---

## 1. Database

Minimum one table (adjust names to your conventions):

```sql
CREATE TABLE face_templates (
    id            BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    emp_id        BIGINT UNSIGNED NOT NULL,           -- FK -> employees.id (or att_id)
    embedding     JSON NOT NULL,                      -- 512 floats, L2-normalized
    quality_score DECIMAL(5,4) NULL,                  -- optional (det_score / gate metric)
    image_path    VARCHAR(255) NULL,                  -- optional: stored enrollment photo
    is_active     TINYINT(1) NOT NULL DEFAULT 1,
    enrolled_at   TIMESTAMP NULL,
    updated_at    TIMESTAMP NULL,
    created_at    TIMESTAMP NULL,
    UNIQUE KEY uniq_emp_active (emp_id, is_active),
    FOREIGN KEY (emp_id) REFERENCES employees(id) ON DELETE CASCADE
);
```

Notes:
- Store the embedding as **JSON array of 512 floats** (or a packed binary blob if you
  prefer). It is already **L2-normalized** — store it verbatim.
- One **active** template per employee. Re-enrollment = set old row `is_active=0` and
  insert a new one (keeps history/audit).
- The Python service does not need or use this table — it is passed the arrays.

---

## 2. Enrollment (HR uploads an employee's photo)

**Where:** your existing admin/HR web panel. HR uploads one clear, frontal, well-lit
photo per employee (ID-photo style). This is the single most important input — match
accuracy is capped by enrollment photo quality.

**Flow:**

1. HR submits `emp_id` + image to your Laravel enrollment endpoint.
2. Laravel forwards the image to Python:
   `POST {PYTHON_URL}/extract-embedding-buffalo-with-quality` (multipart, field `image`).
3. Read the response:
   - If `quality.passed == false` → reject, show `quality.message` to HR
     (e.g. "Remove sunglasses", "Look straight at the camera", "Too dark"). Do **not** store.
   - If `success == true && quality.passed == true` → store `embedding` (512 floats) in
     `face_templates` for that `emp_id`, mark `is_active=1` (deactivate any previous).
4. (Optional) For higher robustness, collect 2–3 photos, extract each embedding, and call
   `POST /fuse-enrollment-template` with the list to get one fused template to store.

**Example (Laravel, Guzzle):**

```php
$resp = Http::attach('image', file_get_contents($request->file('photo')->path()), 'enroll.jpg')
    ->post($pythonUrl.'/extract-embedding-buffalo-with-quality');

$body = $resp->json();
if (!($body['success'] ?? false) || !($body['quality']['passed'] ?? false)) {
    return response()->json([
        'status'  => false,
        'message' => $body['quality']['message'] ?? 'Face not clear. Try another photo.',
    ], 422);
}

FaceTemplate::where('emp_id', $empId)->update(['is_active' => 0]);
FaceTemplate::create([
    'emp_id'        => $empId,
    'embedding'     => $body['embedding'],            // array of 512 floats
    'quality_score' => $body['detection']['det_score'] ?? null,
    'is_active'     => 1,
    'enrolled_at'   => now(),
]);
```

---

## 3. Verification (mobile check-in) — build `POST /face/verify`

This is the endpoint the **mobile app already calls**. Contract is fixed on the app
side (see §4) — you implement the Laravel handler.

**Request (from the app):**
- `POST {API_BASE_URL}/face/verify`
- Headers: `Authorization: Bearer <token>`, `Accept: application/json`
- multipart/form-data:
  - `image` — the live check-in JPEG (file)
  - `att_id` — attendance id (string)
  - `emp_id` — employee id (string)

**What Laravel does:**

1. Authenticate the Bearer token (same as your other attendance endpoints).
2. Extract the live embedding + quality from the uploaded image:
   `POST {PYTHON_URL}/extract-embedding-buffalo-with-quality` (multipart `image`).
   - If `quality.passed == false` → respond `{ verified:false, message: quality.message }`.
     (Catches sunglasses / off-pose / dark / blurry on the server.)
3. Load the employee's **active** stored template embedding from `face_templates`.
   - If none → respond `{ verified:false, message:"No face enrolled for this employee." }`.
4. Ask Python to score the match (keeps the threshold in one place):
   `POST {PYTHON_URL}/verify` with JSON
   `{ emp_id, stored_embedding, live_embedding }`.
5. Return the result to the app (shape below). On `verified==true` the app then does its
   GPS geofence + writes attendance via your existing `/attendance` endpoint — **face
   verification does not write attendance itself.**

**Response (to the app)** — the app reads these keys (extra keys are ignored):

```json
{
  "verified": true,
  "score": 0.69,
  "confidence_percent": 96.4,
  "live": true,
  "message": "Face verified"
}
```
- `verified` (bool) — REQUIRED. The app also accepts `match` / `success` / `status` as aliases.
- `score` (0–1 cosine), `confidence_percent` (0–100) — optional, shown in logs/UI.
- `message` — shown to the user on failure (make it human-friendly).

**Example (Laravel):**

```php
// 2) live embedding + quality
$ext = Http::attach('image', file_get_contents($request->file('image')->path()), 'live.jpg')
    ->post($pythonUrl.'/extract-embedding-buffalo-with-quality')->json();

if (!($ext['success'] ?? false) || !($ext['quality']['passed'] ?? false)) {
    return response()->json(['verified' => false,
        'message' => $ext['quality']['message'] ?? 'Face not clear.']);
}

// 3) stored template
$tpl = FaceTemplate::where('emp_id', $empId)->where('is_active', 1)->first();
if (!$tpl) {
    return response()->json(['verified' => false, 'message' => 'No face enrolled.']);
}

// 4) match (Python owns the threshold)
$match = Http::post($pythonUrl.'/verify', [
    'emp_id'          => (int) $empId,
    'stored_embedding'=> $tpl->embedding,        // array of 512 floats
    'live_embedding'  => $ext['embedding'],
])->json();

return response()->json([
    'verified'           => $match['verified'] ?? false,
    'score'              => $match['score'] ?? null,
    'confidence_percent' => $match['confidence'] ?? null, // or map string->%
    'message'            => ($match['verified'] ?? false)
                              ? 'Face verified'
                              : 'Face does not match the enrolled employee.',
]);
```

> **Liveness:** the mobile app runs an **active random gesture challenge** (blink / turn /
> smile) on-device before capturing, so a printed photo or replayed video is rejected
> before anything is uploaded. Passive server-side anti-spoofing (`POST /check-liveness`,
> needs a small burst of frames) is available if you later want a second layer — optional,
> not required for v1.

---

## 4. The fixed mobile contract (already implemented — do not change the app)

- Endpoint path is configurable in the app via `assets/env/app.env` → `FACE_VERIFY_PATH`
  (default `/face/verify`), resolved against `API_BASE_URL`.
- The app sends multipart `image` + `att_id` + `emp_id` with the Bearer token.
- The app treats the check-in as verified only when the response has
  `verified` (or `match`/`success`/`status`) == true; otherwise it shows `message` and
  lets the user retry.

Relevant app code (for reference, no changes needed):
`lib/core/api/attendance_api_service.dart` → `verifyFace()`,
`lib/features/attendance/models/attendance_api_models.dart` → `FaceVerifyResult`,
`lib/features/attendance/screens/face_verification_screen.dart` (the check-in flow).

---

## 5. Checklist for the backend developer

- [ ] Deploy `face-service/` to Railway (see its README) and note the URL. Confirm `GET /health`.
- [ ] Add `PYTHON_FACE_SERVICE_URL` to Laravel `.env`; never expose it to the app.
- [ ] Migration: `face_templates` table.
- [ ] Enrollment endpoint in the HR panel (extract → quality check → store embedding).
- [ ] `POST /face/verify` endpoint (auth → extract+quality → load template → `/verify` → respond).
- [ ] Keep thresholds in Python; just relay its decision.
- [ ] (Optional) 2–3 photo enrollment with `/fuse-enrollment-template`.
- [ ] (Optional) server-side passive liveness via `/check-liveness`.
