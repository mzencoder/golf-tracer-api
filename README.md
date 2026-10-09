# Golf Tracer API

Backend-only video tracer for Subpar. Upload a golf shot, poll the job, then download an MP4 with the orange detected path and red predicted continuation. The API selects the ball automatically; the client does not send a start point. Prediction is a visual estimate, not measured ball flight.

## Run

Requires Python 3.11+, FFmpeg (`ffmpeg` on `PATH`), and the included `models/best.onnx`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export GOLF_TRACER_JWT_SECRET='replace-with-a-random-secret-of-at-least-32-characters'
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8765
```

For Docker, copy `.env.example` to `.env` and change the secret:

```bash
docker build -t golf-tracer-api .
docker run --rm -p 8765:8765 --env-file .env -v golf-tracer-data:/app/data golf-tracer-api
```

The server reads environment variables directly; `.env` is not loaded automatically outside Docker. Use **one Uvicorn worker per data directory**. Jobs are processed sequentially by one background thread. SQLite keeps queue state across restarts; unfinished jobs are queued again. Store `/app/data` on a persistent volume. Output and metadata expire after 24 hours by default.

## Authentication

Every `/v1` request requires `Authorization: Bearer <token>`. Subpar's **backend** issues a short-lived HS256 JWT; never put the shared secret in a mobile or browser app. Claims: `iss=subpar`, `aud=golf-tracer`, `sub=<stable Subpar user ID, 1–128 characters>`, `scope=shot-tracer:analyze`, `iat=<Unix seconds>`, `exp=<Unix seconds>`. Set the same secret as `GOLF_TRACER_JWT_SECRET` on this server. Job status, video, and deletion are restricted to the user ID in `sub`.

Example Node.js token issuance using `jose`:

```js
import { SignJWT } from 'jose'

const token = await new SignJWT({ scope: 'shot-tracer:analyze' })
  .setProtectedHeader({ alg: 'HS256' })
  .setIssuer('subpar')
  .setAudience('golf-tracer')
  .setSubject(String(userId))
  .setIssuedAt()
  .setExpirationTime('15m')
  .sign(new TextEncoder().encode(process.env.GOLF_TRACER_JWT_SECRET))
```

This is an integration example; wiring Subpar itself is outside this repository.

## API

`GET /healthz` is public. OpenAPI documentation is at `/docs`.

1. `POST /v1/jobs` with multipart form field `video` (`.mp4`, `.mov`, `.m4v`). Returns HTTP 202: `{"id":"...","status":"queued","progress":0,"status_url":"/v1/jobs/..."}`.
2. `GET /v1/jobs/{id}` until `status` becomes `complete` or `error`. `progress` is 0–100. Success includes `video_url` and `details`; failure includes `error`.
3. `GET /v1/jobs/{id}/video` with the Bearer token returns the finished MP4. Download it with an authenticated request and then display the local file in Subpar.
4. `DELETE /v1/jobs/{id}` after completion for early cleanup.

```bash
curl -H "Authorization: Bearer $TOKEN" -F "video=@shot.mp4" http://127.0.0.1:8765/v1/jobs
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8765/v1/jobs/JOB_ID
curl -H "Authorization: Bearer $TOKEN" -o traced.mp4 http://127.0.0.1:8765/v1/jobs/JOB_ID/video
```

`details` contains `trace_found`, `source`, `detected_frames`, `motion_frames`, `predicted_frames`, `total_frames`, `fps`, `width`, `height`, `duration`, and up to 240 `trace_points`. Points have normalized coordinates `x`/`y` (0–1), time `t` in seconds, and `kind` (`seed`, `detected`, `motion`, or `predicted`). A completed video can still have `trace_found: false`; inspect this before presenting it as traced. Predicted points must not be presented as measured data.

The API rejects oversized uploads with HTTP 413 and a full queue with HTTP 429. It validates extension and whether OpenCV can decode the video; extension alone does not prove valid media.

## Video guidance

Keep the camera steady and show the ball near impact, with enough contrast between the white ball and background. Prefer daylight or bright even light and minimal occlusion. 24, 30, and 60 fps are supported; 60 fps usually gives more observed positions but takes more CPU time. A fixed camera position is not required across different uploads, though a stationary camera within each clip helps. A distant or blurred ball may be missed; the red segment is then an estimate. Very large, long, or high frame rate clips use proportionally more time, memory, and disk.

## Configuration and deployment

| Variable | Default | Purpose |
| --- | --- | --- |
| `GOLF_TRACER_JWT_SECRET` | required | HS256 secret, at least 32 characters |
| `GOLF_TRACER_DATA_DIR` | `./data` | Persistent SQLite database, uploads, results |
| `GOLF_TRACER_MAX_UPLOAD_MB` | `250` | Upload size limit |
| `GOLF_TRACER_MAX_ACTIVE_JOBS` | `8` | Maximum queued plus processing jobs |
| `GOLF_TRACER_RETENTION_HOURS` | `24` | Retention for finished jobs |
| `GOLF_TRACER_ALLOWED_ORIGINS` | empty | Comma-separated direct browser origins, if needed |

Run behind HTTPS and a reverse proxy. Proxy upload limits and timeouts must allow your upload size. This single-process queue suits an initial deployment. Horizontal scaling requires a shared queue and result storage; do not run multiple Uvicorn workers against the same data directory.

The bundled model's source and licensing caveat are recorded in [MODEL_SOURCE.md](MODEL_SOURCE.md). Verify its license before commercial deployment.
