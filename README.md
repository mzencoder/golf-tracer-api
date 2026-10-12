# Golf Tracer API

A standalone API that detects a golf ball in a shot video and renders its path into an MP4. Upload a clip, poll the job, and download the result. The API also supports manually supplied trace points and optional generic video overlays. It needs no other application or account system.

The orange path comes from image detections or motion tracking. The red continuation is a visual prediction, **not** a launch-monitor measurement. A completed job can have `trace_found: false` when the video does not provide enough evidence for a path.

## Example video

This 12-second clip shows the rendered shot path. The red segment is predicted flight.

https://github.com/user-attachments/assets/1d316933-39eb-43f3-b240-84dba1851531

## Quick start

Requires Python 3.11+, FFmpeg on `PATH`, and the included `models/best.onnx`. See [model provenance](MODEL_SOURCE.md) before redistributing the model or using it commercially.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
printf 'GOLF_TRACER_JWT_SECRET=%s\n' "$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')" > .env
set -a; . ./.env; set +a
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8765
```

In another shell, load the same local secret and create a short-lived token:

```bash
set -a; . ./.env; set +a
export TOKEN="$($PWD/.venv/bin/python - <<'PY'
import os
import time
import jwt

now = int(time.time())
print(jwt.encode({
    'iss': os.getenv('GOLF_TRACER_JWT_ISSUER', 'golf-tracer'),
    'aud': 'golf-tracer',
    'sub': 'local-user',
    'scope': 'shot-tracer:analyze',
    'iat': now,
    'exp': now + 900,
}, os.environ['GOLF_TRACER_JWT_SECRET'], algorithm='HS256'))
PY
)"
curl -H "Authorization: Bearer $TOKEN" -F "video=@shot.mp4" http://127.0.0.1:8765/v1/jobs
```

Replace `shot.mp4` with your own golf-shot video. The response contains a job ID. Poll and download with the same token:

```bash
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8765/v1/jobs/JOB_ID
curl -H "Authorization: Bearer $TOKEN" -o traced.mp4 http://127.0.0.1:8765/v1/jobs/JOB_ID/video
```

`GET /healthz` is public, and interactive API documentation is at `/docs`. For a browser or mobile client, issue tokens on **your backend** and keep the shared secret off the client. Set `GOLF_TRACER_ALLOWED_ORIGINS` if the browser calls this API directly. To use an existing token issuer, set `GOLF_TRACER_JWT_ISSUER` to that issuer's `iss` value on the API.

## Docker

Copy `.env.example` to `.env`, replace the sample secret, then run:

```bash
docker build -t golf-tracer-api .
docker run --rm -p 8765:8765 --env-file .env -v golf-tracer-data:/app/data golf-tracer-api
```

The server reads environment variables directly; it does not automatically load `.env` outside Docker. Use one Uvicorn worker per data directory. SQLite persists queue state across restarts, and unfinished jobs are requeued. Output and metadata expire after 24 hours by default.

## API

All `/v1` endpoints require `Authorization: Bearer <token>`. The HS256 JWT must have `iss` matching `GOLF_TRACER_JWT_ISSUER` (default `golf-tracer`), `aud=golf-tracer`, a stable `sub` string of 1–128 characters, `scope=shot-tracer:analyze`, and `iat` and `exp` Unix timestamps. Each job is accessible only to the matching `sub`.

1. `POST /v1/jobs`: multipart `video` (`.mp4`, `.mov`, `.m4v`). By default this runs automatic tracing. To render a manual path without detection, also send `trace` as JSON containing 2–240 `{x,y,t}` points, with normalized coordinates and seconds from the video start. To decorate an already traced MP4, send `decorate=true`, optional `club` (up to 60 characters), and `watermark=true` or `false`. At least one overlay must be selected. `trace` and `decorate=true` cannot be combined. Returns HTTP 202 with `id`, `status`, `progress`, and `status_url`.
2. `GET /v1/jobs/{id}`: poll until `status` is `complete` or `error`. Success includes `video_url` and `details`.
3. `GET /v1/jobs/{id}/video`: download the completed MP4 with the Bearer token.
4. `DELETE /v1/jobs/{id}`: remove a finished job early.

The source upload is removed after processing. The rendered MP4 stays in `data/results` until deletion or retention expiry. `details` includes `trace_found`, `source`, `detected_frames`, `motion_frames`, `predicted_frames`, `total_frames`, `fps`, dimensions, duration, and up to 240 `trace_points`. Each point has normalized `x`/`y`, time `t`, and `kind` (`seed`, `detected`, `motion`, or `predicted`). Treat `predicted` points as estimates.

Oversized uploads return HTTP 413; a full queue returns HTTP 429. Video decoding is validated during processing. A valid file extension alone does not guarantee readable media.

## Video guidance

Keep the camera steady during each clip and show the ball near impact with good contrast and even light. Distant, blurred, or occluded balls may be missed. The detector finds candidate balls in individual frames; continuous flight tracking is less reliable and depends on visible motion. 24, 30, and 60 fps are supported. Larger clips take more CPU time, memory, and disk.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GOLF_TRACER_JWT_SECRET` | required | HS256 secret, at least 32 characters |
| `GOLF_TRACER_JWT_ISSUER` | `golf-tracer` | Expected JWT issuer |
| `GOLF_TRACER_DATA_DIR` | `./data` | SQLite database, uploads, results |
| `GOLF_TRACER_MAX_UPLOAD_MB` | `250` | Upload size limit |
| `GOLF_TRACER_MAX_ACTIVE_JOBS` | `8` | Queued plus processing jobs |
| `GOLF_TRACER_RETENTION_HOURS` | `24` | Finished-job retention |
| `GOLF_TRACER_ALLOWED_ORIGINS` | empty | Comma-separated browser origins |

Run behind HTTPS and a reverse proxy. Proxy upload limits and timeouts must allow your chosen clip size. Horizontal scaling needs shared queue and result storage; multiple Uvicorn workers must not share one data directory.

At Uvicorn's INFO log level, uploads and jobs emit `shot_upload_request`, `shot_upload`, and `shot_timing` lines keyed by `job_id`. Timings ending in `_s` are wall-clock seconds. `inference_s` is included inside detection phases; do not add it to their duration. Logs include job IDs and counts, not video contents, tokens, or user IDs.

## License

The API source code is licensed under [MIT](LICENSE). The bundled ONNX model is a third-party asset with separate, conflicting license notices; see [MODEL_SOURCE.md](MODEL_SOURCE.md). The generated text overlays have no application-specific branding.
