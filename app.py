"""Authenticated golf-shot video analysis API."""

from contextlib import asynccontextmanager, contextmanager
import json
import logging
import os
from pathlib import Path
import sqlite3
from threading import Event, Thread
import time
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import jwt

from tracker import process_video, render_branding_video, render_manual_video


logger = logging.getLogger("uvicorn.error")
BASE = Path(__file__).parent
DATA = Path(os.getenv("GOLF_TRACER_DATA_DIR", BASE / "data"))
UPLOADS = DATA / "uploads"
RESULTS = DATA / "results"
DB = DATA / "jobs.sqlite3"
MAX_BYTES = int(os.getenv("GOLF_TRACER_MAX_UPLOAD_MB", "250")) * 1024 * 1024
MAX_ACTIVE_JOBS = int(os.getenv("GOLF_TRACER_MAX_ACTIVE_JOBS", "8"))
RETENTION_HOURS = int(os.getenv("GOLF_TRACER_RETENTION_HOURS", "24"))
JWT_SECRET = os.getenv("GOLF_TRACER_JWT_SECRET", "")
JWT_ISSUER = os.getenv("GOLF_TRACER_JWT_ISSUER", "golf-tracer")
JWT_AUDIENCE = "golf-tracer"
JWT_SCOPE = "shot-tracer:analyze"
worker_wakeup = Event()
worker_stop = Event()


@contextmanager
def connect():
    connection = sqlite3.connect(DB, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    for directory in (DATA, UPLOADS, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    with connect() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                suffix TEXT NOT NULL,
                trace_json TEXT,
                club_name TEXT,
                decorate INTEGER NOT NULL DEFAULT 0,
                watermark INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                details TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        connection.execute("CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner, created_at)")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
        if "trace_json" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN trace_json TEXT")
        if "club_name" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN club_name TEXT")
        if "decorate" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN decorate INTEGER NOT NULL DEFAULT 0")
        if "watermark" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN watermark INTEGER NOT NULL DEFAULT 1")
        connection.execute("UPDATE jobs SET status = 'queued' WHERE status = 'processing'")


def prune_finished() -> None:
    cutoff = time.time() - RETENTION_HOURS * 3600
    with connect() as connection:
        rows = connection.execute(
            "SELECT id, suffix FROM jobs WHERE status IN ('complete', 'error') AND updated_at < ?",
            (cutoff,),
        ).fetchall()
        for row in rows:
            (UPLOADS / f"{row['id']}{row['suffix']}").unlink(missing_ok=True)
            (RESULTS / f"{row['id']}.mp4").unlink(missing_ok=True)
            connection.execute("DELETE FROM jobs WHERE id = ?", (row["id"],))


def claim_job() -> sqlite3.Row | None:
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
        ).fetchone()
        if row:
            connection.execute(
                "UPDATE jobs SET status = 'processing', updated_at = ? WHERE id = ?",
                (time.time(), row["id"]),
            )
        return row


def update_job(job_id: str, **fields) -> None:
    fields["updated_at"] = time.time()
    columns = ", ".join(f"{key} = ?" for key in fields)
    with connect() as connection:
        connection.execute(
            f"UPDATE jobs SET {columns} WHERE id = ?", (*fields.values(), job_id)
        )


def run_worker() -> None:
    last_prune = 0.0
    while not worker_stop.is_set():
        if time.time() - last_prune > 3600:
            prune_finished()
            last_prune = time.time()
        row = claim_job()
        if row is None:
            worker_wakeup.wait(timeout=5)
            worker_wakeup.clear()
            continue
        job_id = row["id"]
        source = UPLOADS / f"{job_id}{row['suffix']}"
        output = RESULTS / f"{job_id}.mp4"
        processing_started = time.perf_counter()
        queue_wait_s = max(0.0, time.time() - row["created_at"])
        timings: dict[str, float] = {}
        status = "complete"
        mode = "decorate" if row["decorate"] else "manual" if row["trace_json"] else "auto"
        media = ""
        try:
            if row["decorate"]:
                details = render_branding_video(source, output, row["club_name"],
                                                bool(row["watermark"]),
                                                lambda value: update_job(job_id, progress=value),
                                                timings)
            elif row["trace_json"]:
                details = render_manual_video(source, output, json.loads(row["trace_json"]),
                                              lambda value: update_job(job_id, progress=value),
                                              timings)
            else:
                details = process_video(source, output,
                                        lambda value: update_job(job_id, progress=value),
                                        timings)
            media = (f" frames={details['total_frames']} fps={details['fps']}"
                     f" width={details['width']} height={details['height']}"
                     f" trace_found={details['trace_found']}")
            update_job(job_id, status="complete", progress=100,
                       details=json.dumps(details, separators=(",", ":")))
        except Exception as error:
            status = "error"
            output.unlink(missing_ok=True)
            update_job(job_id, status="error", error=str(error)[:500])
            logger.exception("shot_processing_failed job_id=%s", job_id)
        finally:
            source.unlink(missing_ok=True)
            output.with_suffix(".raw.mp4").unlink(missing_ok=True)
            timings["processing_total_s"] = time.perf_counter() - processing_started
            metrics = " ".join(f"{name}={value:.3f}" for name, value in timings.items())
            logger.info("shot_timing job_id=%s mode=%s status=%s%s queue_wait_s=%.3f %s",
                        job_id, mode, status, media, queue_wait_s, metrics)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if len(JWT_SECRET) < 32:
        raise RuntimeError("GOLF_TRACER_JWT_SECRET must contain at least 32 characters")
    init_db()
    worker_stop.clear()
    worker = Thread(target=run_worker, name="golf-tracer-worker", daemon=True)
    worker.start()
    try:
        yield
    finally:
        worker_stop.set()
        worker_wakeup.set()
        worker.join(timeout=5)


app = FastAPI(title="Golf Tracer API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def time_upload_request(request: Request, call_next):
    if request.method != "POST" or request.url.path != "/v1/jobs":
        return await call_next(request)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.info("shot_upload_request status=error ingress_s=%.3f",
                    time.perf_counter() - started)
        raise
    logger.info("shot_upload_request job_id=%s status=%s ingress_s=%.3f",
                getattr(request.state, "upload_job_id", "none"), response.status_code,
                time.perf_counter() - started)
    return response


origins = [item.strip() for item in os.getenv("GOLF_TRACER_ALLOWED_ORIGINS", "").split(",")
           if item.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_methods=["GET", "POST", "DELETE"],
                       allow_headers=["Authorization", "Content-Type"])


def current_user(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Bearer token required")
    try:
        claims = jwt.decode(
            authorization[7:], JWT_SECRET, algorithms=["HS256"],
            audience=JWT_AUDIENCE, issuer=JWT_ISSUER,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token") from None
    subject = claims.get("sub")
    if (claims.get("scope") != JWT_SCOPE or not isinstance(subject, str)
            or not 1 <= len(subject) <= 128):
        raise HTTPException(403, "Token does not allow shot analysis")
    return subject


def owned_job(job_id: str, owner: str) -> sqlite3.Row:
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM jobs WHERE id = ? AND owner = ?", (job_id, owner)
        ).fetchone()
    if row is None:
        raise HTTPException(404, "Job not found")
    return row


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.post("/v1/jobs", status_code=202)
async def create_job(request: Request, video: UploadFile = File(...),
                     trace: str | None = Form(default=None),
                     club: str | None = Form(default=None),
                     decorate: bool = Form(default=False),
                     watermark: bool = Form(default=True),
                     owner: str = Depends(current_user)):
    handler_started = time.perf_counter()
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v"}:
        raise HTTPException(400, "Upload MP4, MOV, or M4V video")
    club_name = club.strip() if club else None
    if club_name and (len(club_name) > 60 or any(ord(character) < 32 for character in club_name)):
        raise HTTPException(400, "Club name must be at most 60 printable characters")
    if decorate and trace is not None:
        raise HTTPException(400, "Decoration cannot include manual trace points")
    if decorate and not club_name and not watermark:
        raise HTTPException(400, "Select a club badge or watermark to render")
    trace_json = None
    if trace is not None:
        try:
            points = json.loads(trace)
            if (not isinstance(points, list) or not 2 <= len(points) <= 240 or
                    any(not isinstance(point, dict) or
                        any(not isinstance(point.get(key), (int, float)) or
                            isinstance(point.get(key), bool) for key in ("x", "y", "t")) or
                        not 0 <= point["x"] <= 1 or not 0 <= point["y"] <= 1 or
                        not 0 <= point["t"] <= 3600 for point in points) or
                    any(points[index]["t"] < points[index - 1]["t"]
                        for index in range(1, len(points))) or
                    points[-1]["t"] <= points[0]["t"]):
                raise ValueError()
            trace_json = json.dumps(points, separators=(",", ":"))
        except (ValueError, TypeError, KeyError):
            raise HTTPException(400, "Invalid manual trace points") from None
    job_id = uuid4().hex
    source = UPLOADS / f"{job_id}{suffix}"
    size = 0
    try:
        save_started = time.perf_counter()
        with source.open("wb") as target:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, f"Video exceeds {MAX_BYTES // (1024 * 1024)} MB")
                target.write(chunk)
        if size == 0:
            raise HTTPException(400, "Empty video")
        save_s = time.perf_counter() - save_started
        now = time.time()
        enqueue_started = time.perf_counter()
        with connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            count = connection.execute(
                "SELECT count(*) FROM jobs WHERE status IN ('queued', 'processing')"
            ).fetchone()[0]
            if count >= MAX_ACTIVE_JOBS:
                raise HTTPException(429, "Analysis queue is full")
            connection.execute(
                "INSERT INTO jobs(id, owner, suffix, trace_json, club_name, decorate, watermark, status, progress, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', 0, ?, ?)",
                (job_id, owner, suffix, trace_json, club_name, int(decorate), int(watermark), now, now),
            )
        enqueue_s = time.perf_counter() - enqueue_started
    except Exception:
        source.unlink(missing_ok=True)
        raise
    finally:
        await video.close()
    worker_wakeup.set()
    request.state.upload_job_id = job_id
    logger.info("shot_upload job_id=%s bytes=%d save_s=%.3f enqueue_s=%.3f handler_s=%.3f",
                job_id, size, save_s, enqueue_s, time.perf_counter() - handler_started)
    return {"id": job_id, "status": "queued", "progress": 0,
            "status_url": f"/v1/jobs/{job_id}"}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, owner: str = Depends(current_user)):
    row = owned_job(job_id, owner)
    response = {"id": row["id"], "status": row["status"], "progress": row["progress"]}
    if row["status"] == "complete":
        response["details"] = json.loads(row["details"])
        response["video_url"] = f"/v1/jobs/{job_id}/video"
    elif row["status"] == "error":
        response["error"] = row["error"]
    return response


@app.get("/v1/jobs/{job_id}/video")
def get_video(job_id: str, owner: str = Depends(current_user)):
    row = owned_job(job_id, owner)
    output = RESULTS / f"{job_id}.mp4"
    if row["status"] != "complete" or not output.exists():
        raise HTTPException(404, "Result not available")
    return FileResponse(output, media_type="video/mp4", filename="golf-tracer.mp4",
                        headers={"Cache-Control": "private, no-store"})


@app.delete("/v1/jobs/{job_id}", status_code=204)
def delete_job(job_id: str, owner: str = Depends(current_user)):
    row = owned_job(job_id, owner)
    if row["status"] in {"queued", "processing"}:
        raise HTTPException(409, "Wait for the job to finish before deleting it")
    (RESULTS / f"{job_id}.mp4").unlink(missing_ok=True)
    with connect() as connection:
        connection.execute("DELETE FROM jobs WHERE id = ? AND owner = ?", (job_id, owner))
