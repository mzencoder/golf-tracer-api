"""Authenticated video-analysis API for Subpar."""

from contextlib import asynccontextmanager, contextmanager
import json
import os
from pathlib import Path
import sqlite3
from threading import Event, Thread
import time
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import jwt

from tracker import process_video


BASE = Path(__file__).parent
DATA = Path(os.getenv("GOLF_TRACER_DATA_DIR", BASE / "data"))
UPLOADS = DATA / "uploads"
RESULTS = DATA / "results"
DB = DATA / "jobs.sqlite3"
MAX_BYTES = int(os.getenv("GOLF_TRACER_MAX_UPLOAD_MB", "250")) * 1024 * 1024
MAX_ACTIVE_JOBS = int(os.getenv("GOLF_TRACER_MAX_ACTIVE_JOBS", "8"))
RETENTION_HOURS = int(os.getenv("GOLF_TRACER_RETENTION_HOURS", "24"))
JWT_SECRET = os.getenv("GOLF_TRACER_JWT_SECRET", "")
JWT_ISSUER = "subpar"
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
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                details TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        connection.execute("CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner, created_at)")
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
        try:
            details = process_video(source, output, lambda value: update_job(job_id, progress=value))
            update_job(job_id, status="complete", progress=100,
                       details=json.dumps(details, separators=(",", ":")))
        except Exception as error:
            output.unlink(missing_ok=True)
            update_job(job_id, status="error", error=str(error)[:500])
        finally:
            source.unlink(missing_ok=True)
            output.with_suffix(".raw.mp4").unlink(missing_ok=True)


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
async def create_job(video: UploadFile = File(...), owner: str = Depends(current_user)):
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v"}:
        raise HTTPException(400, "Upload MP4, MOV, or M4V video")
    job_id = uuid4().hex
    source = UPLOADS / f"{job_id}{suffix}"
    size = 0
    try:
        with source.open("wb") as target:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, f"Video exceeds {MAX_BYTES // (1024 * 1024)} MB")
                target.write(chunk)
        if size == 0:
            raise HTTPException(400, "Empty video")
        now = time.time()
        with connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            count = connection.execute(
                "SELECT count(*) FROM jobs WHERE status IN ('queued', 'processing')"
            ).fetchone()[0]
            if count >= MAX_ACTIVE_JOBS:
                raise HTTPException(429, "Analysis queue is full")
            connection.execute(
                "INSERT INTO jobs(id, owner, suffix, status, progress, created_at, updated_at) "
                "VALUES (?, ?, ?, 'queued', 0, ?, ?)",
                (job_id, owner, suffix, now, now),
            )
    except Exception:
        source.unlink(missing_ok=True)
        raise
    finally:
        await video.close()
    worker_wakeup.set()
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
