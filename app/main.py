import json
import os
import secrets
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sarvamai import SarvamAI

from app.embedders import Embedder, load_embedder

API_KEY = os.environ.get("INFERENCE_API_KEY", "")
MAX_BATCH = int(os.environ.get("EMBED_MAX_BATCH", "64"))

# Recordings below this are silence or a stray tap; never worth a transcription call.
MIN_AUDIO_BYTES = 2048
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_RUNNING_JOBS = 4
JOB_TTL_SECONDS = 3600

embedders: dict[str, Embedder] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load once at startup so the first request doesn't pay the model-load cost.
    embedders["embed"] = load_embedder()
    yield
    embedders.clear()


app = FastAPI(title="memory-inference", lifespan=lifespan)


def require_key(x_inference_key: str = Header(default="")) -> None:
    if not API_KEY or not secrets.compare_digest(x_inference_key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid api key")


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)


class EmbedResponse(BaseModel):
    model: str
    dimensions: int
    embeddings: list[list[float]]


@app.get("/health")
def health() -> dict:
    e = embedders.get("embed")
    return {
        "status": "ok",
        "embed_model": e.label if e else None,
        "loaded": e is not None,
        # Presence only, never the values: lets you check .env was actually loaded.
        "inference_key_set": bool(API_KEY),
        "sarvam_key_set": bool(os.environ.get("SARVAM_API_KEY")),
    }


@app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(require_key)])
def embed(req: EmbedRequest) -> EmbedResponse:
    if len(req.texts) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"max {MAX_BATCH} texts per request")
    embedder = embedders["embed"]
    try:
        vectors = embedder.embed(req.texts)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"embedding backend error: {e}") from e
    return EmbedResponse(
        model=embedder.label,
        dimensions=len(vectors[0]),
        embeddings=vectors,
    )


# ---------------------------------------------------------------------------
# Transcription: audio is processed and discarded, never stored.
#
# POST /transcribe  (raw audio body) -> {job_id}; the job runs in the background because
# Sarvam's batch API can take longer than a request should be held open.
# GET  /transcribe/{job_id}          -> {status, transcript?, language_code?, error?}
#
# Jobs live in memory only (single instance): a restart loses them and the client simply retries.
# ---------------------------------------------------------------------------

EXTENSIONS = {
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/aac": "aac",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/webm": "webm",
    "audio/ogg": "ogg",
}


@dataclass
class Job:
    id: str
    owner: str
    status: str = "running"  # running | done | failed
    transcript: str = ""
    language_code: str | None = None
    error: str | None = None
    created: float = 0.0


jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()


def _purge_old_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    with jobs_lock:
        for job_id in [j.id for j in jobs.values() if j.created < cutoff]:
            del jobs[job_id]


def _transcribe_with_sarvam(audio: bytes, extension: str, api_key: str) -> tuple[str, str | None]:
    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / f"audio.{extension}"
        audio_path.write_bytes(audio)

        client = SarvamAI(api_subscription_key=api_key)
        job = client.speech_to_text_job.create_job(
            model="saaras:v4", mode="transcribe", language_code="unknown"
        )
        job.upload_files(file_paths=[str(audio_path)])
        job.start()
        job.wait_until_complete()

        results = job.get_file_results()
        if results["failed"]:
            raise RuntimeError(results["failed"][0]["error_message"])

        out_dir = Path(tmp) / "out"
        job.download_outputs(output_dir=str(out_dir))
        files = sorted(out_dir.glob("*.json"))
        if not files:
            raise RuntimeError("no transcript output")
        data = json.loads(files[0].read_text(encoding="utf-8"))
        return data.get("transcript", ""), data.get("language_code")


def _run_job(job: Job, audio: bytes, extension: str, api_key: str) -> None:
    try:
        job.transcript, job.language_code = _transcribe_with_sarvam(audio, extension, api_key)
        job.status = "done"
    except Exception as e:  # noqa: BLE001 - any provider failure is reported to the client, not raised
        job.error = str(e)[:300]
        job.status = "failed"


@app.post("/transcribe", dependencies=[Depends(require_key)])
async def start_transcription(
    request: Request,
    x_user_id: str = Header(default=""),
    content_type: str = Header(default=""),
) -> dict:
    api_key = os.environ.get("SARVAM_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="SARVAM_API_KEY not configured")
    if not x_user_id:
        raise HTTPException(status_code=400, detail="missing X-User-Id")
    extension = EXTENSIONS.get(content_type.split(";")[0].strip().lower())
    if not extension:
        raise HTTPException(status_code=415, detail="unsupported audio type")

    audio = await request.body()
    if len(audio) < MIN_AUDIO_BYTES:
        raise HTTPException(status_code=422, detail="empty_recording")
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="recording too large")

    _purge_old_jobs()
    with jobs_lock:
        if sum(1 for j in jobs.values() if j.status == "running") >= MAX_RUNNING_JOBS:
            raise HTTPException(status_code=429, detail="too many transcriptions in progress")
        job = Job(id=uuid.uuid4().hex, owner=x_user_id, created=time.time())
        jobs[job.id] = job

    threading.Thread(target=_run_job, args=(job, audio, extension, api_key), daemon=True).start()
    return {"job_id": job.id}


@app.get("/transcribe/{job_id}", dependencies=[Depends(require_key)])
def get_transcription(job_id: str, x_user_id: str = Header(default="")) -> dict:
    job = jobs.get(job_id)
    # A job id from another user is indistinguishable from a missing one.
    if not job or not secrets.compare_digest(job.owner, x_user_id):
        raise HTTPException(status_code=404, detail="unknown job")
    if job.status == "running":
        return {"status": "running"}
    if job.status == "failed":
        return {"status": "failed", "error": job.error}
    return {"status": "done", "transcript": job.transcript, "language_code": job.language_code}
