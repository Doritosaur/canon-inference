# canon-inference

FastAPI service used by `../canonevent-backend`: embeddings (bge-m3) and speech-to-text (Sarvam). The Edge
Functions call it with the `X-Inference-Key` header; only `/health` is open.

## Configure

```sh
cp .env.dev.example .env     # set INFERENCE_API_KEY (same value as in canonevent-backend/.env) and SARVAM_API_KEY
```

## Run locally

Ollama on your machine (`ollama pull bge-m3`):

```sh
docker compose up -d --build                    # http://localhost:8001 (INFERENCE_PORT to change)
OLLAMA_URL=http://ollama:11434 docker compose --profile ollama up -d --build   # Ollama in Docker too (pulls OLLAMA_PULL_MODELS)
```

Or without Docker:

```sh
python -m venv .venv && source .venv/Scripts/activate     # .venv\Scripts\activate in PowerShell
pip install -r requirements.txt            # requirements-local.txt only for EMBED_BACKEND=local (torch)
uvicorn app.main:app --port 8001 --env-file .env
```

`GET /health` reports the embedding model and whether the keys are loaded (never their values).
The backend's `INFERENCE_URL` for local dev is `http://host.docker.internal:8001`.

## API

All routes except `/health` need the `X-Inference-Key` header.

| Route | Purpose |
| --- | --- |
| `GET /health` | Status, embedding model label, whether the keys are set. Open. |
| `POST /embed` | `{"texts": [...]}` (up to `EMBED_MAX_BATCH`) → `{model, dimensions, embeddings}`, 1024-d L2-normalized. |
| `POST /transcribe` | Raw audio body (`audio/mp4`, `x-m4a`, `aac`, `mpeg`, `wav`, `webm`, `ogg`; 2 KB to 25 MB) plus `X-User-Id` → `{job_id}`. Needs `SARVAM_API_KEY`. |
| `GET /transcribe/{job_id}` | With the same `X-User-Id` → `{status: running\|done\|failed, transcript?, language_code?, error?}`. |

Audio is never stored. Jobs are held in memory for an hour (at most 4 running), so a restart loses them and the
client retries.

## Deploy

The Edge Functions run in Supabase's cloud, so this must be reachable over public HTTPS.

**Your own Docker host** (a VM with Docker and `cloudflared`):

```sh
cp .env.prod.example .env    # set the keys and the Cloudflare account/token
docker compose -f compose.yaml -f compose.prod.yaml up -d --build
curl http://127.0.0.1:8001/health
```

Runs the service (read-only filesystem, no capabilities) published on `127.0.0.1:8001` only. Put a `cloudflared`
tunnel (or any reverse proxy) in front for HTTPS: `cloudflared tunnel --url http://localhost:8001`. Embeddings come from
Cloudflare Workers AI (`@cf/baai/bge-m3`), so no Ollama container runs (`COMPOSE_PROFILES` stays empty). Update with
`git pull` and the same command.

**A managed container host** (Fly.io, Railway, Render, Cloud Run): deploy this directory's `Dockerfile`, set
`EMBED_BACKEND=api` with `EMBED_API_*`, plus `INFERENCE_API_KEY` and `SARVAM_API_KEY` as the platform's env vars. HTTPS
comes with the platform, so skip `compose.prod.yaml`.

Then set `INFERENCE_URL` (the public URL) and `INFERENCE_API_KEY` in the backend's `.env.prod` and run
`supabase secrets set --env-file .env.prod` there.

## Embedding backends

`EMBED_BACKEND`: `ollama` (default, local dev), `api` (hosted OpenAI-compatible `/embeddings`; production uses
Cloudflare Workers AI, see `.env.prod.example`; clear `COMPOSE_PROFILES`), or `local` (in-process; set `INFERENCE_REQUIREMENTS=requirements-local.txt` and `INFERENCE_MEMORY=4g`). Vectors from
different backends differ slightly, so after switching, re-embed with `POST /memories/reprocess` (body
`{"extract": false}`) on the backend.
