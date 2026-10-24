# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Slow or flaky networks: default is a 15s read timeout and 5 retries.
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    # Where sentence-transformers keeps its model, so a volume can persist the ~2 GB download.
    HF_HOME=/models

WORKDIR /srv

# requirements.txt: Ollama or a hosted embeddings API (small image, the default).
# requirements-local.txt: embeds in this process with sentence-transformers (pulls in torch).
ARG REQUIREMENTS=requirements.txt
# CPU-only torch wheels; skipping CUDA saves several GB when REQUIREMENTS=requirements-local.txt.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-local.txt ./
RUN pip install --extra-index-url "$TORCH_INDEX" -r "$REQUIREMENTS"

COPY app ./app

RUN useradd --system --uid 10001 --home-dir /srv app \
    && mkdir -p /models \
    && chown -R app:app /srv /models
USER app

EXPOSE 8001

# python's own urllib: the slim image has no curl.
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
