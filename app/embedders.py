"""Embedding backends, selected by EMBED_BACKEND:

  ollama  Ollama's /api/embed (default; local dev).
  local   sentence-transformers in this process (self-deployed service).
  api     Any OpenAI-compatible /embeddings endpoint (hosted bge-m3).

All return L2-normalized 1024-d vectors for bge-m3. Vectors from different
backends differ slightly, so don't mix them in one database: the returned
`model` label is stored per chunk in memory_chunks.embedding_model.
"""
import math
import os
from typing import Protocol

import httpx


class Embedder(Protocol):
    label: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _normalize(vectors: list[list[float]]) -> list[list[float]]:
    out = []
    for v in vectors:
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


class OllamaEmbedder:
    def __init__(self) -> None:
        self.url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.model = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
        self.label = f"ollama/{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        r = httpx.post(
            f"{self.url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        r.raise_for_status()
        return _normalize(r.json()["embeddings"])


class ApiEmbedder:
    def __init__(self) -> None:
        self.url = os.environ["EMBED_API_URL"].rstrip("/")
        self.key = os.environ.get("EMBED_API_KEY", "")
        self.model = os.environ["EMBED_API_MODEL"]
        self.label = f"api/{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        r = httpx.post(
            f"{self.url}/embeddings",
            headers={"Authorization": f"Bearer {self.key}"},
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return _normalize([d["embedding"] for d in data])


class LocalEmbedder:
    def __init__(self) -> None:
        # Imported here so ollama/api deployments don't need torch installed.
        from sentence_transformers import SentenceTransformer

        self.label = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
        self.model = SentenceTransformer(self.label)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        ).tolist()


def load_embedder() -> Embedder:
    backend = os.environ.get("EMBED_BACKEND", "ollama")
    if backend == "ollama":
        return OllamaEmbedder()
    if backend == "api":
        return ApiEmbedder()
    if backend == "local":
        return LocalEmbedder()
    raise RuntimeError(f"unknown EMBED_BACKEND {backend!r} (ollama | local | api)")
