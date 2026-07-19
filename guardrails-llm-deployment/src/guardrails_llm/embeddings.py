from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from hashlib import blake2b, sha256
from pathlib import Path
from typing import Protocol

from .model_config import DEFAULT_OPENAI_EMBEDDING_MODEL, OpenAIModelConfig
from .retrieval import tokenize


HASHING_EMBEDDING_MODEL = "hashing-blake2b-384"


class TextEmbedder(Protocol):
    model_name: str

    def embed(self, text: str) -> list[float]:
        ...

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        ...


class CachedEmbedder:
    def __init__(self, delegate: TextEmbedder) -> None:
        self.model_name = delegate.model_name
        self._delegate = delegate
        self._cache: dict[str, list[float]] = {}

    @property
    def cached_texts(self) -> int:
        return len(self._cache)

    @property
    def api_call_count(self) -> int | None:
        return getattr(self._delegate, "api_call_count", None)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        if missing:
            vectors = self._delegate.embed_many(missing)
            if len(vectors) != len(missing):
                raise ValueError(
                    f"embedding provider returned {len(vectors)} vectors for {len(missing)} texts"
                )
            self._cache.update(zip(missing, vectors, strict=True))
        return [self._cache[text] for text in texts]


class PersistentCachedEmbedder:
    def __init__(
        self,
        delegate: TextEmbedder,
        cache_path: Path,
        *,
        batch_size: int = 128,
        read_only: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("embedding cache batch_size must be greater than zero")
        self.model_name = delegate.model_name
        self._delegate = delegate
        self._cache_path = cache_path
        self._batch_size = batch_size
        self._read_only = read_only
        self._cache: dict[str, list[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._load()

    @property
    def cached_texts(self) -> int:
        return len(self._cache)

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    @property
    def api_call_count(self) -> int | None:
        return getattr(self._delegate, "api_call_count", None)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        unique = list(dict.fromkeys(texts))
        keys = {text: self._key(text) for text in unique}
        missing = [text for text in unique if keys[text] not in self._cache]
        self._cache_hits += sum(keys[text] in self._cache for text in texts)
        self._cache_misses += len(missing)
        if self._read_only and missing:
            raise ValueError(
                f"read-only embedding cache is missing {len(missing)} text(s)"
            )

        for start in range(0, len(missing), self._batch_size):
            batch = missing[start : start + self._batch_size]
            vectors = self._delegate.embed_many(batch)
            if len(vectors) != len(batch):
                raise ValueError(
                    f"embedding provider returned {len(vectors)} vectors for "
                    f"{len(batch)} texts"
                )
            records = []
            for text, vector in zip(batch, vectors, strict=True):
                validated = _validate_cached_vector(vector)
                key = keys[text]
                self._cache[key] = validated
                records.append(
                    {
                        "schema_version": 1,
                        "model": self.model_name,
                        "text_sha256": key,
                        "dimensions": len(validated),
                        "vector": validated,
                    }
                )
            self._append(records)
        return [self._cache[self._key(text)] for text in texts]

    def _key(self, text: str) -> str:
        material = f"v1\0{self.model_name}\0{text}".encode("utf-8")
        return sha256(material).hexdigest()

    def _load(self) -> None:
        if not self._cache_path.exists():
            return
        for line_number, line in enumerate(
            self._cache_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                record = json.loads(line)
                if (
                    not isinstance(record, dict)
                    or record.get("schema_version") != 1
                    or not isinstance(record.get("model"), str)
                    or not isinstance(record.get("text_sha256"), str)
                    or len(record["text_sha256"]) != 64
                    or record.get("dimensions") != len(record.get("vector", []))
                ):
                    raise ValueError
                vector = _validate_cached_vector(record["vector"])
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid embedding cache record at "
                    f"{self._cache_path}:{line_number}"
                ) from exc
            if record["model"] == self.model_name:
                self._cache[record["text_sha256"]] = vector

    def _append(self, records: list[dict[str, object]]) -> None:
        if not records:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _validate_cached_vector(vector: object) -> list[float]:
    if not isinstance(vector, list) or not vector:
        raise ValueError("embedding vector must be a non-empty list")
    if any(
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in vector
    ):
        raise ValueError("embedding vector must contain only finite numbers")
    return [float(value) for value in vector]


class HashingEmbedder:
    # Local deterministic embedding function for demos: no API keys, downloads,
    # or randomness. This is not a production semantic embedding model, but it
    # creates numeric vectors that can power vector search and similarity guards.
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.model_name = HASHING_EMBEDDING_MODEL

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensions")
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


def resolve_embedding_model(provider: str, model: str | None = None) -> str:
    if provider == "hashing":
        return model or HASHING_EMBEDDING_MODEL
    if provider == "openai":
        return model or DEFAULT_OPENAI_EMBEDDING_MODEL
    raise ValueError("embedding_provider must be 'hashing' or 'openai'")


def create_embedder(
    provider: str,
    *,
    model: str | None = None,
    allow_remote_models: bool = False,
    env_file: Path | None = None,
    cache_path: Path | None = None,
    cache_read_only: bool = False,
    model_config: OpenAIModelConfig | None = None,
) -> TextEmbedder:
    resolved_model = resolve_embedding_model(provider, model)
    if provider == "hashing":
        embedder: TextEmbedder = HashingEmbedder()
    elif provider == "openai":
        from .openai_models import OpenAIEmbeddingModel

        config = (
            replace(model_config, embedding_model=resolved_model)
            if model_config is not None
            else OpenAIModelConfig(
                embedding_model=resolved_model,
                allow_remote_models=allow_remote_models,
                env_file=env_file,
            )
        )
        embedder = OpenAIEmbeddingModel(
            config
        )
    else:
        raise ValueError("embedding_provider must be 'hashing' or 'openai'")
    if cache_path is not None:
        return PersistentCachedEmbedder(
            embedder,
            cache_path,
            read_only=cache_read_only,
        )
    return embedder
