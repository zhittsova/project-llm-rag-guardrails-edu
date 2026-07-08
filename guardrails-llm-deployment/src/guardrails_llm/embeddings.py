from __future__ import annotations

import math
from hashlib import blake2b
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
) -> TextEmbedder:
    resolved_model = resolve_embedding_model(provider, model)
    if provider == "hashing":
        return HashingEmbedder()
    if provider == "openai":
        from .openai_models import OpenAIEmbeddingModel

        return OpenAIEmbeddingModel(
            OpenAIModelConfig(
                embedding_model=resolved_model,
                allow_remote_models=allow_remote_models,
                env_file=env_file,
            )
        )
    raise ValueError("embedding_provider must be 'hashing' or 'openai'")
