from __future__ import annotations

from typing import Any

from openai import OpenAI

from .model_config import OpenAIModelConfig, ensure_openai_api_key, ensure_remote_models_allowed


class OpenAIEmbeddingModel:
    def __init__(self, config: OpenAIModelConfig, *, client: Any | None = None) -> None:
        ensure_remote_models_allowed(config)
        ensure_openai_api_key(config)
        self.model_name = config.embedding_model
        self._client = client or OpenAI()

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model_name, input=texts)
        return [list(item.embedding) for item in response.data]
