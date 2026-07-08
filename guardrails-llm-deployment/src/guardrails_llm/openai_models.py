from __future__ import annotations

from typing import Any

from openai import OpenAI

from .corpus import Chunk
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


class OpenAIAnswerGenerator:
    def __init__(self, config: OpenAIModelConfig, *, client: Any | None = None) -> None:
        ensure_remote_models_allowed(config)
        ensure_openai_api_key(config)
        self.model_name = config.answer_model
        self._client = client or OpenAI()

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return "I do not know based on the available course material."
        response = self._client.responses.create(
            model=self.model_name,
            input=_answer_prompt(question, chunks),
        )
        return _response_text(response)


def _answer_prompt(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(
        f"[{chunk.chunk_id}] {chunk.title}\n{chunk.text}"
        for chunk in chunks
    )
    return (
        "You are a course-material assistant. Answer only from the provided context. "
        "If the context does not support an answer, say you do not know based on the "
        "available course material. Keep the answer concise.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    output = getattr(response, "output", None)
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
        if texts:
            return "\n".join(texts)
    raise ValueError("OpenAI response did not contain text output")
