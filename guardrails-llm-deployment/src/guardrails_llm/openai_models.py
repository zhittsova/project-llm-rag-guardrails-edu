from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .corpus import Chunk
from .guard_classifier import GuardClassification
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


class OpenAIGuardClassifier:
    def __init__(self, config: OpenAIModelConfig, *, client: Any | None = None) -> None:
        ensure_remote_models_allowed(config)
        ensure_openai_api_key(config)
        self.model_name = config.classifier_model
        self._client = client or OpenAI()

    def classify(self, text: str) -> GuardClassification:
        response = self._client.responses.create(
            model=self.model_name,
            input=_guard_classifier_prompt(text),
        )
        payload = _json_response(response)
        label = str(payload.get("label", "unsafe_request"))
        confidence = _float_in_range(payload.get("confidence", 1.0))
        explanation = str(payload.get("explanation", "")).strip()
        allowed_labels = {
            "safe",
            "prompt_injection",
            "pii",
            "academic_integrity",
            "unsafe_request",
            "unsupported",
        }
        if label not in allowed_labels:
            label = "unsafe_request"
        return GuardClassification(label=label, confidence=confidence, explanation=explanation)


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


def _guard_classifier_prompt(text: str) -> str:
    return (
        "Classify the user request for a course RAG assistant. Return only JSON with "
        "keys: label, confidence, explanation. Allowed labels: safe, prompt_injection, "
        "pii, academic_integrity, unsafe_request, unsupported.\n\n"
        f"User request:\n{text}"
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


def _json_response(response: Any) -> dict[str, object]:
    text = _response_text(response)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("OpenAI classifier response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("OpenAI classifier response must be a JSON object")
    return payload


def _float_in_range(value: object) -> float:
    if not isinstance(value, int | float):
        return 1.0
    return max(0.0, min(1.0, float(value)))
