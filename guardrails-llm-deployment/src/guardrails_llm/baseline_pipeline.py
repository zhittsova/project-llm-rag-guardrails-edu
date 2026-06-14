from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Protocol

from .corpus import Chunk, chunk_documents, load_documents
from .retrieval import LexicalRetriever


@dataclass(frozen=True)
class BaselineRagResponse:
    answer: str
    citations: list[str]
    guard_triggers: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    retrieved_chunks: list[str] = field(default_factory=list)


class BaselineRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        course_id: str | None = None,
        allowed_visibility: set[str] | None = None,
        top_k: int = 3,
    ) -> list[tuple[Chunk, float]]:
        ...


class BaselineRagAssistant:
    """Минимальный baseline RAG без guardrails.

    Этот класс специально оставлен простым для Workshop 2 comparison:
    тот же retrieval/citation flow, но без prompt-injection detection,
    PII filter, visibility filter, context sanitization и output guard.
    """

    def __init__(
        self,
        retriever: BaselineRetriever,
        *,
        course_id: str = "guardrails-101",
        retriever_backend: str = "lexical",
    ) -> None:
        self._retriever = retriever
        self._course_id = course_id
        self._retriever_backend = retriever_backend

    def answer(self, question: str) -> BaselineRagResponse:
        started_at = perf_counter()

        # Baseline retrieval intentionally has no course_id/public filters.
        # This is useful for failure analysis: it can retrieve private chunks
        # or injected content, showing why guardrails are needed.
        retrieved = self._retriever.search(question)

        # Baseline generation is extractive and local: no LLM call here.
        # It simply turns retrieved chunks into a short evidence-based answer.
        answer = synthesize_baseline_answer([chunk for chunk, _score in retrieved])
        citations = [citation_for(chunk) for chunk, _score in retrieved]

        return BaselineRagResponse(
            answer=answer,
            citations=citations,
            guard_triggers=[],
            latency_ms=(perf_counter() - started_at) * 1000,
            retrieved_chunks=[chunk.chunk_id for chunk, _score in retrieved],
        )


def build_baseline_assistant(
    corpus_path,
    *,
    retriever_backend: str = "lexical",
    index_dir: Path | None = None,
    course_id: str = "guardrails-101",
) -> BaselineRagAssistant:
    if retriever_backend == "lexical":
        documents = load_documents(corpus_path)
        retriever = LexicalRetriever(chunk_documents(documents))
    elif retriever_backend == "langchain":
        from .langchain_rag import LangChainLexicalRetriever

        documents = load_documents(corpus_path)
        retriever = LangChainLexicalRetriever.from_documents(documents)
    elif retriever_backend == "vector":
        from .vector import VectorRetriever, default_index_path

        retriever = VectorRetriever(index_dir or default_index_path())
    else:
        raise ValueError("retriever_backend must be 'lexical', 'langchain', or 'vector'")
    return BaselineRagAssistant(retriever, course_id=course_id, retriever_backend=retriever_backend)


def synthesize_baseline_answer(chunks: list[Chunk]) -> str:
    if not chunks:
        return "I do not know based on the available course material."

    evidence = " ".join(chunk.text for chunk in chunks[:2])
    sentences = [sentence.strip() for sentence in evidence.split(".") if sentence.strip()]
    if not sentences:
        return evidence[:500]
    selected = sentences[:3]
    return " ".join(sentence + "." for sentence in selected)


def citation_for(chunk: Chunk) -> str:
    details = [chunk.doc_id]
    section = _metadata_text(chunk, "section")
    slide = _metadata_text(chunk, "slide")
    page = _metadata_text(chunk, "page")
    if section:
        details.append(section)
    if slide:
        details.append(f"slide {slide}")
    if page:
        details.append(f"page {page}")
    return f"{chunk.title} ({', '.join(details)})"


def _metadata_text(chunk: Chunk, key: str) -> str:
    value = chunk.metadata.get(key)
    if value is None or isinstance(value, list):
        return ""
    return str(value).strip()
