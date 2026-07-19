from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Protocol

from .answering import AnswerGenerator, unpack_generated_answer
from .corpus import Chunk, chunk_documents, load_documents
from .dispositions import ResponseDisposition
from .embeddings import TextEmbedder
from .retrieval import LexicalRetriever


@dataclass(frozen=True)
class BaselineRagResponse:
    answer: str
    citations: list[str]
    cited_doc_ids: list[str]
    disposition: ResponseDisposition
    guard_triggers: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    retrieved_chunks: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieval_scores: dict[str, float] = field(default_factory=dict)
    retrieved_evidence: list[dict[str, object]] = field(default_factory=list)


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
    """Minimal baseline RAG without guardrails.

    This class intentionally stays simple for the Workshop 2 comparison. It
    uses the same retrieval and citation flow without prompt-injection
    detection, PII filtering, visibility filtering, context sanitization, or
    output guards.
    """

    def __init__(
        self,
        retriever: BaselineRetriever,
        *,
        course_id: str = "guardrails-101",
        retriever_backend: str = "lexical",
        answer_generator: AnswerGenerator | None = None,
    ) -> None:
        self._retriever = retriever
        self._course_id = course_id
        self._retriever_backend = retriever_backend
        self._answer_generator = answer_generator

    def answer(self, question: str) -> BaselineRagResponse:
        started_at = perf_counter()

        # Baseline retrieval intentionally has no course_id/public filters.
        # This is useful for failure analysis: it can retrieve private chunks
        # or injected content, showing why guardrails are needed.
        retrieved = self._retriever.search(question)

        retrieved_chunks = [chunk for chunk, _score in retrieved]
        if self._answer_generator:
            answer = unpack_generated_answer(
                self._answer_generator.generate(question, retrieved_chunks)
            ).text
        else:
            # Baseline generation is extractive and local by default: no LLM call.
            answer = synthesize_baseline_answer(retrieved_chunks)
        citations = [citation_for(chunk) for chunk, _score in retrieved]

        return BaselineRagResponse(
            answer=answer,
            citations=citations,
            cited_doc_ids=[chunk.doc_id for chunk, _score in retrieved],
            disposition=(
                ResponseDisposition.ANSWER
                if citations
                else ResponseDisposition.ABSTAIN
            ),
            guard_triggers=[],
            latency_ms=(perf_counter() - started_at) * 1000,
            retrieved_chunks=[chunk.chunk_id for chunk, _score in retrieved],
            retrieved_doc_ids=[chunk.doc_id for chunk, _score in retrieved],
            retrieval_scores={
                chunk.chunk_id: round(float(score), 6)
                for chunk, score in retrieved
            },
            retrieved_evidence=[
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "score": round(float(score), 6),
                }
                for chunk, score in retrieved
            ],
        )


def build_baseline_assistant(
    corpus_path,
    *,
    retriever_backend: str = "lexical",
    index_dir: Path | None = None,
    course_id: str = "guardrails-101",
    embedding_provider: str = "hashing",
    embedding_model: str | None = None,
    allow_remote_models: bool = False,
    env_file: Path | None = None,
    embedding_cache_path: Path | None = None,
    answer_generator: AnswerGenerator | None = None,
    retrieval_embedder: TextEmbedder | None = None,
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

        retriever = VectorRetriever(
            index_dir or default_index_path(),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            allow_remote_models=allow_remote_models,
            env_file=env_file,
            embedding_cache_path=embedding_cache_path,
            embedder=retrieval_embedder,
            corpus_path=Path(corpus_path),
        )
    else:
        raise ValueError("retriever_backend must be 'lexical', 'langchain', or 'vector'")
    return BaselineRagAssistant(
        retriever,
        course_id=course_id,
        retriever_backend=retriever_backend,
        answer_generator=answer_generator,
    )


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
