from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter

from .corpus import Chunk, chunk_documents, load_documents
from .guards import input_guard, make_integrity_safe, output_guard, sanitize_untrusted_context
from .retrieval import LexicalRetriever


@dataclass(frozen=True)
class AssistantResponse:
    answer: str
    citations: list[str]
    guard_triggers: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    retrieved_chunks: list[str] = field(default_factory=list)


class LearningAssistant:
    def __init__(
        self,
        retriever: LexicalRetriever,
        *,
        mode: str = "guardrailed",
        course_id: str = "guardrails-101",
        retriever_backend: str = "lexical",
    ) -> None:
        if mode not in {"baseline", "guardrailed"}:
            raise ValueError("mode must be 'baseline' or 'guardrailed'")
        self._retriever = retriever
        self._mode = mode
        self._course_id = course_id
        self._retriever_backend = retriever_backend

    def answer(self, question: str) -> AssistantResponse:
        started_at = perf_counter()
        triggers: list[str] = []

        if self._mode == "guardrailed":
            input_result = input_guard(question)
            triggers.extend(input_result.triggers)
            if not input_result.allowed:
                return self._response(input_result.message or "Request blocked.", [], triggers, started_at, [])

        visibility = {"public"} if self._mode == "guardrailed" else None
        retrieved = self._retriever.search(
            question,
            course_id=self._course_id if self._mode == "guardrailed" else None,
            allowed_visibility=visibility,
        )
        if self._mode == "guardrailed":
            retrieved = [(sanitize_chunk(chunk), score) for chunk, score in retrieved]

        if "academic_integrity" in triggers:
            retrieved = self._retriever.search(
                "academic integrity graded work complete submissions hints similar examples",
                course_id=self._course_id,
                allowed_visibility=visibility,
            )
            answer = make_integrity_safe(question)
            citations = [citation_for(chunk) for chunk, _score in retrieved[:1]]
        else:
            answer = synthesize_answer(question, [chunk for chunk, _score in retrieved])
            citations = [citation_for(chunk) for chunk, _score in retrieved]

        if self._mode == "guardrailed":
            output_result = output_guard(answer, citations, triggers)
            triggers.extend(output_result.triggers)
            if not output_result.allowed:
                return self._response(output_result.message or "Answer blocked.", [], triggers, started_at, [])

        return self._response(answer, citations, triggers, started_at, [chunk.chunk_id for chunk, _score in retrieved])

    def _response(
        self,
        answer: str,
        citations: list[str],
        triggers: list[str],
        started_at: float,
        retrieved_chunks: list[str],
    ) -> AssistantResponse:
        return AssistantResponse(
            answer=answer,
            citations=citations,
            guard_triggers=sorted(set(triggers)),
            latency_ms=(perf_counter() - started_at) * 1000,
            retrieved_chunks=retrieved_chunks,
        )


def build_assistant(corpus_path, *, mode: str = "guardrailed", retriever_backend: str = "lexical") -> LearningAssistant:
    documents = load_documents(corpus_path)
    if retriever_backend == "lexical":
        retriever = LexicalRetriever(chunk_documents(documents))
    elif retriever_backend == "langchain":
        from .langchain_rag import LangChainLexicalRetriever

        retriever = LangChainLexicalRetriever.from_documents(documents)
    else:
        raise ValueError("retriever_backend must be 'lexical' or 'langchain'")
    return LearningAssistant(retriever, mode=mode, retriever_backend=retriever_backend)


def synthesize_answer(question: str, chunks: list[Chunk]) -> str:
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


def sanitize_chunk(chunk: Chunk) -> Chunk:
    sanitized = sanitize_untrusted_context(chunk.text)
    return replace(chunk, text=sanitized)


def _metadata_text(chunk: Chunk, key: str) -> str:
    value = chunk.metadata.get(key)
    if value is None or isinstance(value, list):
        return ""
    return str(value).strip()
