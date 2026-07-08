from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Protocol

from .answering import AnswerGenerator
from .baseline_pipeline import BaselineRagAssistant, build_baseline_assistant
from .corpus import Chunk, chunk_documents, load_documents
from .guard_classifier import GuardClassifier, should_use_model_classifier
from .guardrail_policy import GuardrailPolicy
from .guards import input_guard, make_integrity_safe, output_guard, sanitize_untrusted_context
from .retrieval import LexicalRetriever


@dataclass(frozen=True)
class AssistantResponse:
    answer: str
    citations: list[str]
    guard_triggers: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    retrieved_chunks: list[str] = field(default_factory=list)


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        course_id: str | None = None,
        allowed_visibility: set[str] | None = None,
        top_k: int = 3,
    ) -> list[tuple[Chunk, float]]:
        ...


class LearningAssistant:
    def __init__(
        self,
        retriever: Retriever,
        *,
        mode: str = "guardrailed",
        course_id: str = "guardrails-101",
        retriever_backend: str = "lexical",
        guardrail_policy: GuardrailPolicy | None = None,
        answer_generator: AnswerGenerator | None = None,
        guard_classifier: GuardClassifier | None = None,
    ) -> None:
        if mode not in {"baseline", "guardrailed"}:
            raise ValueError("mode must be 'baseline' or 'guardrailed'")
        self._retriever = retriever
        self._mode = mode
        self._course_id = course_id
        self._retriever_backend = retriever_backend
        self._guardrail_policy = guardrail_policy or GuardrailPolicy.default()
        self._answer_generator = answer_generator
        self._guard_classifier = guard_classifier

    def answer(self, question: str) -> AssistantResponse:
        started_at = perf_counter()
        triggers: list[str] = []

        # Guardrailed mode сначала проверяет сам пользовательский вопрос.
        # Baseline RAG намеренно пропускает этот блок, чтобы показать, как
        # обычный RAG ведет себя без prompt-injection/PII/integrity защит.
        if self._mode == "guardrailed":
            input_result = input_guard(question, self._guardrail_policy)
            triggers.extend(input_result.triggers)
            if not input_result.allowed:
                return self._response(input_result.message or "Request blocked.", [], triggers, started_at, [])
            if should_use_model_classifier(question, self._guardrail_policy, triggers) and self._guard_classifier:
                classification = self._guard_classifier.classify(question)
                if classification.label != "safe" and classification.confidence >= 0.65:
                    triggers.append(classification.label)
                    if classification.label == "unsupported":
                        triggers.append("ungrounded")
                        return self._response(
                            self._guardrail_policy.ungrounded_message,
                            [],
                            triggers,
                            started_at,
                            [],
                        )
                    if classification.label in self._guardrail_policy.blocking_triggers:
                        return self._response(
                            self._guardrail_policy.input_block_message,
                            [],
                            triggers,
                            started_at,
                            [],
                        )

        # Retrieval общий для baseline и guardrailed режимов, но фильтры разные:
        # baseline ищет по всему индексу, guardrailed ограничивает поиск текущим
        # курсом и только public-документами.
        visibility = set(self._guardrail_policy.allowed_visibility) if self._mode == "guardrailed" else None
        retrieved = self._retriever.search(
            question,
            course_id=self._course_id if self._mode == "guardrailed" else None,
            allowed_visibility=visibility,
        )
        if self._mode == "guardrailed":
            # Retrieved context считается недоверенным: даже текст из corpus
            # может содержать indirect prompt injection.
            retrieved = [(sanitize_chunk(chunk, self._guardrail_policy), score) for chunk, score in retrieved]

        if "academic_integrity" in triggers:
            # Для cheating-запросов guardrailed режим не дает готовое решение,
            # а достает policy chunk и отвечает в формате помощи/скэффолдинга.
            retrieved = self._retriever.search(
                "academic integrity graded work complete submissions hints similar examples",
                course_id=self._course_id,
                allowed_visibility=visibility,
            )
            answer = make_integrity_safe(question, self._guardrail_policy)
            citations = [citation_for(chunk) for chunk, _score in retrieved[:1]]
        else:
            # Default answer generation is local/extractive. Optional remote
            # generation is gated by --allow-remote-models in the CLI.
            retrieved_chunks = [chunk for chunk, _score in retrieved]
            if self._answer_generator:
                answer = self._answer_generator.generate(question, retrieved_chunks)
            else:
                answer = synthesize_answer(question, retrieved_chunks)
            citations = [citation_for(chunk) for chunk, _score in retrieved]

        # Output guard проверяет уже готовый ответ. Baseline снова пропускает
        # этот этап, поэтому может вернуть private data или ungrounded answer.
        if self._mode == "guardrailed":
            output_result = output_guard(answer, citations, triggers, self._guardrail_policy)
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


def build_assistant(
    corpus_path,
    *,
    mode: str = "guardrailed",
    retriever_backend: str = "lexical",
    index_dir: Path | None = None,
    course_id: str = "guardrails-101",
    guardrail_policy: GuardrailPolicy | None = None,
    embedding_provider: str = "hashing",
    embedding_model: str | None = None,
    allow_remote_models: bool = False,
    env_file: Path | None = None,
    generator: str = "extractive",
    answer_model: str | None = None,
    guard_classifier: str = "none",
    classifier_model: str | None = None,
) -> BaselineRagAssistant | LearningAssistant:
    answer_generator = _build_answer_generator(
        generator,
        answer_model=answer_model,
        allow_remote_models=allow_remote_models,
        env_file=env_file,
    )
    if mode == "baseline":
        return build_baseline_assistant(
            corpus_path,
            retriever_backend=retriever_backend,
            index_dir=index_dir,
            course_id=course_id,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            allow_remote_models=allow_remote_models,
            env_file=env_file,
            answer_generator=answer_generator,
        )

    # Ниже строится guardrailed assistant. Baseline уже ушел в отдельный
    # baseline_pipeline.py, чтобы его можно было читать без guardrail веток.
    classifier = _build_guard_classifier(
        guard_classifier,
        classifier_model=classifier_model,
        allow_remote_models=allow_remote_models,
        env_file=env_file,
    )
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
        )
    else:
        raise ValueError("retriever_backend must be 'lexical', 'langchain', or 'vector'")
    return LearningAssistant(
        retriever,
        mode=mode,
        course_id=course_id,
        retriever_backend=retriever_backend,
        guardrail_policy=guardrail_policy,
        answer_generator=answer_generator,
        guard_classifier=classifier,
    )


def _build_answer_generator(
    generator: str,
    *,
    answer_model: str | None = None,
    allow_remote_models: bool = False,
    env_file: Path | None = None,
) -> AnswerGenerator | None:
    if generator == "extractive":
        return None
    if generator == "openai":
        from .model_config import DEFAULT_OPENAI_ANSWER_MODEL, OpenAIModelConfig
        from .openai_models import OpenAIAnswerGenerator

        return OpenAIAnswerGenerator(
            OpenAIModelConfig(
                answer_model=answer_model or DEFAULT_OPENAI_ANSWER_MODEL,
                allow_remote_models=allow_remote_models,
                env_file=env_file,
            )
        )
    raise ValueError("generator must be 'extractive' or 'openai'")


def _build_guard_classifier(
    guard_classifier: str,
    *,
    classifier_model: str | None = None,
    allow_remote_models: bool = False,
    env_file: Path | None = None,
) -> GuardClassifier | None:
    if guard_classifier == "none":
        return None
    if guard_classifier == "openai":
        from .model_config import DEFAULT_OPENAI_CLASSIFIER_MODEL, OpenAIModelConfig
        from .openai_models import OpenAIGuardClassifier

        return OpenAIGuardClassifier(
            OpenAIModelConfig(
                classifier_model=classifier_model or DEFAULT_OPENAI_CLASSIFIER_MODEL,
                allow_remote_models=allow_remote_models,
                env_file=env_file,
            )
        )
    raise ValueError("guard_classifier must be 'none' or 'openai'")


def synthesize_answer(question: str, chunks: list[Chunk]) -> str:
    # Если retrieval ничего не нашел, baseline abstains простой фразой. В
    # guardrailed режиме output_guard превращает это в более строгий отказ.
    if not chunks:
        return "I do not know based on the available course material."

    # Для reproducible demo берем первые найденные chunks и возвращаем первые
    # предложения как evidence-based ответ. Это проще, чем LLM, но достаточно,
    # чтобы тестировать retrieval, citations и guardrails.
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


def sanitize_chunk(chunk: Chunk, policy: GuardrailPolicy) -> Chunk:
    sanitized = sanitize_untrusted_context(chunk.text, policy)
    return replace(chunk, text=sanitized)


def _metadata_text(chunk: Chunk, key: str) -> str:
    value = chunk.metadata.get(key)
    if value is None or isinstance(value, list):
        return ""
    return str(value).strip()
