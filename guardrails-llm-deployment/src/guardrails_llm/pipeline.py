from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Protocol

from .answering import AnswerGenerator, unpack_generated_answer
from .baseline_pipeline import BaselineRagAssistant, build_baseline_assistant
from .corpus import Chunk, chunk_documents, load_documents
from .dispositions import ResponseDisposition
from .embeddings import TextEmbedder
from .guard_classifier import GuardClassifier, should_use_model_classifier
from .guardrail_policy import GuardrailPolicy
from .guards import input_guard, make_integrity_safe, output_guard, sanitize_untrusted_context
from .grounding import EntailmentVerifier, select_relevant_evidence
from .model_config import OpenAIModelConfig
from .retrieval import LexicalRetriever
from .retrieval_routing import route_retrieval_query


@dataclass(frozen=True)
class AssistantResponse:
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
    supporting_chunks: list[str] = field(default_factory=list)
    grounding_supported: bool | None = None
    grounding_confidence: float | None = None
    grounding_error: str | None = None
    unsupported_claims: list[str] = field(default_factory=list)


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
        classifier_strategy: str = "ambiguous",
        retrieval_top_k: int = 3,
        evidence_min_score: float | None = None,
        policy_context_min_score: float | None = None,
        entailment_verifier: EntailmentVerifier | None = None,
        entailment_min_confidence: float = 0.80,
    ) -> None:
        if mode not in {"baseline", "guardrailed"}:
            raise ValueError("mode must be 'baseline' or 'guardrailed'")
        if classifier_strategy not in {"ambiguous", "always"}:
            raise ValueError("classifier_strategy must be 'ambiguous' or 'always'")
        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be greater than zero")
        self._retriever = retriever
        self._mode = mode
        self._course_id = course_id
        self._retriever_backend = retriever_backend
        self._guardrail_policy = guardrail_policy or GuardrailPolicy.default()
        self._answer_generator = answer_generator
        self._guard_classifier = guard_classifier
        self._classifier_strategy = classifier_strategy
        self._retrieval_top_k = retrieval_top_k
        self._evidence_min_score = evidence_min_score
        self._policy_context_min_score = policy_context_min_score
        self._entailment_verifier = entailment_verifier
        self._entailment_min_confidence = entailment_min_confidence

    def answer(self, question: str) -> AssistantResponse:
        started_at = perf_counter()
        triggers: list[str] = []

        # Guardrailed mode сначала проверяет сам пользовательский вопрос.
        # Baseline RAG намеренно пропускает этот блок, чтобы показать, как
        # обычный RAG ведет себя без prompt-injection/PII/integrity защит.
        if self._mode == "guardrailed":
            input_result = input_guard(
                question,
                self._guardrail_policy,
                include_similarity=self._guard_classifier is None,
            )
            triggers.extend(input_result.triggers)
            if not input_result.allowed:
                return self._response(
                    input_result.message or "Request blocked.",
                    [],
                    ResponseDisposition.BLOCK,
                    triggers,
                    started_at,
                    [],
                )
            similarity_candidates = (
                self._guardrail_policy.input_similarity_triggers(question)
                if self._guard_classifier is not None and not triggers
                else []
            )
            should_classify = (
                not triggers
                and self._guard_classifier is not None
                and (
                    bool(similarity_candidates)
                    or
                    self._classifier_strategy == "always"
                    or should_use_model_classifier(
                        question,
                        self._guardrail_policy,
                        triggers,
                    )
                )
            )
            if should_classify:
                classification = self._guard_classifier.classify(question)
                if classification.explanation.startswith("model_classifier_error:"):
                    triggers.append(classification.explanation)
                if classification.label != "safe" and classification.confidence >= 0.65:
                    triggers.append(classification.label)
                    if classification.label == "unsupported":
                        triggers.append("ungrounded")
                        return self._response(
                            self._guardrail_policy.ungrounded_message,
                            [],
                            ResponseDisposition.ABSTAIN,
                            triggers,
                            started_at,
                            [],
                        )
                    if classification.label in self._guardrail_policy.blocking_triggers:
                        return self._response(
                            self._guardrail_policy.input_block_message,
                            [],
                            ResponseDisposition.BLOCK,
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
            top_k=self._retrieval_top_k,
        )
        if self._mode == "guardrailed":
            # Retrieved context считается недоверенным: даже текст из corpus
            # может содержать indirect prompt injection.
            retrieved = [(sanitize_chunk(chunk, self._guardrail_policy), score) for chunk, score in retrieved]

        retrieval_scores = {
            chunk.chunk_id: round(float(score), 6)
            for chunk, score in retrieved
        }
        retrieved_evidence = evidence_records(retrieved)
        supporting_chunks: list[str] = []
        cited_doc_ids: list[str] = []
        grounding_supported: bool | None = None
        grounding_confidence: float | None = None
        grounding_error: str | None = None
        unsupported_claims: list[str] = []

        if "academic_integrity" in triggers:
            # Для cheating-запросов guardrailed режим не дает готовое решение,
            # а достает policy chunk и отвечает в формате помощи/скэффолдинга.
            retrieved = self._retriever.search(
                route_retrieval_query(question, set(triggers)),
                course_id=self._course_id,
                allowed_visibility=visibility,
                top_k=self._retrieval_top_k,
            )
            retrieved = [
                (sanitize_chunk(chunk, self._guardrail_policy), score)
                for chunk, score in retrieved
            ]
            retrieval_scores = {
                chunk.chunk_id: round(float(score), 6)
                for chunk, score in retrieved
            }
            retrieved_evidence = evidence_records(retrieved)
            answer = make_integrity_safe(question, self._guardrail_policy)
            citations = [citation_for(chunk) for chunk, _score in retrieved[:1]]
            cited_doc_ids = [chunk.doc_id for chunk, _score in retrieved[:1]]
            disposition = ResponseDisposition.REDIRECT
        else:
            evidence = select_relevant_evidence(
                retrieved,
                self._evidence_min_score,
                policy_min_score=self._policy_context_min_score,
            )
            if self._evidence_min_score is not None and not evidence:
                triggers.append("ungrounded")
                return self._response(
                    self._guardrail_policy.ungrounded_message,
                    [],
                    ResponseDisposition.ABSTAIN,
                    triggers,
                    started_at,
                    [chunk.chunk_id for chunk, _score in retrieved],
                    retrieved_doc_ids=[chunk.doc_id for chunk, _score in retrieved],
                    retrieval_scores=retrieval_scores,
                    retrieved_evidence=retrieved_evidence,
                    grounding_supported=False,
                )
            retrieved = evidence
            retrieval_scores = {
                chunk.chunk_id: round(float(score), 6)
                for chunk, score in retrieved
            }
            retrieved_evidence = evidence_records(retrieved)
            # Default answer generation is local/extractive. Optional remote
            # generation is gated by --allow-remote-models in the CLI.
            retrieved_chunks = [chunk for chunk, _score in retrieved]
            if self._answer_generator:
                generated = unpack_generated_answer(
                    self._answer_generator.generate(question, retrieved_chunks)
                )
                answer = generated.text
                if generated.answerable is False:
                    triggers.append("ungrounded")
                    return self._response(
                        answer,
                        [],
                        ResponseDisposition.ABSTAIN,
                        triggers,
                        started_at,
                        [chunk.chunk_id for chunk, _score in retrieved],
                        retrieved_doc_ids=[chunk.doc_id for chunk, _score in retrieved],
                        retrieval_scores=retrieval_scores,
                        retrieved_evidence=retrieved_evidence,
                        grounding_supported=False,
                    )
            else:
                answer = synthesize_answer(question, retrieved_chunks)
            if self._entailment_verifier and retrieved_chunks:
                try:
                    verification = self._entailment_verifier.verify(
                        question,
                        answer,
                        retrieved_chunks,
                    )
                except Exception as exc:
                    verification = None
                    grounding_error = f"verification_error:{type(exc).__name__}"

                retrieved_ids = {chunk.chunk_id for chunk in retrieved_chunks}
                if verification is not None:
                    supporting_chunks = list(dict.fromkeys(verification.supporting_chunk_ids))
                    grounding_supported = verification.supported
                    grounding_confidence = verification.confidence
                    grounding_error = getattr(verification, "error", None)
                    unsupported_claims = list(verification.unsupported_claims)
                    invalid_support_ids = set(supporting_chunks) - retrieved_ids
                    verification_failed = bool(getattr(verification, "error", None))
                else:
                    invalid_support_ids = set()
                    verification_failed = True

                grounding_ok = (
                    not verification_failed
                    and grounding_supported is True
                    and grounding_confidence is not None
                    and grounding_confidence >= self._entailment_min_confidence
                    and not unsupported_claims
                    and bool(supporting_chunks)
                    and not invalid_support_ids
                )
                if not grounding_ok:
                    triggers.append("ungrounded")
                    return self._response(
                        self._guardrail_policy.ungrounded_message,
                        [],
                        ResponseDisposition.ABSTAIN,
                        triggers,
                        started_at,
                        [chunk.chunk_id for chunk, _score in retrieved],
                        retrieved_doc_ids=[chunk.doc_id for chunk, _score in retrieved],
                        retrieval_scores=retrieval_scores,
                        retrieved_evidence=retrieved_evidence,
                        supporting_chunks=supporting_chunks,
                        grounding_supported=False,
                        grounding_confidence=grounding_confidence,
                        grounding_error=grounding_error,
                        unsupported_claims=unsupported_claims,
                    )
                supporting_ids = set(supporting_chunks)
                citations = [
                    citation_for(chunk)
                    for chunk, _score in retrieved
                    if chunk.chunk_id in supporting_ids
                ]
                cited_doc_ids = [
                    chunk.doc_id
                    for chunk, _score in retrieved
                    if chunk.chunk_id in supporting_ids
                ]
            else:
                citations = [citation_for(chunk) for chunk, _score in retrieved]
                cited_doc_ids = [chunk.doc_id for chunk, _score in retrieved]
            disposition = (
                ResponseDisposition.ANSWER
                if citations
                else ResponseDisposition.ABSTAIN
            )

        # Output guard проверяет уже готовый ответ. Baseline снова пропускает
        # этот этап, поэтому может вернуть private data или ungrounded answer.
        if self._mode == "guardrailed":
            output_result = output_guard(answer, citations, triggers, self._guardrail_policy)
            triggers.extend(output_result.triggers)
            if not output_result.allowed:
                output_disposition = (
                    ResponseDisposition.ABSTAIN
                    if set(output_result.triggers) == {"ungrounded"}
                    else ResponseDisposition.BLOCK
                )
                return self._response(
                    output_result.message or "Answer blocked.",
                    [],
                    output_disposition,
                    triggers,
                    started_at,
                    [chunk.chunk_id for chunk, _score in retrieved],
                    retrieved_doc_ids=[chunk.doc_id for chunk, _score in retrieved],
                    retrieval_scores=retrieval_scores,
                    retrieved_evidence=retrieved_evidence,
                    supporting_chunks=supporting_chunks,
                    grounding_supported=grounding_supported,
                    grounding_confidence=grounding_confidence,
                    grounding_error=grounding_error,
                    unsupported_claims=unsupported_claims,
                )

        return self._response(
            answer,
            citations,
            disposition,
            triggers,
            started_at,
            [chunk.chunk_id for chunk, _score in retrieved],
            retrieved_doc_ids=[chunk.doc_id for chunk, _score in retrieved],
            cited_doc_ids=cited_doc_ids,
            retrieval_scores=retrieval_scores,
            retrieved_evidence=retrieved_evidence,
            supporting_chunks=supporting_chunks,
            grounding_supported=grounding_supported,
            grounding_confidence=grounding_confidence,
            grounding_error=grounding_error,
            unsupported_claims=unsupported_claims,
        )

    def _response(
        self,
        answer: str,
        citations: list[str],
        disposition: ResponseDisposition,
        triggers: list[str],
        started_at: float,
        retrieved_chunks: list[str],
        *,
        retrieved_doc_ids: list[str] | None = None,
        cited_doc_ids: list[str] | None = None,
        retrieval_scores: dict[str, float] | None = None,
        retrieved_evidence: list[dict[str, object]] | None = None,
        supporting_chunks: list[str] | None = None,
        grounding_supported: bool | None = None,
        grounding_confidence: float | None = None,
        grounding_error: str | None = None,
        unsupported_claims: list[str] | None = None,
    ) -> AssistantResponse:
        return AssistantResponse(
            answer=answer,
            citations=citations,
            cited_doc_ids=cited_doc_ids or [],
            disposition=disposition,
            guard_triggers=sorted(set(triggers)),
            latency_ms=(perf_counter() - started_at) * 1000,
            retrieved_chunks=retrieved_chunks,
            retrieved_doc_ids=retrieved_doc_ids or [],
            retrieval_scores=retrieval_scores or {},
            retrieved_evidence=retrieved_evidence or [],
            supporting_chunks=supporting_chunks or [],
            grounding_supported=grounding_supported,
            grounding_confidence=grounding_confidence,
            grounding_error=grounding_error,
            unsupported_claims=unsupported_claims or [],
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
    embedding_cache_path: Path | None = None,
    generator: str = "extractive",
    answer_model: str | None = None,
    guard_classifier: str = "none",
    classifier_model: str | None = None,
    classifier_strategy: str = "ambiguous",
    retrieval_top_k: int = 3,
    evidence_min_score: float | None = None,
    policy_context_top_k: int = 0,
    policy_context_min_score: float | None = None,
    entailment_verifier: str = "none",
    entailment_model: str | None = None,
    entailment_min_confidence: float = 0.80,
    retrieval_embedder: TextEmbedder | None = None,
    model_config: OpenAIModelConfig | None = None,
) -> BaselineRagAssistant | LearningAssistant:
    answer_generator = _build_answer_generator(
        generator,
        answer_model=answer_model,
        allow_remote_models=allow_remote_models,
        env_file=env_file,
        model_config=model_config,
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
            embedding_cache_path=embedding_cache_path,
            answer_generator=answer_generator,
            retrieval_embedder=retrieval_embedder,
        )

    # Ниже строится guardrailed assistant. Baseline уже ушел в отдельный
    # baseline_pipeline.py, чтобы его можно было читать без guardrail веток.
    classifier = _build_guard_classifier(
        guard_classifier,
        classifier_model=classifier_model,
        allow_remote_models=allow_remote_models,
        env_file=env_file,
        model_config=model_config,
    )
    verifier = _build_entailment_verifier(
        entailment_verifier,
        entailment_model=entailment_model,
        allow_remote_models=allow_remote_models,
        env_file=env_file,
        model_config=model_config,
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
            embedding_cache_path=embedding_cache_path,
            embedder=retrieval_embedder,
            corpus_path=Path(corpus_path),
            policy_context_top_k=policy_context_top_k,
            policy_context_min_score=(
                policy_context_min_score
                if policy_context_min_score is not None
                else 0.0
            ),
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
        classifier_strategy=classifier_strategy,
        retrieval_top_k=retrieval_top_k,
        evidence_min_score=evidence_min_score,
        policy_context_min_score=policy_context_min_score,
        entailment_verifier=verifier,
        entailment_min_confidence=entailment_min_confidence,
    )


def _build_answer_generator(
    generator: str,
    *,
    answer_model: str | None = None,
    allow_remote_models: bool = False,
    env_file: Path | None = None,
    model_config: OpenAIModelConfig | None = None,
) -> AnswerGenerator | None:
    if generator == "extractive":
        return None
    if generator == "openai":
        from .model_config import DEFAULT_OPENAI_ANSWER_MODEL, OpenAIModelConfig
        from .openai_models import OpenAIAnswerGenerator

        config = (
            replace(
                model_config,
                answer_model=answer_model or model_config.answer_model,
            )
            if model_config is not None
            else OpenAIModelConfig(
                answer_model=answer_model or DEFAULT_OPENAI_ANSWER_MODEL,
                allow_remote_models=allow_remote_models,
                env_file=env_file,
            )
        )
        return OpenAIAnswerGenerator(config)
    raise ValueError("generator must be 'extractive' or 'openai'")


def _build_guard_classifier(
    guard_classifier: str,
    *,
    classifier_model: str | None = None,
    allow_remote_models: bool = False,
    env_file: Path | None = None,
    model_config: OpenAIModelConfig | None = None,
) -> GuardClassifier | None:
    if guard_classifier == "none":
        return None
    if guard_classifier == "openai":
        from .model_config import DEFAULT_OPENAI_CLASSIFIER_MODEL, OpenAIModelConfig
        from .openai_models import OpenAIGuardClassifier

        config = (
            replace(
                model_config,
                classifier_model=classifier_model or model_config.classifier_model,
            )
            if model_config is not None
            else OpenAIModelConfig(
                classifier_model=classifier_model or DEFAULT_OPENAI_CLASSIFIER_MODEL,
                allow_remote_models=allow_remote_models,
                env_file=env_file,
            )
        )
        return OpenAIGuardClassifier(config)
    raise ValueError("guard_classifier must be 'none' or 'openai'")


def _build_entailment_verifier(
    entailment_verifier: str,
    *,
    entailment_model: str | None = None,
    allow_remote_models: bool = False,
    env_file: Path | None = None,
    model_config: OpenAIModelConfig | None = None,
) -> EntailmentVerifier | None:
    if entailment_verifier == "none":
        return None
    if entailment_verifier == "openai":
        from .model_config import DEFAULT_OPENAI_ENTAILMENT_MODEL, OpenAIModelConfig
        from .openai_models import OpenAIEntailmentVerifier

        config = (
            replace(
                model_config,
                entailment_model=entailment_model or model_config.entailment_model,
            )
            if model_config is not None
            else OpenAIModelConfig(
                entailment_model=(
                    entailment_model or DEFAULT_OPENAI_ENTAILMENT_MODEL
                ),
                allow_remote_models=allow_remote_models,
                env_file=env_file,
            )
        )
        return OpenAIEntailmentVerifier(config)
    raise ValueError("entailment_verifier must be 'none' or 'openai'")


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


def evidence_records(
    retrieved: list[tuple[Chunk, float]],
) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "text": chunk.text,
            "score": round(float(score), 6),
        }
        for chunk, score in retrieved
    ]


def sanitize_chunk(chunk: Chunk, policy: GuardrailPolicy) -> Chunk:
    sanitized = sanitize_untrusted_context(chunk.text, policy)
    return replace(chunk, text=sanitized)


def _metadata_text(chunk: Chunk, key: str) -> str:
    value = chunk.metadata.get(key)
    if value is None or isinstance(value, list):
        return ""
    return str(value).strip()
