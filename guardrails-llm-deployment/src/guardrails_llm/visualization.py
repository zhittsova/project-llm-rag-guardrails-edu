from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

from .corpus import Chunk, chunk_documents, load_documents
from .guardrail_policy import GuardrailPolicy
from .langchain_rag import langchain_chunk_documents
from .pipeline import build_assistant


@dataclass(frozen=True)
class VisualizationStats:
    output_path: Path
    retrieved_chunks: int


def write_rag_visualization(
    *,
    corpus_path: Path,
    output_path: Path,
    question: str,
    mode: str,
    retriever_backend: str,
    index_dir: Path | None,
    course_id: str,
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
    evidence_min_score: float | None = None,
    entailment_verifier: str = "none",
    entailment_model: str | None = None,
    entailment_min_confidence: float = 0.80,
) -> VisualizationStats:
    assistant = build_assistant(
        corpus_path,
        mode=mode,
        retriever_backend=retriever_backend,
        index_dir=index_dir,
        course_id=course_id,
        guardrail_policy=guardrail_policy,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        allow_remote_models=allow_remote_models,
        env_file=env_file,
        embedding_cache_path=embedding_cache_path,
        generator=generator,
        answer_model=answer_model,
        guard_classifier=guard_classifier,
        classifier_model=classifier_model,
        classifier_strategy=classifier_strategy,
        evidence_min_score=evidence_min_score,
        entailment_verifier=entailment_verifier,
        entailment_model=entailment_model,
        entailment_min_confidence=entailment_min_confidence,
    )
    response = assistant.answer(question)
    chunk_lookup = _chunk_lookup(corpus_path, retriever_backend)
    retrieved = [
        chunk_lookup[chunk_id]
        for chunk_id in response.retrieved_chunks
        if chunk_id in chunk_lookup
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_html(
            corpus_path=corpus_path,
            question=question,
            mode=mode,
            retriever_backend=retriever_backend,
            course_id=course_id,
            response=response,
            retrieved=retrieved,
        ),
        encoding="utf-8",
    )
    return VisualizationStats(output_path=output_path, retrieved_chunks=len(retrieved))


def _chunk_lookup(corpus_path: Path, retriever_backend: str) -> dict[str, Chunk]:
    documents = load_documents(corpus_path)
    if retriever_backend == "lexical":
        chunks = chunk_documents(documents)
    else:
        chunks = langchain_chunk_documents(documents)
    return {chunk.chunk_id: chunk for chunk in chunks}


def _render_html(
    *,
    corpus_path: Path,
    question: str,
    mode: str,
    retriever_backend: str,
    course_id: str,
    response,
    retrieved: list[Chunk],
) -> str:
    stages = _stages_for_mode(mode, response.guard_triggers)
    trigger_text = ", ".join(response.guard_triggers) if response.guard_triggers else "none"
    citations = response.citations or ["none"]
    grounding_status = (
        "Verifier not run"
        if response.grounding_supported is None
        else "Supported"
        if response.grounding_supported
        else "Rejected or insufficient"
    )
    grounding_confidence = (
        "not available"
        if response.grounding_confidence is None
        else f"{response.grounding_confidence:.3f}"
    )
    grounding_error = response.grounding_error or "none"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RAG Pipeline Demo</title>
  <style>
    body {{
      background: #f6f7f9;
      color: #16181d;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
      margin: 0;
    }}
    main {{
      margin: 0 auto;
      max-width: 1080px;
      padding: 32px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    section {{
      background: #ffffff;
      border: 1px solid #d9dee7;
      border-radius: 8px;
      margin: 16px 0;
      padding: 20px;
    }}
    .meta {{
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .pill {{
      background: #eef2f7;
      border-radius: 999px;
      display: inline-block;
      margin: 4px 6px 4px 0;
      padding: 4px 10px;
    }}
    ol {{
      padding-left: 22px;
    }}
    pre {{
      background: #f2f4f8;
      border-radius: 6px;
      overflow-x: auto;
      padding: 12px;
      white-space: pre-wrap;
    }}
    .chunk {{
      border-top: 1px solid #e3e7ee;
      padding-top: 14px;
    }}
  </style>
</head>
<body>
<main>
  <h1>RAG Pipeline Demo</h1>
  <section class="meta">
    <div><strong>Corpus</strong><br>{escape(str(corpus_path))}</div>
    <div><strong>Mode</strong><br>{escape(mode)}</div>
    <div><strong>Retriever</strong><br>{escape(retriever_backend)}</div>
    <div><strong>Course ID</strong><br>{escape(course_id)}</div>
    <div><strong>Latency</strong><br>{response.latency_ms:.2f} ms</div>
    <div><strong>Guard triggers</strong><br>{escape(trigger_text)}</div>
    <div><strong>Disposition</strong><br>{escape(response.disposition.value)}</div>
    <div><strong>Grounding</strong><br>{escape(grounding_status)}</div>
  </section>

  <section>
    <h2>User Question</h2>
    <pre>{escape(question)}</pre>
  </section>

  <section>
    <h2>Pipeline Stages</h2>
    <ol>
      {"".join(f"<li>{escape(stage)}</li>" for stage in stages)}
    </ol>
  </section>

  <section>
    <h2>Retrieved Chunks</h2>
    {_render_chunks(retrieved, response.retrieval_scores, response.supporting_chunks)}
  </section>

  <section>
    <h2>Grounding Decision</h2>
    <div><strong>Status:</strong> {escape(grounding_status)}</div>
    <div><strong>Confidence:</strong> {escape(grounding_confidence)}</div>
    <div><strong>Verification error:</strong> {escape(grounding_error)}</div>
    <div><strong>Supporting chunks:</strong> {_render_values(response.supporting_chunks)}</div>
    <div><strong>Unsupported claims:</strong> {_render_values(response.unsupported_claims)}</div>
  </section>

  <section>
    <h2>Final Answer</h2>
    <pre>{escape(response.answer)}</pre>
  </section>

  <section>
    <h2>Citations</h2>
    {"".join(f'<span class="pill">{escape(citation)}</span>' for citation in citations)}
  </section>
</main>
</body>
</html>
"""


def _stages_for_mode(mode: str, triggers: list[str]) -> list[str]:
    if mode == "baseline":
        return [
            "Load corpus/index",
            "Retrieve closest chunks without safety filters",
            "Build an extractive answer from retrieved chunks",
            "Attach citations",
        ]
    return [
        "Run input guard",
        "Retrieve chunks using native course and visibility filters",
        "Sanitize retrieved context as untrusted text",
        "Apply the configured retrieval evidence threshold",
        "Generate an answer from the retained evidence",
        "Verify answer entailment when a verifier is configured",
        "Run output guard",
        f"Return answer with triggers: {', '.join(triggers) if triggers else 'none'}",
    ]


def _render_chunks(
    chunks: list[Chunk],
    scores: dict[str, float],
    supporting_chunks: list[str],
) -> str:
    if not chunks:
        return "<p>No chunks were used for the final answer.</p>"
    rendered = []
    supporting = set(supporting_chunks)
    for chunk in chunks:
        score = scores.get(chunk.chunk_id)
        score_text = "not available" if score is None else f"{score:.6f}"
        support_text = "yes" if chunk.chunk_id in supporting else "no"
        rendered.append(
            f"""<div class="chunk">
  <strong>{escape(chunk.chunk_id)}</strong>
  <div>{escape(chunk.title)} ({escape(chunk.doc_id)})</div>
  <div>Retrieval score: {escape(score_text)} | Verifier support: {support_text}</div>
  <pre>{escape(_excerpt(chunk.text))}</pre>
</div>"""
        )
    return "\n".join(rendered)


def _render_values(values: list[str]) -> str:
    if not values:
        return "none"
    return "".join(
        f'<span class="pill">{escape(value)}</span>'
        for value in values
    )


def _excerpt(text: str, limit: int = 900) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."
