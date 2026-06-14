from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

from .corpus import Chunk, chunk_documents, load_documents
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
) -> VisualizationStats:
    assistant = build_assistant(
        corpus_path,
        mode=mode,
        retriever_backend=retriever_backend,
        index_dir=index_dir,
        course_id=course_id,
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
    {_render_chunks(retrieved)}
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
        "Retrieve public chunks for the selected course",
        "Sanitize retrieved context as untrusted text",
        "Build an extractive answer or safe refusal",
        "Run output guard",
        f"Return answer with triggers: {', '.join(triggers) if triggers else 'none'}",
    ]


def _render_chunks(chunks: list[Chunk]) -> str:
    if not chunks:
        return "<p>No chunks were used for the final answer.</p>"
    rendered = []
    for chunk in chunks:
        rendered.append(
            f"""<div class="chunk">
  <strong>{escape(chunk.chunk_id)}</strong>
  <div>{escape(chunk.title)} ({escape(chunk.doc_id)})</div>
  <pre>{escape(_excerpt(chunk.text))}</pre>
</div>"""
        )
    return "\n".join(rendered)


def _excerpt(text: str, limit: int = 900) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."
