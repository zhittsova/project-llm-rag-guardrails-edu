from pathlib import Path

from guardrails_llm.pipeline import build_assistant
from guardrails_llm.vector import VectorRetriever, build_vector_index


DATA = Path(__file__).resolve().parents[1] / "data" / "course_docs.jsonl"


def test_build_vector_index_and_query_with_assistant(tmp_path: Path) -> None:
    index_dir = tmp_path / "chroma"

    stats = build_vector_index(DATA, index_dir)
    assistant = build_assistant(DATA, mode="guardrailed", retriever_backend="vector", index_dir=index_dir)
    response = assistant.answer("What is retrieval augmented generation?")

    assert stats.documents == 6
    assert stats.chunks >= 6
    assert response.citations
    assert "rag-basics" in response.retrieved_chunks[0]


def test_vector_retriever_filters_private_chunks(tmp_path: Path) -> None:
    index_dir = tmp_path / "chroma"
    build_vector_index(DATA, index_dir)
    retriever = VectorRetriever(index_dir)

    results = retriever.search(
        "student email addresses accommodations grades",
        course_id="guardrails-101",
        allowed_visibility={"public"},
    )

    assert all(chunk.visibility == "public" for chunk, _score in results)
    assert all(chunk.doc_id != "private-roster" for chunk, _score in results)
