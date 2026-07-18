from pathlib import Path

import pytest

from guardrails_llm import vector
from guardrails_llm.pipeline import build_assistant
from guardrails_llm.vector import (
    VectorIndexConfigurationError,
    VectorIndexNotFoundError,
    VectorRetriever,
    build_vector_index,
)


DATA = Path(__file__).resolve().parents[1] / "data" / "course_docs.jsonl"


class FakeEmbedder:
    model_name = "text-embedding-3-small"

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class FailingEmbedder:
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("embedding provider unavailable")

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding provider unavailable")


class RecordingCollection:
    def __init__(self) -> None:
        self.query_kwargs: dict[str, object] = {}

    def count(self) -> int:
        return 10

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return {
            "documents": [["Public course evidence."]],
            "metadatas": [[{
                "chunk_id": "public-doc:0",
                "doc_id": "public-doc",
                "course_id": "guardrails-101",
                "title": "Public Doc",
                "visibility": "public",
                "source_type": "lecture",
            }]],
            "distances": [[0.1]],
        }


class RecordingClient:
    def __init__(self, collection: RecordingCollection) -> None:
        self.collection = collection

    def get_collection(self, name: str):
        assert name == "course_chunks"
        return self.collection


def test_build_vector_index_and_query_with_assistant(tmp_path: Path) -> None:
    index_dir = tmp_path / "chroma"

    stats = build_vector_index(DATA, index_dir)
    assistant = build_assistant(DATA, mode="guardrailed", retriever_backend="vector", index_dir=index_dir)
    response = assistant.answer("What is retrieval augmented generation?")

    assert stats.documents == 6
    assert stats.chunks >= 6
    assert response.citations
    assert "rag-basics" in response.retrieved_chunks[0]
    assert stats.embedding_provider == "hashing"


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


def test_vector_retriever_applies_metadata_filters_inside_chroma_query(
    tmp_path: Path,
    monkeypatch,
) -> None:
    collection = RecordingCollection()
    monkeypatch.setattr(
        vector,
        "_persistent_client",
        lambda _index_dir: RecordingClient(collection),
    )
    retriever = VectorRetriever(tmp_path / "chroma")

    results = retriever.search(
        "What is RAG?",
        course_id="guardrails-101",
        allowed_visibility={"public"},
        top_k=3,
    )

    assert results[0][0].doc_id == "public-doc"
    assert collection.query_kwargs["n_results"] == 3
    assert collection.query_kwargs["where"] == {
        "$and": [
            {"course_id": {"$eq": "guardrails-101"}},
            {"visibility": {"$in": ["public"]}},
        ]
    }


def test_assistant_uses_injected_retrieval_embedder(tmp_path: Path) -> None:
    index_dir = tmp_path / "chroma"
    embedder = FakeEmbedder()
    build_vector_index(
        DATA,
        index_dir,
        embedding_provider="openai",
        embedding_model=embedder.model_name,
        embedder=embedder,
    )

    assistant = build_assistant(
        DATA,
        mode="guardrailed",
        retriever_backend="vector",
        index_dir=index_dir,
        embedding_provider="openai",
        embedding_model=embedder.model_name,
        retrieval_embedder=embedder,
    )

    assert assistant.answer("What is RAG?").citations


def test_vector_retriever_explains_missing_index(tmp_path: Path) -> None:
    with pytest.raises(VectorIndexNotFoundError, match="build-index"):
        VectorRetriever(tmp_path / "missing-chroma")


def test_vector_index_manifest_rejects_embedding_mismatch(tmp_path: Path) -> None:
    index_dir = tmp_path / "chroma"
    build_vector_index(DATA, index_dir)

    with pytest.raises(VectorIndexConfigurationError, match="was built with hashing"):
        VectorRetriever(index_dir, embedding_provider="openai")


def test_vector_index_can_record_openai_embedding_provider_with_fake_embedder(tmp_path: Path) -> None:
    index_dir = tmp_path / "chroma"

    stats = build_vector_index(
        DATA,
        index_dir,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedder=FakeEmbedder(),
    )
    retriever = VectorRetriever(
        index_dir,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedder=FakeEmbedder(),
    )

    results = retriever.search("retrieval augmented generation")

    assert stats.embedding_provider == "openai"
    assert stats.embedding_model == "text-embedding-3-small"
    assert results


def test_failed_vector_rebuild_preserves_existing_collection(tmp_path: Path) -> None:
    index_dir = tmp_path / "chroma"
    build_vector_index(DATA, index_dir)

    with pytest.raises(RuntimeError, match="embedding provider unavailable"):
        build_vector_index(DATA, index_dir, embedder=FailingEmbedder())

    retriever = VectorRetriever(index_dir)
    results = retriever.search("retrieval augmented generation")

    assert results
