import json
from pathlib import Path

import pytest

from guardrails_llm.embeddings import CachedEmbedder, PersistentCachedEmbedder


class CountingEmbedder:
    model_name = "test-embedding"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text))] for text in texts]


def test_cached_embedder_batches_missing_unique_texts() -> None:
    delegate = CountingEmbedder()
    embedder = CachedEmbedder(delegate)

    first = embedder.embed_many(["alpha", "beta", "alpha"])
    second = embedder.embed_many(["beta", "gamma", "gamma"])

    assert first == [[5.0], [4.0], [5.0]]
    assert second == [[4.0], [5.0], [5.0]]
    assert delegate.calls == [["alpha", "beta"], ["gamma"]]
    assert embedder.cached_texts == 3


def test_cached_embedder_rejects_incomplete_provider_response() -> None:
    class IncompleteEmbedder(CountingEmbedder):
        def embed_many(self, texts: list[str]) -> list[list[float]]:
            return []

    embedder = CachedEmbedder(IncompleteEmbedder())

    try:
        embedder.embed("missing")
    except ValueError as exc:
        assert "returned 0 vectors for 1 texts" in str(exc)
    else:
        raise AssertionError("expected incomplete embedding response to fail")


def test_persistent_cache_reuses_vectors_without_raw_text(tmp_path: Path) -> None:
    cache_path = tmp_path / "bge-cache.jsonl"
    first_delegate = CountingEmbedder()
    first = PersistentCachedEmbedder(first_delegate, cache_path, batch_size=2)

    assert first.embed_many(["secret alpha", "beta", "gamma"]) == [
        [12.0],
        [4.0],
        [5.0],
    ]
    assert first_delegate.calls == [["secret alpha", "beta"], ["gamma"]]

    second_delegate = CountingEmbedder()
    second = PersistentCachedEmbedder(second_delegate, cache_path, batch_size=2)

    assert second.embed_many(["gamma", "secret alpha"]) == [[5.0], [12.0]]
    assert second_delegate.calls == []
    assert second.cache_hits == 2
    serialized = cache_path.read_text(encoding="utf-8")
    assert "secret alpha" not in serialized
    assert "beta" not in serialized
    assert "gamma" not in serialized


def test_persistent_cache_keeps_models_isolated(tmp_path: Path) -> None:
    cache_path = tmp_path / "embedding-cache.jsonl"
    first_delegate = CountingEmbedder()
    PersistentCachedEmbedder(first_delegate, cache_path).embed("alpha")

    second_delegate = CountingEmbedder()
    second_delegate.model_name = "different-model"
    PersistentCachedEmbedder(second_delegate, cache_path).embed("alpha")

    records = [json.loads(line) for line in cache_path.read_text().splitlines()]
    assert {record["model"] for record in records} == {
        "test-embedding",
        "different-model",
    }
    assert second_delegate.calls == [["alpha"]]


def test_persistent_cache_rejects_corrupt_record(tmp_path: Path) -> None:
    cache_path = tmp_path / "embedding-cache.jsonl"
    cache_path.write_text('{"schema_version": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid embedding cache record"):
        PersistentCachedEmbedder(CountingEmbedder(), cache_path)


def test_persistent_cache_read_only_mode_rejects_missing_text(tmp_path: Path) -> None:
    cache_path = tmp_path / "embedding-cache.jsonl"
    PersistentCachedEmbedder(CountingEmbedder(), cache_path).embed("cached")
    delegate = CountingEmbedder()
    read_only = PersistentCachedEmbedder(delegate, cache_path, read_only=True)

    assert read_only.embed("cached") == [6.0]
    with pytest.raises(ValueError, match="read-only embedding cache is missing 1 text"):
        read_only.embed("missing")
    assert delegate.calls == []
