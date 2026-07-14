from guardrails_llm.embeddings import CachedEmbedder


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
