from __future__ import annotations

from types import SimpleNamespace

from guardrails_llm.model_config import OpenAIModelConfig
from guardrails_llm.openai_models import OpenAIEmbeddingModel


class FakeEmbeddingsEndpoint:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def create(self, *, model: str, input: list[str]):
        self.calls.append((model, input))
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[float(index), 1.0])
                for index, _text in enumerate(input)
            ]
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsEndpoint()


def test_openai_embedding_model_uses_configured_model_with_fake_client(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    client = FakeOpenAIClient()

    embedder = OpenAIEmbeddingModel(
        OpenAIModelConfig(
            embedding_model="text-embedding-3-small",
            allow_remote_models=True,
            env_file=env_file,
        ),
        client=client,
    )

    vectors = embedder.embed_many(["alpha", "beta"])

    assert vectors == [[0.0, 1.0], [1.0, 1.0]]
    assert client.embeddings.calls == [("text-embedding-3-small", ["alpha", "beta"])]
