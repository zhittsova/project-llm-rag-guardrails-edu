from __future__ import annotations

from types import SimpleNamespace

from guardrails_llm.corpus import Chunk
from guardrails_llm.model_config import OpenAIModelConfig
from guardrails_llm.openai_models import OpenAIAnswerGenerator, OpenAIEmbeddingModel


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
        self.responses = FakeResponsesEndpoint()


class FakeResponsesEndpoint:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def create(self, *, model: str, input: str):
        self.calls.append((model, input))
        return SimpleNamespace(output_text="RAG combines retrieval with generation.")


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


def test_openai_answer_generator_uses_retrieved_context_with_fake_client(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    client = FakeOpenAIClient()
    chunk = Chunk(
        chunk_id="rag-basics:0",
        doc_id="rag-basics",
        course_id="guardrails-101",
        title="RAG Basics",
        visibility="public",
        source_type="lecture",
        text="Retrieval augmented generation combines retrieval with generation.",
    )

    generator = OpenAIAnswerGenerator(
        OpenAIModelConfig(
            answer_model="gpt-5.4-mini",
            allow_remote_models=True,
            env_file=env_file,
        ),
        client=client,
    )

    answer = generator.generate("What is RAG?", [chunk])

    assert answer == "RAG combines retrieval with generation."
    model, prompt = client.responses.calls[0]
    assert model == "gpt-5.4-mini"
    assert "rag-basics:0" in prompt
    assert "What is RAG?" in prompt
