from pathlib import Path

from guardrails_llm.bge_evaluation import run_bge_common_split_evaluation
from guardrails_llm.inhouse_experiment import prepare_inhouse_bge
from guardrails_llm.model_config import OpenAIModelConfig


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "data" / "eval_cases_milestone3_v2_development.jsonl"
CALIBRATION = ROOT / "data" / "eval_cases_milestone3_v2_calibration.jsonl"
CORPUS = ROOT / "data" / "python_course_docs.jsonl"
POLICY = ROOT / "data" / "guardrail_policy_bge_m3.toml"


class FakeBgeEmbedder:
    model_name = "BAAI/bge-m3"

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [
            [
                float(bool(text)),
                float(sum(ord(character) for character in text) % 17),
                float(len(text) % 11),
            ]
            for text in texts
        ]


def test_bge_evaluation_uses_common_dev_and_calibration_splits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://learning-services4.fokus.fraunhofer.de/litellm/v1",
    )
    config = OpenAIModelConfig(
        embedding_model="BAAI/bge-m3",
        allow_remote_models=True,
    )
    embedder = FakeBgeEmbedder()
    bge_index = tmp_path / "bge-index"
    prepare_inhouse_bge(
        config=config,
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        corpus_path=CORPUS,
        policy_path=POLICY,
        index_dir=bge_index,
        cache_path=tmp_path / "cache.jsonl",
        manifest_path=tmp_path / "prepare.json",
        embedder=embedder,
    )

    summary, details = run_bge_common_split_evaluation(
        config=config,
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        corpus_path=CORPUS,
        policy_path=POLICY,
        bge_index_dir=bge_index,
        hashing_index_dir=tmp_path / "hashing-index",
        cache_path=tmp_path / "cache.jsonl",
        bge_embedder=embedder,
    )

    assert summary["case_counts"] == {"development": 1200, "calibration": 400}
    assert summary["threshold_selection_split"] == "development"
    assert summary["threshold_validation_split"] == "calibration"
    assert set(summary["techniques"]) == {"hashing", "bge_m3"}
    assert len(details["bge_m3"]) == 1600
    assert len(details["hashing"]) == 1600
    for technique in summary["techniques"].values():
        assert technique["retrieval"]["development"]["total"] == 1200
        assert technique["retrieval"]["calibration"]["total"] == 400
        assert technique["retrieval"]["selected_threshold"] is not None
        assert set(technique["guard_similarity"]) == {
            "prompt_injection",
            "pii",
            "academic_integrity",
        }
