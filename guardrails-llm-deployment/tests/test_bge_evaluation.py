from pathlib import Path

import guardrails_llm.bge_evaluation as bge_evaluation
from guardrails_llm.bge_evaluation import _select_threshold, run_bge_common_split_evaluation
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


def test_threshold_selection_preserves_runtime_score_precision() -> None:
    lower = 0.6003326
    upper = 0.600334

    threshold = _select_threshold([lower, upper], [False, True])

    assert threshold == (lower + upper) / 2


def test_threshold_selection_ranks_candidates_with_unrounded_metrics(monkeypatch) -> None:
    original = bge_evaluation._binary_metrics
    round_modes = []

    def observed_binary_metrics(scores, targets, threshold, *, round_metrics=True):
        round_modes.append(round_metrics)
        return original(
            scores,
            targets,
            threshold,
            round_metrics=round_metrics,
        )

    monkeypatch.setattr(bge_evaluation, "_binary_metrics", observed_binary_metrics)

    _select_threshold([0.1, 0.2, 0.3], [False, True, True])

    assert round_modes
    assert all(mode is False for mode in round_modes)


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
        assert technique["retrieval"]["development"]["total"] == 900
        assert technique["retrieval"]["calibration"]["total"] == 300
        assert technique["retrieval"]["selected_threshold"] is not None
        assert technique["retrieval"]["development"]["document_recall_at_k"] >= 0
        assert (
            technique["retrieval"]["development"][
                "document_recall_within_top_k_chunks"
            ]
            >= 0
        )
        assert set(technique["guard_similarity"]) == {
            "prompt_injection",
            "pii",
            "academic_integrity",
        }
        for guard in technique["guard_similarity"].values():
            assert guard["development"]["total"] == 1200
            assert guard["calibration"]["total"] == 400
            assert guard["development"]["true_positive"] + guard["development"][
                "false_negative"
            ] > 0
            assert guard["calibration"]["true_positive"] + guard["calibration"][
                "false_negative"
            ] > 0

    academic = next(
        row
        for row in details["bge_m3"]
        if row["expected_classifier_label"] == "academic_integrity"
    )
    blocked = next(
        row
        for row in details["bge_m3"]
        if row["expected_classifier_label"] == "prompt_injection"
    )
    assert academic["retrieval_query"] == (
        "academic integrity graded work complete submissions hints similar examples"
    )
    assert academic["retrieval_attempted"] is True
    assert academic["document_ranking_candidate_limit"] == 782
    assert len(academic["ranked_doc_ids"]) <= 3
    assert blocked["retrieval_attempted"] is False
    assert blocked["retrieval_query"] is None
    assert summary["runtime_retriever_min_score"] == 0.05
