from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from guardrails_llm.cli import _load_comparison_policy, _preload_retrieval_embedder, main
from guardrails_llm.evaluation import load_eval_cases
from guardrails_llm.model_config import RemoteModelCallError


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "course_docs.jsonl"
CASES = ROOT / "data" / "eval_cases.jsonl"
RETRIEVAL_CASES = ROOT / "data" / "retrieval_cases_milestone3_v1.jsonl"
CLASSIFIER_CALIBRATION_CASES = ROOT / "data" / "model_classifier_calibration_v1.jsonl"
CLASSIFIER_CALIBRATION_PREDICTIONS = (
    ROOT / "tests" / "fixtures" / "classifier_predictions_v1.jsonl"
)
JUDGE_CALIBRATION_CASES = ROOT / "data" / "judge_calibration_v1.jsonl"
JUDGE_CALIBRATION_PREDICTIONS = (
    ROOT / "tests" / "fixtures" / "judge_predictions_v1.jsonl"
)
MILESTONE3_CASES = ROOT / "data" / "eval_cases_milestone3_holdout_v3.jsonl"
MILESTONE3_RESULTS = ROOT / "reports" / "disposition_guardrail_holdout_v3_results.json"


class TrackingEmbedder:
    model_name = "BAAI/bge-m3"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0, 0.0] for _text in texts]


def test_compare_guardrails_writes_json_artifact(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "compare-guardrails",
            "--corpus",
            str(DATA),
            "--cases",
            str(CASES),
            "--limit-cases",
            "2",
            "--output-json",
            str(output),
        ],
    )

    main()

    data = json.loads(output.read_text(encoding="utf-8"))
    assert list(data) == [
        "baseline",
        "normalized_regex_guardrails",
        "default_guardrails",
        "hybrid_policy_guardrails",
    ]


def test_compare_guardrails_writes_detailed_results(tmp_path: Path, monkeypatch) -> None:
    summary_output = tmp_path / "comparison.json"
    results_output = tmp_path / "comparison-results.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "compare-guardrails",
            "--corpus",
            str(DATA),
            "--cases",
            str(CASES),
            "--limit-cases",
            "2",
            "--output-json",
            str(summary_output),
            "--output-results-json",
            str(results_output),
        ],
    )

    main()

    summaries = json.loads(summary_output.read_text(encoding="utf-8"))
    details = json.loads(results_output.read_text(encoding="utf-8"))
    assert list(details) == list(summaries)
    for label, results in details.items():
        assert len(results) == 2, label
        assert {
            "expected_behavior",
            "actual_behavior",
            "attack_type",
            "difficulty",
        } <= set(results[0])
        assert summaries[label]["behavior_confusion_matrix"]


def test_benchmark_retrieval_writes_summary_and_details(
    tmp_path: Path,
    monkeypatch,
) -> None:
    summary_output = tmp_path / "retrieval-summary.json"
    results_output = tmp_path / "retrieval-results.json"
    index_dir = tmp_path / "index"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "benchmark-retrieval",
            "--corpus",
            str(DATA),
            "--cases",
            str(RETRIEVAL_CASES),
            "--index-dir",
            str(index_dir),
            "--output-json",
            str(summary_output),
            "--output-results-json",
            str(results_output),
        ],
    )

    main()

    summaries = json.loads(summary_output.read_text(encoding="utf-8"))
    details = json.loads(results_output.read_text(encoding="utf-8"))
    manifest = json.loads(
        (index_dir / "course_chunks_manifest.json").read_text(encoding="utf-8")
    )
    assert list(summaries) == ["lexical", "hashing_vector"]
    assert list(details) == list(summaries)
    assert manifest["chunk_size"] == 650
    assert manifest["chunk_overlap"] == 80
    for label in summaries:
        assert summaries[label]["total"] == 24
        assert summaries[label]["relevance_total"] == 20
        assert summaries[label]["visibility_total"] == 4
        assert "recall_at_1" in summaries[label]
        assert "recall_at_3" in summaries[label]
        assert "mrr" in summaries[label]
        assert len(details[label]) == 24
        assert {
            "retrieved_doc_ids",
            "first_relevant_rank",
            "forbidden_hits",
        } <= set(details[label][0])


def test_evaluate_model_calibration_writes_local_fixture_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "calibration.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "evaluate-model-calibration",
            "--classifier-cases",
            str(CLASSIFIER_CALIBRATION_CASES),
            "--classifier-predictions",
            str(CLASSIFIER_CALIBRATION_PREDICTIONS),
            "--judge-cases",
            str(JUDGE_CALIBRATION_CASES),
            "--judge-predictions",
            str(JUDGE_CALIBRATION_PREDICTIONS),
            "--source-cases",
            str(MILESTONE3_CASES),
            "--source-results",
            str(MILESTONE3_RESULTS),
            "--output-json",
            str(output),
        ],
    )

    main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["evidence_scope"] == "fixture_replay_only"
    assert payload["classifier"]["summary"]["total"] == 36
    assert payload["judge"]["summary"]["total"] == 24


def test_compare_guardrails_records_validation_split(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "compare-guardrails",
            "--corpus",
            str(DATA),
            "--cases",
            str(CASES),
            "--retriever",
            "langchain",
            "--case-split",
            "validation",
            "--output-json",
            str(output),
        ],
    )

    main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    for scenario in payload.values():
        assert scenario["eval_split"] == "validation"
        assert scenario["total"] < 12


def test_comparison_preloads_retrieval_questions_once(monkeypatch) -> None:
    cases = load_eval_cases(CASES)[:2]
    delegate = TrackingEmbedder()
    monkeypatch.setattr("guardrails_llm.cli.create_embedder", lambda *args, **kwargs: delegate)
    args = SimpleNamespace(
        retriever="vector",
        embedding_provider="openai",
        embedding_model="BAAI/bge-m3",
        allow_remote_models=True,
        env_file=None,
    )

    embedder, stats = _preload_retrieval_embedder(args, cases)

    assert delegate.calls == [[case.question for case in cases]]
    assert embedder.cached_texts == 2
    assert stats["texts"] == 2
    assert stats["model"] == "BAAI/bge-m3"


def test_comparison_preloads_guard_examples_and_questions(monkeypatch) -> None:
    cases = load_eval_cases(CASES)[:2]
    delegate = TrackingEmbedder()
    monkeypatch.setattr("guardrails_llm.cli.create_embedder", lambda *args, **kwargs: delegate)
    args = SimpleNamespace(
        policy=ROOT / "data" / "guardrail_policy.toml",
        guard_embedding_provider="openai",
        guard_embedding_model="BAAI/bge-m3",
        allow_remote_models=True,
        env_file=None,
    )

    policy, stats = _load_comparison_policy(args, cases)

    example_count = sum(len(rule.examples) for rule in policy.input_similarity_rules)
    assert len(delegate.calls) == 1
    assert len(delegate.calls[0]) == example_count + len(cases)
    assert stats["texts"] == example_count + len(cases)


def test_build_index_reports_remote_model_error_without_traceback(monkeypatch, capsys) -> None:
    def fail_build(*args, **kwargs):
        raise RemoteModelCallError("OpenAI embedding request failed: AuthenticationError")

    monkeypatch.setattr("guardrails_llm.cli.build_vector_index", fail_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "build-index",
            "--corpus",
            str(DATA),
            "--embedding-provider",
            "openai",
            "--allow-remote-models",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "OpenAI embedding request failed: AuthenticationError" in captured.err
    assert "Traceback" not in captured.err


def test_remote_guard_embeddings_require_explicit_allowance(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "compare-guardrails",
            "--corpus",
            str(DATA),
            "--cases",
            str(CASES),
            "--guard-embedding-provider",
            "openai",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Remote model calls are disabled" in captured.err
    assert "Traceback" not in captured.err
