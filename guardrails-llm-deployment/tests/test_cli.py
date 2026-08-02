from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from guardrails_llm.cli import (
    _comparison_scenarios,
    _load_comparison_policy,
    _preload_retrieval_embedder,
    main,
)
from guardrails_llm.embeddings import HashingEmbedder
from guardrails_llm.evaluation import load_eval_cases
from guardrails_llm.evaluation_dataset import DATASET_FILENAMES, write_evaluation_dataset
from guardrails_llm.guardrail_policy import load_guardrail_policy
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


@pytest.mark.parametrize("command", ["evaluate", "compare-guardrails"])
def test_evaluation_commands_reject_unreviewed_v2_holdout(
    command: str,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    write_evaluation_dataset(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            command,
            "--corpus",
            str(ROOT / "data" / "python_course_docs.jsonl"),
            "--course-id",
            "python-intro",
            "--cases",
            str(tmp_path / DATASET_FILENAMES["holdout"]),
            "--limit-cases",
            "1",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        main()

    assert "independently reviewed and adjudicated" in capsys.readouterr().err


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
        "fuzzy_plus_shared_controls",
        "similarity_plus_shared_controls",
        "default_guardrails",
        "hybrid_policy_guardrails",
    ]
    assert data["baseline"]["confidence_intervals"]["row"]["sampling_units"] == 2
    for label in (
        "similarity_plus_shared_controls",
        "hybrid_policy_guardrails",
    ):
        assert data[label]["embedding_preload"]["guard_similarity"]
        assert data[label]["avg_batch_amortized_latency_ms"] >= 0
        assert data[label]["latency_scope"] == "pipeline_after_batch_preload"

    for label in (
        "baseline",
        "normalized_regex_guardrails",
        "fuzzy_plus_shared_controls",
        "default_guardrails",
    ):
        assert "embedding_preload" not in data[label]


def test_review_judge_study_command_starts_selected_reviewer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_serve_review_ui(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("guardrails_llm.cli.serve_review_ui", fake_serve_review_ui)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "review-judge-study",
            "--study-dir",
            str(tmp_path),
            "--reviewer",
            "reviewer_b",
            "--port",
            "8877",
            "--section-size",
            "8",
            "--open",
        ],
    )

    main()

    assert captured == {
        "study_dir": tmp_path,
        "reviewer": "reviewer_b",
        "port": 8877,
        "section_size": 8,
        "open_browser": True,
        "allow_reviewer_switch": False,
    }


def test_manage_policy_command_starts_local_instructor_ui(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy_path = tmp_path / "policy.toml"
    runtime_path = tmp_path / "runtime.toml"
    policy_path.write_text("extends_default = true\n", encoding="utf-8")
    runtime_path.write_text("schema_version = 1\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeManager:
        def __init__(self, policy, runtime_config, *, state_dir=None):
            captured.update(
                policy=policy,
                runtime_config=runtime_config,
                state_dir=state_dir,
            )

    def fake_serve_policy_ui(manager, **kwargs: object) -> None:
        captured["manager"] = manager
        captured.update(kwargs)

    monkeypatch.setattr("guardrails_llm.cli.PolicyManager", FakeManager)
    monkeypatch.setattr("guardrails_llm.cli.serve_policy_ui", fake_serve_policy_ui)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "manage-policy",
            "--policy",
            str(policy_path),
            "--runtime-config",
            str(runtime_path),
            "--state-dir",
            str(tmp_path / "state"),
            "--port",
            "8870",
            "--open",
        ],
    )

    main()

    assert isinstance(captured["manager"], FakeManager)
    assert captured["policy"] == policy_path
    assert captured["runtime_config"] == runtime_path
    assert captured["state_dir"] == tmp_path / "state"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8870
    assert captured["open_browser"] is True


def test_prepare_judge_recommendations_command(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_prepare(study_dir: Path) -> dict[str, object]:
        captured["study_dir"] = study_dir
        return {"items": 400}

    monkeypatch.setattr(
        "guardrails_llm.cli.prepare_review_recommendations",
        fake_prepare,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "prepare-judge-recommendations",
            "--study-dir",
            str(tmp_path),
        ],
    )

    main()

    assert captured == {"study_dir": tmp_path}


def test_review_judge_reconciliation_command(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_serve(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "guardrails_llm.cli.serve_reconciliation_ui",
        fake_serve,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "review-judge-reconciliation",
            "--study-dir",
            str(tmp_path),
            "--port",
            "8890",
            "--section-size",
            "12",
            "--open",
        ],
    )

    main()

    assert captured == {
        "study_dir": tmp_path,
        "port": 8890,
        "section_size": 12,
        "open_browser": True,
    }


def test_workshop3_demo_command_passes_live_options(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_write_workshop3_demo(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"output_path": str(kwargs["output_path"]), "mode": "live"}

    monkeypatch.setattr(
        "guardrails_llm.cli.write_workshop3_demo",
        fake_write_workshop3_demo,
    )
    monkeypatch.setattr(sys, "argv", [
        "guardrails-llm",
        "workshop3-demo",
        "--evidence",
        str(tmp_path / "evidence.json"),
        "--output",
        str(tmp_path / "demo.html"),
        "--live",
        "--allow-remote-models",
        "--env-file",
        str(tmp_path / ".env"),
    ])

    main()

    assert captured == {
        "evidence_path": tmp_path / "evidence.json",
        "output_path": tmp_path / "demo.html",
        "live": True,
        "allow_remote_models": True,
        "env_file": tmp_path / ".env",
        "open_browser": False,
    }


def test_workshop3_demo_entrypoints_exist_and_are_executable() -> None:
    package_script = ROOT / "scripts" / "run_workshop3_demo.sh"
    root_script = ROOT.parent / "scripts" / "run_workshop3_demo.sh"

    assert package_script.exists()
    assert root_script.exists()
    assert package_script.stat().st_mode & 0o111
    assert root_script.stat().st_mode & 0o111


def test_workshop3_demo_script_runs_offline_without_open_flag(
    tmp_path: Path,
) -> None:
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\"\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    environment = os.environ | {
        "OPEN_BROWSER": "0",
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
    }

    result = subprocess.run(
        [str(ROOT / "scripts" / "run_workshop3_demo.sh")],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.stdout.strip() == "run guardrails-llm workshop3-demo"


def test_comparison_scenarios_isolate_local_guardrail_techniques() -> None:
    policy = load_guardrail_policy(
        ROOT / "data" / "guardrail_policy.toml",
        similarity_embedder=HashingEmbedder(),
    )
    args = SimpleNamespace(guard_classifier="none")
    scenarios = {
        label: (scenario_policy, profile)
        for label, _mode, scenario_policy, _classifier, profile
        in _comparison_scenarios(args, policy)
    }

    fuzzy_policy, fuzzy_profile = scenarios["fuzzy_plus_shared_controls"]
    assert not fuzzy_policy.input_rules
    assert fuzzy_policy.input_fuzzy_rules
    assert not fuzzy_policy.input_similarity_rules
    assert not fuzzy_policy.output_rules
    assert fuzzy_policy.output_fuzzy_rules
    assert not fuzzy_policy.context_rules
    assert fuzzy_policy.context_fuzzy_rules

    similarity_policy, similarity_profile = scenarios[
        "similarity_plus_shared_controls"
    ]
    assert not similarity_policy.input_rules
    assert not similarity_policy.input_fuzzy_rules
    assert similarity_policy.input_similarity_rules
    assert not similarity_policy.output_rules
    assert not similarity_policy.output_fuzzy_rules
    assert not similarity_policy.context_rules
    assert not similarity_policy.context_fuzzy_rules

    for scenario_policy, profile in (
        (fuzzy_policy, fuzzy_profile),
        (similarity_policy, similarity_profile),
    ):
        assert scenario_policy.allowed_visibility == frozenset({"public"})
        assert scenario_policy.require_citations is True
        assert profile["shared_controls"] == [
            "metadata_filter",
            "citation_requirement",
        ]


def test_comparison_scenarios_include_qwen_only_and_complete_hybrid() -> None:
    policy = load_guardrail_policy(
        ROOT / "data" / "guardrail_policy.toml",
        similarity_embedder=HashingEmbedder(),
    )
    args = SimpleNamespace(guard_classifier="openai")
    scenarios = {
        label: (scenario_policy, classifier, profile)
        for label, _mode, scenario_policy, classifier, profile
        in _comparison_scenarios(args, policy)
    }

    qwen_policy, qwen_classifier, qwen_profile = scenarios["qwen_classifier_only"]
    assert qwen_classifier == "openai"
    assert qwen_profile["classifier_strategy"] == "always"
    assert not qwen_policy.input_rules
    assert not qwen_policy.input_fuzzy_rules
    assert not qwen_policy.input_similarity_rules

    hybrid_policy, hybrid_classifier, hybrid_profile = scenarios[
        "complete_inhouse_hybrid"
    ]
    assert hybrid_classifier == "openai"
    assert hybrid_profile["classifier_strategy"] == "always"
    assert hybrid_policy.input_rules
    assert hybrid_policy.input_fuzzy_rules
    assert hybrid_policy.input_similarity_rules


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


def test_capture_model_calibration_requires_remote_allowance(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "capture-model-calibration",
            "--component",
            "classifier",
            "--classifier-output",
            str(tmp_path / "classifier.jsonl"),
            "--manifest-output",
            str(tmp_path / "manifest.json"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Remote model calls are disabled" in captured.err
    assert "Traceback" not in captured.err
    assert not list(tmp_path.iterdir())


def test_capture_model_calibration_wires_safe_cli_options(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    captured_kwargs = {}

    def fake_capture(**kwargs):
        captured_kwargs.update(kwargs)
        return {"evidence_scope": "live_remote_model_capture"}

    monkeypatch.setattr(
        "guardrails_llm.cli.run_model_calibration_capture",
        fake_capture,
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "capture-model-calibration",
            "--component",
            "both",
            "--limit-cases",
            "5",
            "--classifier-model",
            "classifier-model",
            "--judge-model",
            "judge-model",
            "--allow-remote-models",
            "--classifier-output",
            str(tmp_path / "classifier.jsonl"),
            "--judge-output",
            str(tmp_path / "judge.jsonl"),
            "--manifest-output",
            str(tmp_path / "manifest.json"),
        ],
    )

    main()

    assert captured_kwargs["component"] == "both"
    assert captured_kwargs["limit_cases"] == 5
    assert captured_kwargs["config"].allow_remote_models is True
    assert captured_kwargs["config"].classifier_model == "classifier-model"
    assert captured_kwargs["config"].judge_model == "judge-model"
    assert json.loads(capsys.readouterr().out)["evidence_scope"] == (
        "live_remote_model_capture"
    )


def test_capture_v2_classifier_uses_inhouse_profile(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    captured_kwargs = {}

    def fake_capture(**kwargs):
        captured_kwargs.update(kwargs)
        return {"status": "complete", "completed_cases": 2}

    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://learning-services4.fokus.fraunhofer.de/litellm/v1",
    )
    monkeypatch.setattr(
        "guardrails_llm.cli.run_v2_classifier_capture",
        fake_capture,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "capture-v2-classifier",
            "--allow-remote-models",
            "--limit-cases",
            "2",
            "--max-concurrency",
            "4",
            "--retry-failures",
            "--output",
            str(tmp_path / "predictions.jsonl"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ],
    )

    main()

    assert captured_kwargs["config"].classifier_model == "Qwen/Qwen3.6-35B-A3B"
    assert captured_kwargs["config"].allow_remote_models is True
    assert captured_kwargs["limit_cases"] == 2
    assert captured_kwargs["max_concurrency"] == 4
    assert captured_kwargs["retry_failures"] is True
    assert json.loads(capsys.readouterr().out)["completed_cases"] == 2


def test_evaluate_v2_classifier_is_local(tmp_path: Path, monkeypatch, capsys) -> None:
    captured_kwargs = {}

    def fake_evaluate(**kwargs):
        captured_kwargs.update(kwargs)
        return {"quality_gates": {"all_passed": False}}

    monkeypatch.setattr(
        "guardrails_llm.cli.evaluate_v2_classifier_capture",
        fake_evaluate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "evaluate-v2-classifier",
            "--predictions",
            str(tmp_path / "predictions.jsonl"),
            "--limit-cases",
            "10",
        ],
    )

    main()

    assert captured_kwargs["limit_cases"] == 10
    assert "config" not in captured_kwargs
    assert json.loads(capsys.readouterr().out)["quality_gates"]["all_passed"] is False


def test_qwen3guard_capture_requires_explicit_remote_permission(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "capture-qwen3guard-classifier",
            "--limit-cases",
            "1",
            "--output",
            str(tmp_path / "predictions.jsonl"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        main()

    assert "remote model calls are disabled" in capsys.readouterr().err.lower()
    assert not list(tmp_path.iterdir())


def test_qwen3guard_capture_wires_explicit_options(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    captured: dict[str, object] = {}

    def fake_capture(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "complete", "completed_cases": 4}

    monkeypatch.setattr(
        "guardrails_llm.cli.run_qwen3guard_capture",
        fake_capture,
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "capture-qwen3guard-classifier",
            "--allow-remote-models",
            "--classifier-model",
            "Qwen3guard-gen-4b",
            "--limit-cases",
            "4",
            "--max-concurrency",
            "2",
            "--retry-failures",
            "--output",
            str(tmp_path / "predictions.jsonl"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ],
    )

    main()

    config = captured["config"]
    assert config.classifier_model == "Qwen3guard-gen-4b"
    assert config.allow_remote_models is True
    assert captured["limit_cases"] == 4
    assert captured["max_concurrency"] == 2
    assert captured["retry_failures"] is True
    assert json.loads(capsys.readouterr().out)["completed_cases"] == 4


def test_qwen3guard_comparison_writes_offline_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    output = tmp_path / "comparison.json"
    captured: dict[str, object] = {}

    def fake_compare(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "case_alignment": {
                "identical_case_ids": True,
                "total": 600,
            }
        }

    monkeypatch.setattr(
        "guardrails_llm.cli.compare_qwen_classifier_captures",
        fake_compare,
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "compare-qwen3guard-classifier",
            "--qwen-predictions",
            str(tmp_path / "qwen.jsonl"),
            "--qwen3guard-predictions",
            str(tmp_path / "qwen3guard.jsonl"),
            "--output-json",
            str(output),
        ],
    )

    main()

    assert json.loads(output.read_text(encoding="utf-8"))[
        "case_alignment"
    ]["total"] == 600
    assert "config" not in captured
    assert json.loads(capsys.readouterr().out)["case_alignment"][
        "identical_case_ids"
    ] is True


def test_prepare_inhouse_bge_wires_profile_without_calling_api(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    captured_kwargs = {}

    def fake_prepare(**kwargs):
        captured_kwargs.update(kwargs)
        return {"status": "prepared", "index": {"chunks": 10}}

    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://learning-services4.fokus.fraunhofer.de/litellm/v1",
    )
    monkeypatch.setattr("guardrails_llm.cli.prepare_inhouse_bge", fake_prepare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "prepare-inhouse-bge",
            "--allow-remote-models",
            "--index-dir",
            str(tmp_path / "chroma"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ],
    )

    main()

    assert captured_kwargs["config"].embedding_model == "BAAI/bge-m3"
    assert captured_kwargs["cache_path"].name == "bge-m3.jsonl"
    assert captured_kwargs["config"].allow_remote_models is True
    assert json.loads(capsys.readouterr().out)["status"] == "prepared"


def test_calibrate_inhouse_bge_writes_summary_and_details(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    captured_kwargs = {}

    def fake_evaluate(**kwargs):
        captured_kwargs.update(kwargs)
        return {"holdout_used": False}, {"bge_m3": [], "hashing": []}

    summary_path = tmp_path / "summary.json"
    details_path = tmp_path / "details.json"
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://learning-services4.fokus.fraunhofer.de/litellm/v1",
    )
    monkeypatch.setattr(
        "guardrails_llm.cli.run_bge_common_split_evaluation",
        fake_evaluate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "calibrate-inhouse-bge",
            "--allow-remote-models",
            "--output-json",
            str(summary_path),
            "--output-details-json",
            str(details_path),
        ],
    )

    main()

    assert captured_kwargs["config"].embedding_model == "BAAI/bge-m3"
    assert captured_kwargs["cache_path"].name == "bge-m3.jsonl"
    assert json.loads(summary_path.read_text())["holdout_used"] is False
    assert set(json.loads(details_path.read_text())) == {"bge_m3", "hashing"}
    assert json.loads(capsys.readouterr().out)["holdout_used"] is False


def test_capture_inhouse_calibration_uses_frozen_threshold_by_default(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    captured_kwargs = {}

    def fake_capture(**kwargs):
        captured_kwargs.update(kwargs)
        return {"status": "complete", "completed_runs": 4}

    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://learning-services4.fokus.fraunhofer.de/litellm/v1",
    )
    monkeypatch.setattr(
        "guardrails_llm.cli.run_calibration_e2e_capture",
        fake_capture,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "capture-inhouse-calibration",
            "--allow-remote-models",
            "--case-id",
            "m3v2-break-c03-a04",
            "--case-id",
            "m3v2-plotting-c03-a04",
            "--max-concurrency",
            "2",
            "--retry-failures",
            "--output",
            str(tmp_path / "capture.jsonl"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ],
    )

    main()

    config = captured_kwargs["config"]
    assert config.embedding_model == "BAAI/bge-m3"
    assert config.answer_model == "Qwen/Qwen3.6-35B-A3B"
    assert config.classifier_model == "Qwen/Qwen3.6-35B-A3B"
    assert config.entailment_model == "Qwen/Qwen3.6-35B-A3B"
    assert captured_kwargs["evidence_min_score"] == 0.5203531980514526
    assert captured_kwargs["classifier_min_confidence"] == 0.65
    assert captured_kwargs["entailment_min_confidence"] == 0.80
    assert captured_kwargs["case_ids"] == [
        "m3v2-break-c03-a04",
        "m3v2-plotting-c03-a04",
    ]
    assert captured_kwargs["max_concurrency"] == 2
    assert captured_kwargs["retry_failures"] is True
    assert json.loads(capsys.readouterr().out)["completed_runs"] == 4


def test_query_wires_grounded_evidence_options(monkeypatch, capsys) -> None:
    captured_kwargs: dict[str, object] = {}

    class FakeAssistant:
        def answer(self, question: str):
            assert question == "What is RAG?"
            return SimpleNamespace(
                answer="Grounded answer.",
                citations=["RAG (rag)"],
                disposition="answer",
            )

    def fake_build_assistant(*_args, **kwargs):
        captured_kwargs.update(kwargs)
        return FakeAssistant()

    monkeypatch.setattr("guardrails_llm.cli.build_assistant", fake_build_assistant)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guardrails-llm",
            "query",
            "--question",
            "What is RAG?",
            "--evidence-min-score",
            "7.3",
            "--policy-context-top-k",
            "2",
            "--policy-context-min-score",
            "0.48",
            "--entailment-verifier",
            "openai",
            "--entailment-model",
            "Qwen/Qwen3.6-35B-A3B",
            "--entailment-min-confidence",
            "0.88",
            "--classifier-min-confidence",
            "0.77",
            "--allow-remote-models",
        ],
    )

    main()

    assert captured_kwargs["evidence_min_score"] == 7.3
    assert captured_kwargs["policy_context_top_k"] == 2
    assert captured_kwargs["policy_context_min_score"] == 0.48
    assert captured_kwargs["entailment_verifier"] == "openai"
    assert captured_kwargs["entailment_model"] == "Qwen/Qwen3.6-35B-A3B"
    assert captured_kwargs["entailment_min_confidence"] == 0.88
    assert captured_kwargs["classifier_min_confidence"] == 0.77
    assert json.loads(capsys.readouterr().out)["answer"] == "Grounded answer."


def test_validate_runtime_config_reports_safe_versioned_controls(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["guardrails-llm", "validate-runtime-config"],
    )

    main()

    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 1
    assert report["profile"] == "inhouse"
    assert report["models"]["embedding"] == "BAAI/bge-m3"
    assert report["thresholds"]["classifier_min_confidence"] == 0.65
    assert len(report["sha256"]) == 64
    assert "api_key" not in json.dumps(report)


def test_refresh_calibration_evidence_forwards_all_artifact_paths(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    captured = {}

    def fake_refresh(**kwargs):
        captured.update(kwargs)
        return {"cases": 400}

    monkeypatch.setattr(
        "guardrails_llm.cli.refresh_calibration_evidence", fake_refresh
    )
    paths = {
        "--deterministic-capture": tmp_path / "deterministic.json",
        "--deterministic-manifest": tmp_path / "deterministic-manifest.json",
        "--model-evaluation": tmp_path / "model-evaluation.json",
        "--model-manifest": tmp_path / "manifest.json",
        "--model-capture": tmp_path / "capture.jsonl",
        "--deterministic-output": tmp_path / "deterministic-output.json",
        "--model-output": tmp_path / "model-output.json",
        "--failure-output": tmp_path / "failures.json",
        "--output-json": tmp_path / "final.json",
        "--output-markdown": tmp_path / "final.md",
    }
    argv = ["guardrails-llm", "refresh-calibration-evidence"]
    for option, path in paths.items():
        argv.extend([option, str(path)])
    monkeypatch.setattr(sys, "argv", argv)

    main()

    assert captured == {
        "deterministic_path": paths["--deterministic-capture"],
        "deterministic_manifest_path": paths["--deterministic-manifest"],
        "model_evaluation_path": paths["--model-evaluation"],
        "model_manifest_path": paths["--model-manifest"],
        "model_capture_path": paths["--model-capture"],
        "deterministic_output": paths["--deterministic-output"],
        "model_output": paths["--model-output"],
        "failure_output": paths["--failure-output"],
        "final_json_output": paths["--output-json"],
        "final_markdown_output": paths["--output-markdown"],
    }
    assert json.loads(capsys.readouterr().out) == {"cases": 400}


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
