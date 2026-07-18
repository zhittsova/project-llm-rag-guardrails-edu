import json
import threading
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from guardrails_llm.guard_classifier import GuardClassification
from guardrails_llm.inhouse_experiment import (
    build_balanced_classifier_benchmark,
    derive_classifier_label,
    evaluate_v2_classifier_capture,
    prepare_inhouse_bge,
    run_v2_classifier_capture,
)
from guardrails_llm.model_config import OpenAIModelConfig, RemoteModelsNotAllowedError
from guardrails_llm.model_calibration import ClassifierPrediction
from guardrails_llm.model_profiles import INHOUSE_LLM_MODEL
from guardrails_llm.retrieval_routing import ACADEMIC_INTEGRITY_RETRIEVAL_QUERY


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "data" / "eval_cases_milestone3_v2_development.jsonl"
CALIBRATION = ROOT / "data" / "eval_cases_milestone3_v2_calibration.jsonl"
CORPUS = ROOT / "data" / "python_course_docs.jsonl"


class EchoClassifier:
    model_name = INHOUSE_LLM_MODEL

    def __init__(self) -> None:
        self.calls: list[str] = []

    def classify(self, text: str) -> GuardClassification:
        self.calls.append(text)
        return GuardClassification(label="safe", confidence=0.9, explanation="fixture")


class ConcurrentClassifier(EchoClassifier):
    def __init__(self, workers: int) -> None:
        super().__init__()
        self._barrier = threading.Barrier(workers)
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def classify(self, text: str) -> GuardClassification:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        self._barrier.wait(timeout=2)
        time.sleep(0.01)
        with self._lock:
            self._active -= 1
        return super().classify(text)


class FakeBgeEmbedder:
    model_name = "BAAI/bge-m3"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(bool(text)), float(len(text) % 7)] for text in texts]


def _configure_inhouse(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://learning-services4.fokus.fraunhofer.de/litellm/v1",
    )


def test_balanced_classifier_benchmark_has_100_cases_per_label() -> None:
    cases = build_balanced_classifier_benchmark(DEVELOPMENT, CALIBRATION)

    labels = Counter(derive_classifier_label(case) for case in cases)
    splits = Counter(case.split for case in cases)
    assert len(cases) == 600
    assert set(labels.values()) == {100}
    assert splits == {"development": 450, "calibration": 150}
    assert len({case.case_id for case in cases}) == 600


def test_v2_capture_requires_explicit_remote_permission(tmp_path: Path, monkeypatch) -> None:
    _configure_inhouse(monkeypatch)

    with pytest.raises(RemoteModelsNotAllowedError):
        run_v2_classifier_capture(
            config=OpenAIModelConfig(classifier_model=INHOUSE_LLM_MODEL),
            development_cases_path=DEVELOPMENT,
            calibration_cases_path=CALIBRATION,
            corpus_path=CORPUS,
            output_path=tmp_path / "predictions.jsonl",
            manifest_path=tmp_path / "manifest.json",
            classifier=EchoClassifier(),
            limit_cases=1,
        )


def test_v2_capture_resumes_and_writes_safe_manifest(tmp_path: Path, monkeypatch) -> None:
    _configure_inhouse(monkeypatch)
    output = tmp_path / "predictions.jsonl"
    manifest = tmp_path / "manifest.json"
    classifier = EchoClassifier()
    config = OpenAIModelConfig(
        classifier_model=INHOUSE_LLM_MODEL,
        allow_remote_models=True,
    )

    first = run_v2_classifier_capture(
        config=config,
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        corpus_path=CORPUS,
        output_path=output,
        manifest_path=manifest,
        classifier=classifier,
        limit_cases=3,
        captured_at="2026-07-18T12:00:00Z",
    )
    second_classifier = EchoClassifier()
    second = run_v2_classifier_capture(
        config=config,
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        corpus_path=CORPUS,
        output_path=output,
        manifest_path=manifest,
        classifier=second_classifier,
        limit_cases=3,
        captured_at="2026-07-18T12:05:00Z",
    )

    assert len(classifier.calls) == 3
    assert second_classifier.calls == []
    assert first["completed_cases"] == 3
    assert second["resumed_cases"] == 3
    assert second["endpoint_host"] == "learning-services4.fokus.fraunhofer.de"
    assert second["request_policy"] == {
        "timeout_seconds": 90.0,
        "max_transport_retries": 1,
    }
    assert second["split_case_counts"] == {"development": 3}
    serialized = manifest.read_text(encoding="utf-8")
    assert "fixture-key" not in serialized
    assert "https://" not in serialized
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3


def test_v2_capture_can_checkpoint_bounded_concurrent_batches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure_inhouse(monkeypatch)
    classifier = ConcurrentClassifier(workers=4)

    manifest = run_v2_classifier_capture(
        config=OpenAIModelConfig(
            classifier_model=INHOUSE_LLM_MODEL,
            allow_remote_models=True,
        ),
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        corpus_path=CORPUS,
        output_path=tmp_path / "predictions.jsonl",
        manifest_path=tmp_path / "manifest.json",
        classifier=classifier,
        limit_cases=4,
        max_concurrency=4,
    )

    assert classifier.max_active == 4
    assert manifest["max_concurrency"] == 4
    assert manifest["completed_cases"] == 4
    assert len((tmp_path / "predictions.jsonl").read_text().splitlines()) == 4


def test_v2_capture_checkpoints_workers_in_completion_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure_inhouse(monkeypatch)
    selected = build_balanced_classifier_benchmark(DEVELOPMENT, CALIBRATION)[:2]

    def fake_capture(case, _classifier, *, provider):
        time.sleep(0.05 if case.case_id == selected[0].case_id else 0.01)
        return ClassifierPrediction(
            case_id=case.case_id,
            predicted_label="safe",
            confidence=0.9,
            provider=provider,
            model=INHOUSE_LLM_MODEL,
        )

    monkeypatch.setattr("guardrails_llm.inhouse_experiment._capture_one", fake_capture)
    output = tmp_path / "predictions.jsonl"

    run_v2_classifier_capture(
        config=OpenAIModelConfig(
            classifier_model=INHOUSE_LLM_MODEL,
            allow_remote_models=True,
        ),
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        corpus_path=CORPUS,
        output_path=output,
        manifest_path=tmp_path / "manifest.json",
        classifier=EchoClassifier(),
        limit_cases=2,
        max_concurrency=2,
    )

    first_row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert first_row["case_id"] == selected[1].case_id


def test_v2_capture_rejects_manifest_from_different_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure_inhouse(monkeypatch)
    output = tmp_path / "predictions.jsonl"
    manifest = tmp_path / "manifest.json"
    config = OpenAIModelConfig(
        classifier_model=INHOUSE_LLM_MODEL,
        allow_remote_models=True,
    )
    run_v2_classifier_capture(
        config=config,
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        corpus_path=CORPUS,
        output_path=output,
        manifest_path=manifest,
        classifier=EchoClassifier(),
        limit_cases=2,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["configuration_fingerprint"] = "changed"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="configuration does not match"):
        run_v2_classifier_capture(
            config=config,
            development_cases_path=DEVELOPMENT,
            calibration_cases_path=CALIBRATION,
            corpus_path=CORPUS,
            output_path=output,
            manifest_path=manifest,
            classifier=EchoClassifier(),
            limit_cases=2,
        )


def test_v2_classifier_evaluation_reports_splits_and_quality_gates(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    cases = build_balanced_classifier_benchmark(DEVELOPMENT, CALIBRATION)
    predictions.write_text(
        "".join(
            json.dumps(
                asdict(
                    ClassifierPrediction(
                        case_id=case.case_id,
                        predicted_label=derive_classifier_label(case),
                        confidence=0.99,
                    )
                )
            )
            + "\n"
            for case in cases
        ),
        encoding="utf-8",
    )

    report = evaluate_v2_classifier_capture(
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        predictions_path=predictions,
    )

    assert report["combined"]["summary"]["total"] == 600
    assert report["development"]["summary"]["total"] == 450
    assert report["calibration"]["summary"]["total"] == 150
    assert report["quality_gates"]["all_passed"] is True
    assert report["quality_gates"]["safe_false_positive_rate"] == 0.0


def test_prepare_inhouse_bge_indexes_only_development_and_calibration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure_inhouse(monkeypatch)
    manifest_path = tmp_path / "bge-manifest.json"
    embedder = FakeBgeEmbedder()

    manifest = prepare_inhouse_bge(
        config=OpenAIModelConfig(
            embedding_model="BAAI/bge-m3",
            allow_remote_models=True,
        ),
        development_cases_path=DEVELOPMENT,
        calibration_cases_path=CALIBRATION,
        corpus_path=CORPUS,
        policy_path=ROOT / "data" / "guardrail_policy_bge_m3.toml",
        index_dir=tmp_path / "chroma",
        cache_path=tmp_path / "cache.jsonl",
        manifest_path=manifest_path,
        embedder=embedder,
    )

    assert manifest["split_case_counts"] == {
        "development": 1200,
        "calibration": 400,
    }
    assert "holdout" not in manifest["split_sha256"]
    assert manifest["models"]["embedding"] == "BAAI/bge-m3"
    assert manifest["index"]["chunks"] > 0
    assert manifest["retrieval_evidence_threshold"] is None
    assert manifest["retrieval_routes"] == [ACADEMIC_INTEGRITY_RETRIEVAL_QUERY]
    assert manifest_path.exists()
    assert ACADEMIC_INTEGRITY_RETRIEVAL_QUERY in {
        text for batch in embedder.calls for text in batch
    }
