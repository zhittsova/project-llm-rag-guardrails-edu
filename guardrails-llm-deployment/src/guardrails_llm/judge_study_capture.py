from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from statistics import median
from threading import Lock, local
from time import perf_counter

from .evaluation import load_eval_cases
from .model_calibration import JudgePrediction, load_judge_predictions
from .model_capture import eval_result_from_payload
from .model_config import (
    OpenAIModelConfig,
    ensure_openai_api_key,
    ensure_remote_models_allowed,
    openai_request_policy,
)
from .model_profiles import ensure_inhouse_endpoint
from .openai_models import JUDGE_PROMPT_VERSION, OpenAIJudge


def run_judge_study_capture(
    *,
    config: OpenAIModelConfig,
    study_dir: Path,
    source_cases_path: Path,
    source_results_path: Path,
    output_path: Path,
    manifest_path: Path,
    max_concurrency: int = 1,
    retry_failures: bool = False,
    judge_split: str | None = None,
    limit_cases: int | None = None,
) -> dict[str, object]:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be greater than zero")
    if judge_split not in {None, "judge_calibration", "judge_validation"}:
        raise ValueError("judge_split must be judge_calibration or judge_validation")
    if limit_cases is not None and limit_cases < 1:
        raise ValueError("limit_cases must be greater than zero")
    ensure_remote_models_allowed(config)
    ensure_openai_api_key(config)
    endpoint_host = ensure_inhouse_endpoint(config.env_file)
    cases = {case.case_id: case for case in load_eval_cases(source_cases_path)}
    results = _load_source_results(source_results_path)
    mappings = _load_study_mappings(study_dir / "judge_study_mapping.jsonl")
    if judge_split is not None:
        mappings = [
            mapping
            for mapping in mappings
            if mapping["judge_split"] == judge_split
        ]
    if limit_cases is not None:
        mappings = mappings[:limit_cases]
    study_manifest_path = study_dir / "judge_study_manifest.json"
    configuration = {
        "schema_version": 1,
        "experiment": "milestone3_judge_study_capture",
        "provider": "openai_compatible",
        "endpoint_host": endpoint_host,
        "model": config.judge_model,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "request_policy": openai_request_policy(config),
        "study_manifest_sha256": _file_sha256(study_manifest_path),
        "source_cases_sha256": _file_sha256(source_cases_path),
        "source_results_sha256": _file_sha256(source_results_path),
        "selected_items": len(mappings),
        "judge_split": judge_split or "all",
        "case_limit": limit_cases,
        "max_concurrency": max_concurrency,
        "human_labels_used": False,
        "holdout_used": False,
    }
    fingerprint = _json_sha256(configuration)
    existing_manifest = _load_json(manifest_path) if manifest_path.exists() else None
    if existing_manifest is not None and existing_manifest.get(
        "configuration_fingerprint"
    ) != fingerprint:
        raise ValueError("existing judge capture manifest does not match this run")
    if output_path.exists() and existing_manifest is None:
        raise ValueError("judge capture output exists without its manifest")
    predictions = {
        prediction.case_id: prediction
        for prediction in (
            load_judge_predictions(output_path) if output_path.exists() else []
        )
    }
    expected_ids = {mapping["item_id"] for mapping in mappings}
    unknown = set(predictions) - expected_ids
    if unknown:
        raise ValueError("judge capture output contains unknown item IDs")
    if retry_failures:
        predictions = {
            item_id: prediction
            for item_id, prediction in predictions.items()
            if prediction.error is None
        }

    _validate_sources(mappings, cases, results)
    pending = [mapping for mapping in mappings if mapping["item_id"] not in predictions]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = Lock()
    worker_state = local()

    def capture(mapping: dict[str, str]) -> JudgePrediction:
        judge = getattr(worker_state, "judge", None)
        if judge is None:
            judge = OpenAIJudge(config)
            worker_state.judge = judge
        case = cases[mapping["source_case_id"]]
        result = eval_result_from_payload(
            results[mapping["scenario"]][mapping["source_case_id"]]
        )
        started_at = perf_counter()
        judgment = judge.judge(case, result)
        latency_ms = round((perf_counter() - started_at) * 1000, 3)
        error = next(
            (
                note
                for note in judgment.notes
                if note.startswith("llm_judge_error:")
            ),
            None,
        )
        if error is not None:
            return JudgePrediction(
                case_id=mapping["item_id"],
                grounded=None,
                privacy_safe=None,
                injection_safe=None,
                integrity_safe=None,
                refusal_appropriate=None,
                score=None,
                error=error,
                provider="openai_compatible",
                model=config.judge_model,
                latency_ms=latency_ms,
            )
        return JudgePrediction(
            case_id=mapping["item_id"],
            grounded=judgment.grounded,
            privacy_safe=judgment.privacy_safe,
            injection_safe=judgment.injection_safe,
            integrity_safe=judgment.integrity_safe,
            refusal_appropriate=judgment.refusal_appropriate,
            score=judgment.score,
            notes=judgment.notes,
            provider="openai_compatible",
            model=config.judge_model,
            latency_ms=latency_ms,
        )

    def store(prediction: JudgePrediction) -> None:
        with write_lock:
            predictions[prediction.case_id] = prediction
            _write_predictions(output_path, predictions.values())
            _write_manifest(
                manifest_path,
                configuration,
                fingerprint,
                predictions.values(),
            )

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {executor.submit(capture, mapping): mapping for mapping in pending}
        for future in as_completed(futures):
            mapping = futures[future]
            try:
                prediction = future.result()
            except Exception as exc:
                prediction = JudgePrediction(
                    case_id=mapping["item_id"],
                    grounded=None,
                    privacy_safe=None,
                    injection_safe=None,
                    integrity_safe=None,
                    refusal_appropriate=None,
                    score=None,
                    error=f"capture_error:{type(exc).__name__}",
                    provider="openai_compatible",
                    model=config.judge_model,
                    latency_ms=0.0,
                )
            store(prediction)

    return _write_manifest(
        manifest_path,
        configuration,
        fingerprint,
        predictions.values(),
    )


def _load_study_mappings(path: Path) -> list[dict[str, str]]:
    mappings = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        required = {
            "item_id",
            "judge_split",
            "source_case_id",
            "parent_case_id",
            "scenario",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("judge study mapping has invalid fields")
        mapping = {key: str(value) for key, value in payload.items()}
        if mapping["item_id"] in seen:
            raise ValueError("judge study mapping contains duplicate item IDs")
        seen.add(mapping["item_id"])
        mappings.append(mapping)
    return sorted(mappings, key=lambda mapping: mapping["item_id"])


def _load_source_results(path: Path) -> dict[str, dict[str, dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source results must be a scenario object")
    indexed = {}
    for scenario, rows in payload.items():
        if not isinstance(scenario, str) or not isinstance(rows, list):
            raise ValueError("source results must map scenarios to result lists")
        indexed[scenario] = {
            str(row["case_id"]): row
            for row in rows
            if isinstance(row, dict) and "case_id" in row
        }
        if len(indexed[scenario]) != len(rows):
            raise ValueError(f"scenario {scenario} has invalid or duplicate results")
    return indexed


def _validate_sources(mappings, cases, results) -> None:
    for mapping in mappings:
        if mapping["source_case_id"] not in cases:
            raise ValueError(f"unknown source case: {mapping['source_case_id']}")
        scenario = results.get(mapping["scenario"])
        if scenario is None or mapping["source_case_id"] not in scenario:
            raise ValueError(
                f"missing source result for {mapping['scenario']}:{mapping['source_case_id']}"
            )


def _write_predictions(path: Path, predictions) -> None:
    ordered = sorted(predictions, key=lambda prediction: prediction.case_id)
    with path.open("w", encoding="utf-8") as handle:
        for prediction in ordered:
            handle.write(json.dumps(asdict(prediction), separators=(",", ":")))
            handle.write("\n")


def _write_manifest(path, configuration, fingerprint, predictions) -> dict[str, object]:
    predictions = list(predictions)
    successful = [prediction for prediction in predictions if prediction.error is None]
    latencies = [float(prediction.latency_ms or 0.0) for prediction in predictions]
    manifest = {
        **configuration,
        "configuration_fingerprint": fingerprint,
        "completed_items": len(predictions),
        "successful_predictions": len(successful),
        "failed_predictions": len(predictions) - len(successful),
        "structured_response_validity": (
            round(len(successful) / len(predictions), 3) if predictions else 0.0
        ),
        "complete": len(predictions) == int(configuration["selected_items"]),
        "p50_latency_ms": round(median(latencies), 3) if latencies else None,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _json_sha256(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
