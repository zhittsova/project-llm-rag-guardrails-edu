from __future__ import annotations

import json
import os
from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from .confidence_intervals import bootstrap_confidence_intervals
from .e2e_capture import _quality_gates
from .evaluation import EvalResult, summarize
from .final_evidence import FinalEvidenceError, write_calibration_evidence


DETERMINISTIC_TECHNIQUES = {
    "baseline": "baseline",
    "normalized_regex_guardrails": "regex_only_with_shared_controls",
    "fuzzy_plus_shared_controls": "fuzzy_only_with_shared_controls",
    "similarity_plus_shared_controls": "bge_similarity_with_shared_controls",
    "hybrid_policy_guardrails": "deterministic_hybrid",
}
MODEL_SCENARIOS = ("qwen_classifier_only", "complete_inhouse_hybrid")
PRIMARY_GATE_KEYS = (
    "complete_capture",
    "behavior_accuracy_passed",
    "macro_f1_passed",
    "answer_recall_passed",
    "block_recall_passed",
    "abstain_recall_passed",
    "redirect_recall_passed",
    "safe_false_refusal_passed",
    "supported_answer_precision_passed",
    "citation_entailment_precision_passed",
)
FAILURE_STAGES = (
    "expected_policy_document_not_retrieved",
    "answerability_rejected_with_expected_document_present",
    "entailment_rejected_unsupported_extra_claims",
    "other",
)


def refresh_calibration_evidence(
    *,
    deterministic_path: Path,
    deterministic_manifest_path: Path,
    model_evaluation_path: Path,
    model_manifest_path: Path,
    model_capture_path: Path,
    deterministic_output: Path,
    model_output: Path,
    failure_output: Path,
    final_json_output: Path,
    final_markdown_output: Path,
) -> dict[str, object]:
    deterministic_raw = _load_object(deterministic_path)
    deterministic_manifest = _load_object(deterministic_manifest_path)
    model_manifest = _load_object(model_manifest_path)
    capture_rows = _load_jsonl(model_capture_path)

    _validate_model_capture(model_manifest, capture_rows)
    _validate_deterministic_capture(
        deterministic_path,
        deterministic_manifest,
        model_manifest,
    )
    model_evaluation = _evaluate_model_capture(
        capture_rows,
        model_manifest=model_manifest,
        source_path=model_capture_path,
    )
    deterministic_report = _build_deterministic_report(
        deterministic_raw,
        deterministic_manifest=deterministic_manifest,
        model_manifest=model_manifest,
    )
    model_report = _build_model_report(
        model_evaluation,
        model_manifest=model_manifest,
        source_capture_path=model_capture_path,
    )
    failure_report = _build_failure_report(
        capture_rows,
        model_manifest=model_manifest,
        source_path=model_capture_path,
    )

    with TemporaryDirectory(prefix="guardrails-evidence-") as temp_dir:
        stage = Path(temp_dir)
        staged_evaluation = stage / "model-evaluation.json"
        staged_deterministic = stage / "deterministic.json"
        staged_model = stage / "model.json"
        staged_failure = stage / "failures.json"
        staged_final_json = stage / "final.json"
        staged_final_markdown = stage / "final.md"
        _write_json(staged_evaluation, model_evaluation)
        _write_json(staged_deterministic, deterministic_report)
        _write_json(staged_model, model_report)
        _write_json(staged_failure, failure_report)
        report = write_calibration_evidence(
            deterministic_path=staged_deterministic,
            model_path=staged_model,
            failure_path=staged_failure,
            output_json=staged_final_json,
            output_markdown=staged_final_markdown,
        )
        for staged, target in (
            (staged_evaluation, model_evaluation_path),
            (staged_deterministic, deterministic_output),
            (staged_model, model_output),
            (staged_failure, failure_output),
            (staged_final_json, final_json_output),
            (staged_final_markdown, final_markdown_output),
        ):
            _publish_file(staged, target)
    return report


def _build_deterministic_report(
    raw: dict[str, object],
    *,
    deterministic_manifest: dict[str, object],
    model_manifest: dict[str, object],
) -> dict[str, object]:
    techniques = {}
    for source_name, output_name in DETERMINISTIC_TECHNIQUES.items():
        summary = _require_mapping(raw, source_name)
        _require_case_count(summary, source_name)
        techniques[output_name] = _compact_metrics(
            summary,
            confidence_intervals=summary.get("confidence_intervals"),
        )
    return {
        "schema_version": 2,
        "evidence_scope": "calibration_only",
        "dataset_version": _require_string(model_manifest, "dataset_version"),
        "dataset_manifest_sha256": _require_sha256(
            model_manifest, "dataset_manifest_sha256"
        ),
        "calibration_split_sha256": _require_sha256(
            model_manifest, "calibration_split_sha256"
        ),
        "source_report_sha256": deterministic_manifest["source_report_sha256"],
        "cases": 400,
        "cases_per_disposition": 100,
        "holdout_used": False,
        "techniques": techniques,
        "interpretation": (
            "Deterministic techniques were replayed on the repaired 400-case "
            "calibration split with the common BGE-M3 retrieval index."
        ),
    }


def _evaluate_model_capture(
    rows: list[dict[str, object]],
    *,
    model_manifest: dict[str, object],
    source_path: Path,
) -> dict[str, object]:
    report: dict[str, object] = {}
    selected_case_ids = model_manifest.get("selected_case_ids")
    if not isinstance(selected_case_ids, list) or not all(
        isinstance(case_id, str) for case_id in selected_case_ids
    ):
        raise FinalEvidenceError("model manifest has invalid selected calibration case IDs")
    case_order = {case_id: index for index, case_id in enumerate(selected_case_ids)}
    for scenario in MODEL_SCENARIOS:
        results = [
            EvalResult(**row["result"])
            for row in sorted(
                (row for row in rows if row["scenario"] == scenario),
                key=lambda row: case_order[str(row["case_id"])],
            )
        ]
        summary = summarize(results)
        report[scenario] = {
            "expected_cases": 400,
            "successful_cases": len(results),
            "capture_failures": 0,
            "missing_cases": 400 - len(results),
            "metrics_on_successful_cases": summary,
            "quality_gates": _quality_gates(
                summary,
                failures=0,
                missing=400 - len(results),
            ),
            "confidence_intervals": bootstrap_confidence_intervals(results),
        }
    report["_provenance"] = {
        "schema_version": 1,
        "evidence_scope": "calibration_only",
        "dataset_version": _require_string(model_manifest, "dataset_version"),
        "dataset_manifest_sha256": _require_sha256(
            model_manifest, "dataset_manifest_sha256"
        ),
        "calibration_split_sha256": _require_sha256(
            model_manifest, "calibration_split_sha256"
        ),
        "configuration_fingerprint": _require_sha256(
            model_manifest, "configuration_fingerprint"
        ),
        "source_capture_sha256": _file_sha256(source_path),
        "holdout_used": False,
    }
    return report


def _build_model_report(
    evaluation: dict[str, object],
    *,
    model_manifest: dict[str, object],
    source_capture_path: Path,
) -> dict[str, object]:
    scenarios = {}
    scenario_gates = {}
    for scenario_name in MODEL_SCENARIOS:
        scenario = _require_mapping(evaluation, scenario_name)
        if (
            scenario.get("expected_cases") != 400
            or scenario.get("successful_cases") != 400
            or scenario.get("capture_failures") != 0
            or scenario.get("missing_cases") != 0
        ):
            raise FinalEvidenceError(
                f"{scenario_name} must contain 400 successful calibration cases"
            )
        summary = _require_mapping(scenario, "metrics_on_successful_cases")
        _require_case_count(summary, scenario_name)
        scenarios[scenario_name] = _compact_metrics(
            summary,
            confidence_intervals=scenario.get("confidence_intervals"),
        )
        scenario_gates[scenario_name] = _require_mapping(
            scenario, "quality_gates"
        )

    hybrid_gates = scenario_gates["complete_inhouse_hybrid"]
    primary_passed = all(hybrid_gates.get(key) is True for key in PRIMARY_GATE_KEYS)
    expected_document_passed = (
        hybrid_gates.get("expected_document_citation_precision_passed") is True
    )
    return {
        "schema_version": 2,
        "evidence_scope": "inhouse_common_split_calibration",
        "dataset_version": _require_string(model_manifest, "dataset_version"),
        "dataset_manifest_sha256": _require_sha256(
            model_manifest, "dataset_manifest_sha256"
        ),
        "calibration_split_sha256": _require_sha256(
            model_manifest, "calibration_split_sha256"
        ),
        "runtime_config_sha256": _require_sha256(
            model_manifest, "runtime_config_sha256"
        ),
        "source_report_sha256": _payload_sha256(evaluation),
        "source_capture_sha256": _file_sha256(source_capture_path),
        "cases_per_scenario": 400,
        "cases_per_disposition": 100,
        "holdout_used": False,
        "models": _require_mapping(model_manifest, "models"),
        "configuration": {
            "prompt_versions": _require_mapping(
                model_manifest, "prompt_versions"
            ),
            "retrieval": _require_mapping(model_manifest, "retrieval"),
            "thresholds": _require_mapping(model_manifest, "thresholds"),
            "runtime_config_sha256": model_manifest["runtime_config_sha256"],
        },
        "capture": {
            "expected_runs": model_manifest["expected_runs"],
            "completed_runs": model_manifest["completed_runs"],
            "failed_runs": model_manifest["failed_runs"],
            "configuration_fingerprint": _require_sha256(
                model_manifest, "configuration_fingerprint"
            ),
        },
        "scenarios": scenarios,
        "quality_gate_interpretation": {
            "agreed_primary_gates_passed": primary_passed,
            "extra_expected_document_citation_gate_passed": (
                expected_document_passed
            ),
        },
        "limitations": [
            "Calibration evidence is not a frozen-holdout result.",
            "Expected-document labels remain generated until independent human review.",
            "LLM-judge agreement remains unavailable until human adjudication.",
        ],
    }


def _build_failure_report(
    rows: list[dict[str, object]],
    *,
    model_manifest: dict[str, object],
    source_path: Path,
) -> dict[str, object]:
    failed_results = []
    for row in rows:
        if row.get("scenario") != "complete_inhouse_hybrid":
            continue
        result = row.get("result")
        if row.get("status") == "success" and isinstance(result, dict):
            if result.get("passed") is False:
                failed_results.append(result)

    stages = Counter({stage: 0 for stage in FAILURE_STAGES})
    cases_by_stage = {stage: [] for stage in FAILURE_STAGES}
    for result in failed_results:
        stage = _failure_stage(result)
        stages[stage] += 1
        cases_by_stage[stage].append(str(result.get("case_id")))

    transitions = Counter(
        f"{result.get('expected_behavior')}_to_{result.get('actual_behavior')}"
        for result in failed_results
    )
    attack_types = Counter(
        str(result.get("attack_type") or "unknown") for result in failed_results
    )
    return {
        "schema_version": 2,
        "evidence_scope": "complete_inhouse_hybrid_calibration",
        "dataset_version": _require_string(model_manifest, "dataset_version"),
        "source_capture_sha256": _file_sha256(source_path),
        "cases": 400,
        "failed_cases": len(failed_results),
        "case_ids": sorted(str(result.get("case_id")) for result in failed_results),
        "failure_dispositions": dict(sorted(transitions.items())),
        "attack_types": dict(sorted(attack_types.items())),
        "stage_counts": {stage: stages[stage] for stage in FAILURE_STAGES},
        "cases_by_stage": {
            stage: sorted(cases_by_stage[stage]) for stage in FAILURE_STAGES
        },
        "holdout_used": False,
        "interpretation": (
            "Stage assignment is derived from expected-document retrieval, "
            "answerability, and entailment fields in the captured result."
        ),
    }


def _failure_stage(result: dict[str, object]) -> str:
    expected_docs = {
        str(value) for value in result.get("expected_doc_ids", []) if value
    }
    retrieved_docs = {
        str(value) for value in result.get("retrieved_doc_ids", []) if value
    }
    if expected_docs and expected_docs.isdisjoint(retrieved_docs):
        return "expected_policy_document_not_retrieved"
    unsupported_claims = result.get("unsupported_claims")
    if result.get("grounding_supported") is False and isinstance(
        unsupported_claims, list
    ) and unsupported_claims:
        return "entailment_rejected_unsupported_extra_claims"
    if (
        result.get("expected_behavior") == "answer"
        and result.get("actual_behavior") == "abstain"
        and expected_docs.intersection(retrieved_docs)
    ):
        return "answerability_rejected_with_expected_document_present"
    return "other"


def _compact_metrics(
    summary: dict[str, object],
    *,
    confidence_intervals: object,
) -> dict[str, object]:
    behavior_metrics = _require_mapping(summary, "behavior_metrics")
    accuracy = _require_number(summary, "behavior_accuracy")
    total = int(_require_number(summary, "total"))
    return {
        "correct": int(summary.get("passed", round(accuracy * total))),
        "behavior_accuracy": accuracy,
        "macro_f1": _require_number(summary, "macro_behavior_f1"),
        "answer_recall": _disposition_recall(behavior_metrics, "answer"),
        "block_recall": _disposition_recall(behavior_metrics, "block"),
        "abstain_recall": _disposition_recall(behavior_metrics, "abstain"),
        "redirect_recall": _disposition_recall(behavior_metrics, "redirect"),
        "safe_false_refusal_rate": summary.get("safe_false_refusal_rate"),
        "false_unsafe_answer_rate": summary.get("false_unsafe_answer_rate"),
        "retrieval_recall_at_3": summary.get("retrieval_recall_at_3"),
        "supported_answer_precision": summary.get("supported_answer_precision"),
        "supported_answer_total": summary.get("supported_answer_total"),
        "citation_entailment_precision": summary.get(
            "citation_entailment_precision"
        ),
        "citation_entailment_total": summary.get("citation_entailment_total"),
        "citation_entailment_scope": summary.get("citation_entailment_scope"),
        "expected_document_citation_precision": summary.get(
            "expected_document_citation_precision"
        ),
        "expected_document_citation_total": summary.get(
            "expected_document_citation_total"
        ),
        "claim_support_rate": summary.get("claim_support_rate"),
        "claim_support_total": summary.get("claim_support_total"),
        "row_accuracy_95ci": _accuracy_interval(confidence_intervals, "row"),
        "family_accuracy_95ci": _accuracy_interval(
            confidence_intervals, "family"
        ),
    }


def _accuracy_interval(payload: object, level: str) -> list[float] | None:
    if not isinstance(payload, dict):
        return None
    group = payload.get(level)
    metrics = group.get("metrics") if isinstance(group, dict) else None
    accuracy = metrics.get("behavior_accuracy") if isinstance(metrics, dict) else None
    if not isinstance(accuracy, dict):
        return None
    lower = accuracy.get("lower")
    upper = accuracy.get("upper")
    if not isinstance(lower, int | float) or not isinstance(upper, int | float):
        return None
    return [float(lower), float(upper)]


def _disposition_recall(
    behavior_metrics: dict[str, object], disposition: str
) -> float:
    metrics = _require_mapping(behavior_metrics, disposition)
    return _require_number(metrics, "recall")


def _validate_model_capture(
    manifest: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    if (
        manifest.get("status") != "complete"
        or manifest.get("expected_runs") != 800
        or manifest.get("completed_runs") != 800
        or manifest.get("failed_runs") != 0
        or manifest.get("holdout_used") is not False
    ):
        raise FinalEvidenceError(
            "model capture must contain 800 successful runs without holdout use"
        )
    counts = Counter(str(row.get("scenario")) for row in rows)
    if len(rows) != 800 or any(counts[scenario] != 400 for scenario in MODEL_SCENARIOS):
        raise FinalEvidenceError(
            "model capture rows must contain 400 runs for each model scenario"
        )
    run_keys = set()
    for row in rows:
        result = row.get("result")
        scenario = row.get("scenario")
        case_id = row.get("case_id")
        if (
            row.get("status") != "success"
            or row.get("error") is not None
            or not isinstance(result, dict)
            or result.get("case_id") != case_id
            or result.get("split") != "calibration"
        ):
            raise FinalEvidenceError(
                "model capture rows must be successful and contain matching results"
            )
        run_keys.add((scenario, case_id))
    if len(run_keys) != 800:
        raise FinalEvidenceError("model capture rows contain duplicate run keys")
    selected_run_keys = manifest.get("selected_run_keys")
    if not isinstance(selected_run_keys, list) or len(selected_run_keys) != 800:
        raise FinalEvidenceError("model manifest has no selected calibration run keys")
    if not all(
        isinstance(run_key, list)
        and len(run_key) == 2
        and run_key[0] in MODEL_SCENARIOS
        and isinstance(run_key[1], str)
        for run_key in selected_run_keys
    ):
        raise FinalEvidenceError(
            "model manifest has invalid selected calibration run keys"
        )
    expected_keys = {(run_key[0], run_key[1]) for run_key in selected_run_keys}
    if len(expected_keys) != 800 or run_keys != expected_keys:
        raise FinalEvidenceError(
            "model capture does not match selected calibration run keys"
        )


def _validate_deterministic_capture(
    source_path: Path,
    manifest: dict[str, object],
    model_manifest: dict[str, object],
) -> None:
    if (
        manifest.get("evidence_scope") != "calibration_only"
        or manifest.get("cases") != 400
        or manifest.get("holdout_used") is not False
    ):
        raise FinalEvidenceError(
            "deterministic manifest must describe 400 calibration cases without holdout use"
        )
    if manifest.get("dataset_version") != model_manifest.get("dataset_version"):
        raise FinalEvidenceError("deterministic and model dataset versions differ")
    for key in ("dataset_manifest_sha256", "calibration_split_sha256"):
        if _require_sha256(manifest, key) != _require_sha256(model_manifest, key):
            raise FinalEvidenceError(
                "deterministic and model calibration provenance differ"
            )
    if _require_sha256(manifest, "source_report_sha256") != _file_sha256(source_path):
        raise FinalEvidenceError("deterministic capture hash does not match its manifest")


def _require_case_count(summary: dict[str, object], label: str) -> None:
    if summary.get("total") != 400:
        raise FinalEvidenceError(f"{label} must contain 400 calibration cases")


def _load_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalEvidenceError(f"cannot read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise FinalEvidenceError(f"{path} must contain a JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FinalEvidenceError(f"cannot read JSONL capture: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FinalEvidenceError(
                f"invalid JSONL row at {path}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise FinalEvidenceError(
                f"invalid JSONL row at {path}:{line_number}"
            )
        rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _publish_file(staged: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(staged.read_bytes())
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _require_mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise FinalEvidenceError(f"missing or invalid object: {key}")
    return value


def _require_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise FinalEvidenceError(f"missing or invalid string: {key}")
    return value


def _require_sha256(payload: dict[str, object], key: str) -> str:
    value = _require_string(payload, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise FinalEvidenceError(f"missing or invalid SHA-256 value: {key}")
    return value


def _require_number(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FinalEvidenceError(f"missing or invalid number: {key}")
    return float(value)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _payload_sha256(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, indent=2) + "\n"
    return sha256(serialized.encode("utf-8")).hexdigest()
