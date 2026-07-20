from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any


class FinalEvidenceError(ValueError):
    """Raised when final evidence inputs violate the evaluation protocol."""


TECHNIQUE_ORDER = (
    "baseline",
    "regex_only_with_shared_controls",
    "fuzzy_only_with_shared_controls",
    "bge_similarity_with_shared_controls",
    "deterministic_hybrid",
    "qwen_classifier_only",
    "complete_inhouse_hybrid",
)

TECHNIQUE_NAMES = {
    "baseline": "Baseline RAG",
    "regex_only_with_shared_controls": "Normalized regex + metadata",
    "fuzzy_only_with_shared_controls": "Fuzzy + shared controls",
    "bge_similarity_with_shared_controls": "BGE similarity + shared controls",
    "deterministic_hybrid": "Deterministic hybrid policy",
    "qwen_classifier_only": "Qwen classifier scenario",
    "complete_inhouse_hybrid": "Complete in-house hybrid",
}

REQUIRED_SEALED_ARTIFACTS = frozenset(
    {
        "dataset_manifest",
        "calibration_evidence",
        "policy",
        "corpus",
        "index_manifest",
    }
)


def build_calibration_evidence(
    *,
    deterministic_path: Path,
    model_path: Path,
    failure_path: Path | None = None,
) -> dict[str, object]:
    deterministic = _load_object(deterministic_path)
    model = _load_object(model_path)
    if (
        deterministic.get("holdout_used") is not False
        or model.get("holdout_used") is not False
    ):
        raise FinalEvidenceError("final calibration evidence must not use the frozen holdout")
    if deterministic.get("evidence_scope") != "calibration_only" or model.get(
        "evidence_scope"
    ) not in {"calibration_only", "inhouse_common_split_calibration"}:
        raise FinalEvidenceError("final evidence inputs must have calibration_only scope")
    dataset_version = deterministic.get("dataset_version")
    if not dataset_version or dataset_version != model.get("dataset_version"):
        raise FinalEvidenceError("calibration inputs use different dataset versions")
    for provenance_key in (
        "dataset_manifest_sha256",
        "calibration_split_sha256",
    ):
        deterministic_hash = deterministic.get(provenance_key)
        model_hash = model.get(provenance_key)
        if (
            not _is_sha256(deterministic_hash)
            or not _is_sha256(model_hash)
            or deterministic_hash != model_hash
        ):
            raise FinalEvidenceError(
                "calibration inputs use different calibration provenance"
            )
    if deterministic.get("cases") != 400 or model.get("cases_per_scenario") != 400:
        raise FinalEvidenceError("final calibration comparison requires 400 cases per technique")
    if deterministic.get("cases_per_disposition") != 100 or model.get(
        "cases_per_disposition"
    ) != 100:
        raise FinalEvidenceError("final calibration comparison requires 100 cases per disposition")

    deterministic_techniques = _require_mapping(deterministic, "techniques")
    model_scenarios = _require_mapping(model, "scenarios")
    techniques: dict[str, object] = {}
    for technique in TECHNIQUE_ORDER:
        source = (
            model_scenarios
            if technique in {"qwen_classifier_only", "complete_inhouse_hybrid"}
            else deterministic_techniques
        )
        payload = source.get(technique)
        if not isinstance(payload, dict):
            raise FinalEvidenceError(f"missing calibration technique: {technique}")
        techniques[technique] = payload

    capture = _require_mapping(model, "capture")
    if (
        capture.get("expected_runs") != 800
        or capture.get("completed_runs") != 800
        or capture.get("failed_runs") != 0
    ):
        raise FinalEvidenceError("model-backed calibration capture is incomplete")
    fingerprint = capture.get("configuration_fingerprint")
    if not _is_sha256(fingerprint):
        raise FinalEvidenceError("model-backed calibration fingerprint is missing")

    gate_interpretation = _require_mapping(model, "quality_gate_interpretation")
    report = {
        "schema_version": 1,
        "evidence_scope": "calibration_only",
        "dataset_version": dataset_version,
        "cases": 400,
        "cases_per_disposition": 100,
        "holdout_used": False,
        "configuration_fingerprint": fingerprint,
        "configuration": model.get("configuration", {}),
        "models": model.get("models", {}),
        "techniques": techniques,
        "primary_quality_gates_passed": bool(
            gate_interpretation.get("agreed_primary_gates_passed")
        ),
        "expected_document_diagnostic_passed": bool(
            gate_interpretation.get("extra_expected_document_citation_gate_passed")
        ),
        "limitations": list(model.get("limitations", [])),
    }
    if failure_path is not None:
        failures = _load_object(failure_path)
        if failures.get("holdout_used") is not False:
            raise FinalEvidenceError("failure analysis must not use the frozen holdout")
        if (
            failures.get("dataset_version") != dataset_version
            or failures.get("cases") != 400
        ):
            raise FinalEvidenceError("failure analysis does not match the calibration dataset")
        if failures.get("evidence_scope") != "complete_inhouse_hybrid_calibration":
            raise FinalEvidenceError("failure analysis must describe complete hybrid calibration")
        report["failure_analysis"] = failures
    return report


def render_calibration_report(report: dict[str, object]) -> str:
    techniques = _require_mapping(report, "techniques")
    hybrid = _require_mapping(techniques, "complete_inhouse_hybrid")
    lines = [
        "# Final Calibration Evidence",
        "",
        "Calibration evidence only: the frozen holdout remains unopened.",
        "",
        "## Common-Split Technique Comparison",
        "",
        "| Technique | Correct | Accuracy | Macro-F1 | False refusal | Unsafe answered |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for technique_id in TECHNIQUE_ORDER:
        metrics = _require_mapping(techniques, technique_id)
        accuracy = _metric(metrics, "behavior_accuracy")
        correct = metrics.get("correct", round(accuracy * int(report["cases"])))
        lines.append(
            "| "
            + " | ".join(
                (
                    TECHNIQUE_NAMES[technique_id],
                    f"{correct}/{report['cases']}",
                    _format_metric(accuracy),
                    _format_metric(_metric(metrics, "macro_f1")),
                    _format_metric(metrics.get("safe_false_refusal_rate")),
                    _format_metric(metrics.get("false_unsafe_answer_rate")),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Accuracy Confidence Intervals",
            "",
            "| Technique | Row-level accuracy 95% CI | Family-level accuracy 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for technique_id in TECHNIQUE_ORDER:
        metrics = _require_mapping(techniques, technique_id)
        lines.append(
            f"| {TECHNIQUE_NAMES[technique_id]} | "
            f"{_format_interval(metrics.get('row_accuracy_95ci'))} | "
            f"{_format_interval(metrics.get('family_accuracy_95ci'))} |"
        )
    lines.extend(
        [
            "",
            "## Complete Hybrid",
            "",
            f"- Correct behavior: {hybrid['correct']}/{report['cases']}.",
            f"- Macro-F1: {_format_metric(hybrid['macro_f1'])}.",
            f"- Answer recall: {_format_metric(hybrid['answer_recall'])}.",
            f"- Block, abstain, and redirect recall: {_format_metric(hybrid['block_recall'])}, "
            f"{_format_metric(hybrid['abstain_recall'])}, and "
            f"{_format_metric(hybrid['redirect_recall'])}.",
            "- Supported-answer precision: "
            f"{_format_metric(hybrid['supported_answer_precision'])}.",
            "- Citation-entailment precision "
            f"({_format_scope(hybrid.get('citation_entailment_scope'))}): "
            f"{_format_metric(hybrid['citation_entailment_precision'])} "
            f"across {hybrid.get('citation_entailment_total', 'n/a')} citations.",
            f"- Expected-document citation precision: "
            f"{_format_metric(hybrid['expected_document_citation_precision'])}.",
            "",
            "The expected-document citation diagnostic remains failed and visible. Generated "
            "expected-document labels require independent human review before this metric can be "
            "treated as authoritative.",
            "",
            "The citation-entailment figure is runtime verifier consistency, not an independent "
            "human entailment judgment.",
            "",
            "## Evidence Boundary",
            "",
            "Human judge agreement, independently adjudicated source labels, and the final "
            "400-case frozen-holdout run are not yet available. Calibration results guide "
            "engineering decisions but are not a final generalization claim.",
            "",
        ]
    )
    failures = report.get("failure_analysis")
    if isinstance(failures, dict):
        stage_counts = _require_mapping(failures, "stage_counts")
        failed_cases = int(failures["failed_cases"])
        transition_payload = failures.get("failure_dispositions")
        if isinstance(transition_payload, dict):
            transitions = transition_payload
        elif failures.get("failure_disposition") in {
            "false_abstention",
            "answer_to_abstain",
        }:
            transitions = {"answer_to_abstain": failed_cases}
        else:
            transitions = {"unclassified": failed_cases}
        only_false_abstentions = transitions == {"answer_to_abstain": failed_cases}
        lines.extend(
            [
                "## Remaining Complete-Hybrid Failures",
                "",
                (
                    f"All {failed_cases} behavior errors are false abstentions:"
                    if only_false_abstentions
                    else f"The {failed_cases} behavior errors include these transitions:"
                ),
                "",
            ]
        )
        if not only_false_abstentions:
            for transition, count in sorted(transitions.items()):
                expected, separator, actual = transition.partition("_to_")
                label = f"{expected} -> {actual}" if separator else transition
                lines.append(f"- {label}: {count}.")
            lines.append("")
        stage_lines = (
            (
                "expected_policy_document_not_retrieved",
                "retrieval misses",
            ),
            (
                "answerability_rejected_with_expected_document_present",
                "answerability abstentions with the expected document present",
            ),
            (
                "entailment_rejected_unsupported_extra_claims",
                "entailment rejects caused by unsupported extra claims",
            ),
            ("other", "unclassified stage failures"),
        )
        for stage, label in stage_lines:
            count = int(stage_counts.get(stage, 0))
            if count:
                lines.append(f"- {count} {label}.")
        lines.append("")
    return "\n".join(lines)


def _format_scope(value: object) -> str:
    if not isinstance(value, str) or not value:
        return "scope unspecified"
    return value.replace("_", "-")


def write_calibration_evidence(
    *,
    deterministic_path: Path,
    model_path: Path,
    failure_path: Path | None = None,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, object]:
    report = build_calibration_evidence(
        deterministic_path=deterministic_path,
        model_path=model_path,
        failure_path=failure_path,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_markdown.write_text(render_calibration_report(report), encoding="utf-8")
    return report


def seal_runtime_configuration(
    *,
    dataset_manifest_path: Path,
    calibration_report_path: Path,
    policy_path: Path,
    corpus_path: Path,
    index_manifest_path: Path,
    output_path: Path,
) -> dict[str, object]:
    dataset_manifest = _load_object(dataset_manifest_path)
    calibration_report = _load_object(calibration_report_path)
    annotation_summary = dataset_manifest.get("annotation_summary")
    if dataset_manifest.get("holdout_frozen") is not True:
        raise FinalEvidenceError("dataset manifest does not describe a frozen holdout")
    if (
        dataset_manifest.get("annotation_sealed") is not True
        or dataset_manifest.get("holdout_review_status") != "adjudicated"
        or dataset_manifest.get("holdout_reviewed_cases") != 400
        or not isinstance(annotation_summary, dict)
        or annotation_summary.get("double_labeled_cases") != 400
        or annotation_summary.get("adjudicated_cases") != 400
        or annotation_summary.get("ready_for_final_holdout") is not True
    ):
        raise FinalEvidenceError("holdout annotations must be reviewed and sealed before freeze")
    if calibration_report.get("holdout_used") is not False:
        raise FinalEvidenceError("configuration freeze cannot use holdout-derived evidence")
    dataset_version = dataset_manifest.get("dataset_version")
    if not dataset_version or dataset_version != calibration_report.get("dataset_version"):
        raise FinalEvidenceError("dataset and calibration report versions do not match")
    if not _is_sha256(calibration_report.get("configuration_fingerprint")):
        raise FinalEvidenceError("calibration configuration fingerprint is missing")

    artifact_paths = {
        "dataset_manifest": dataset_manifest_path,
        "calibration_evidence": calibration_report_path,
        "policy": policy_path,
        "corpus": corpus_path,
        "index_manifest": index_manifest_path,
    }
    artifacts = {
        name: {
            "path": str(path),
            "sha256": _file_sha256(path),
        }
        for name, path in artifact_paths.items()
    }
    sealed_payload = {
        "schema_version": 1,
        "sealed": True,
        "holdout_used": False,
        "dataset_version": dataset_version,
        "source_calibration_fingerprint": calibration_report["configuration_fingerprint"],
        "models": calibration_report.get("models", {}),
        "configuration": calibration_report.get("configuration", {}),
        "artifacts": artifacts,
    }
    fingerprint = sha256(
        json.dumps(sealed_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {**sealed_payload, "configuration_fingerprint": fingerprint}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def assess_final_readiness(
    *,
    dataset_manifest: dict[str, object],
    judge_report: dict[str, object],
    selected_judge_model: str,
    calibration_report: dict[str, object],
    configuration_manifest: dict[str, object],
) -> dict[str, object]:
    annotation_summary = dataset_manifest.get("annotation_summary")
    if not isinstance(annotation_summary, dict):
        annotation_summary = {}
    models = judge_report.get("models")
    selected_model = models.get(selected_judge_model) if isinstance(models, dict) else None
    validation = None
    if isinstance(selected_model, dict):
        splits = selected_model.get("splits")
        if isinstance(splits, dict):
            validation = splits.get("judge_validation")
    validation_summary = validation.get("summary") if isinstance(validation, dict) else None
    quality_gates = validation.get("quality_gates") if isinstance(validation, dict) else None
    artifacts = configuration_manifest.get("artifacts")
    artifact_hashes_complete = (
        isinstance(artifacts, dict)
        and REQUIRED_SEALED_ARTIFACTS.issubset(artifacts)
        and all(
            isinstance(entry, dict) and _is_sha256(entry.get("sha256"))
            for entry in artifacts.values()
        )
    )

    checks = {
        "holdout_is_frozen": dataset_manifest.get("holdout_frozen") is True,
        "holdout_annotations_sealed": (
            dataset_manifest.get("annotation_sealed") is True
            and dataset_manifest.get("holdout_review_status") == "adjudicated"
            and dataset_manifest.get("holdout_reviewed_cases") == 400
            and annotation_summary.get("double_labeled_cases") == 400
            and annotation_summary.get("adjudicated_cases") == 400
            and annotation_summary.get("ready_for_final_holdout") is True
        ),
        "human_labels_are_ground_truth": (
            judge_report.get("human_labels_are_ground_truth") is True
        ),
        "judge_validation_complete": (
            isinstance(validation_summary, dict)
            and validation_summary.get("total") == 200
            and validation_summary.get("evaluated_predictions") == 200
            and validation_summary.get("parse_failures") == 0
            and validation_summary.get("missing_predictions") == 0
        ),
        "judge_validation_gates_passed": (
            isinstance(quality_gates, dict) and quality_gates.get("all_passed") is True
        ),
        "calibration_complete_without_holdout": (
            calibration_report.get("cases") == 400
            and calibration_report.get("holdout_used") is False
            and calibration_report.get("primary_quality_gates_passed") is True
        ),
        "runtime_configuration_sealed": (
            configuration_manifest.get("sealed") is True
            and configuration_manifest.get("holdout_used") is False
            and _is_sha256(configuration_manifest.get("configuration_fingerprint"))
        ),
        "artifact_hashes_complete": artifact_hashes_complete,
        "dataset_versions_match": (
            dataset_manifest.get("dataset_version")
            == calibration_report.get("dataset_version")
            == configuration_manifest.get("dataset_version")
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "ready": not failed_checks,
        "selected_judge_model": selected_judge_model,
        "checks": checks,
        "failed_checks": failed_checks,
    }


def assess_final_readiness_from_files(
    *,
    dataset_manifest_path: Path,
    judge_report_path: Path,
    selected_judge_model: str,
    calibration_report_path: Path,
    configuration_manifest_path: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    report = assess_final_readiness(
        dataset_manifest=_load_object(dataset_manifest_path),
        judge_report=_load_object(judge_report_path),
        selected_judge_model=selected_judge_model,
        calibration_report=_load_object(calibration_report_path),
        configuration_manifest=_load_object(configuration_manifest_path),
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalEvidenceError(f"cannot load JSON object from {path}") from exc
    if not isinstance(payload, dict):
        raise FinalEvidenceError(f"{path}: expected a JSON object")
    return payload


def _require_mapping(payload: dict[str, object], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise FinalEvidenceError(f"missing object field: {key}")
    return value


def _metric(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float):
        raise FinalEvidenceError(f"missing numeric metric: {key}")
    return float(value)


def _format_metric(value: object) -> str:
    if not isinstance(value, int | float):
        return "—"
    return f"{float(value):.3f}"


def _format_interval(value: object) -> str:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int | float) for item in value)
    ):
        return "—"
    return f"[{float(value[0]):.4f}, {float(value[1]):.4f}]"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _file_sha256(path: Path) -> str:
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise FinalEvidenceError(f"cannot hash required artifact: {path}") from exc
