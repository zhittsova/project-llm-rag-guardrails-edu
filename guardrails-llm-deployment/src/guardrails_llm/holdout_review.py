from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .dispositions import ResponseDisposition
from .evaluation import load_eval_cases
from .evaluation_dataset import apply_holdout_annotations, cohens_kappa


REVIEWERS = ("reviewer_a", "reviewer_b")
REVIEW_FIELDS = (
    "expected_behavior",
    "evidence_available",
    "expected_doc_ids",
    "required_claims",
)


@dataclass(frozen=True)
class HoldoutReviewAnnotation:
    case_id: str
    annotator_id: str
    expected_behavior: str | None
    evidence_available: bool | None
    expected_doc_ids: list[str] | None
    required_claims: list[str] | None
    rationale: str

    def validate(self, *, complete: bool) -> None:
        if not self.case_id.strip():
            raise ValueError("holdout annotation case_id must be non-empty")
        if complete and not self.annotator_id.strip():
            raise ValueError(f"{self.case_id}: annotator_id must be non-empty")
        if self.expected_behavior is not None:
            try:
                ResponseDisposition(self.expected_behavior)
            except ValueError as exc:
                raise ValueError(
                    f"{self.case_id}: expected_behavior must be answer, block, "
                    "abstain, or redirect"
                ) from exc
        elif complete:
            raise ValueError(f"{self.case_id}: expected_behavior must be selected")
        if self.evidence_available is not None and not isinstance(
            self.evidence_available, bool
        ):
            raise ValueError(
                f"{self.case_id}: evidence_available must be true, false, or null"
            )
        if complete and not isinstance(self.evidence_available, bool):
            raise ValueError(f"{self.case_id}: evidence_available must be selected")
        _validate_string_list(
            self.case_id,
            "expected_doc_ids",
            self.expected_doc_ids,
            complete=complete,
        )
        _validate_string_list(
            self.case_id,
            "required_claims",
            self.required_claims,
            complete=complete,
        )
        if complete:
            _validate_evidence_fields(
                self.case_id,
                evidence_available=bool(self.evidence_available),
                expected_doc_ids=self.expected_doc_ids or [],
                required_claims=self.required_claims or [],
            )
        if not isinstance(self.rationale, str) or (
            complete and not self.rationale.strip()
        ):
            requirement = "a non-empty string" if complete else "a string"
            raise ValueError(f"{self.case_id}: rationale must be {requirement}")


def prepare_holdout_review(
    *,
    cases_path: Path,
    output_dir: Path,
    expected_cases: int = 400,
) -> dict[str, object]:
    cases = load_eval_cases(cases_path)
    if len(cases) != expected_cases:
        raise ValueError(
            f"holdout review requires {expected_cases} cases; found {len(cases)}"
        )
    if any(case.split != "holdout" for case in cases):
        raise ValueError("holdout review preparation requires holdout cases only")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("holdout review cases contain duplicate case_id values")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_paths = [
        output_dir / "holdout_review_items.jsonl",
        output_dir / "holdout_reviewer_a.jsonl",
        output_dir / "holdout_reviewer_b.jsonl",
        output_dir / "holdout_review_manifest.json",
    ]
    if any(path.exists() for path in target_paths):
        raise ValueError("refusing to overwrite an existing holdout review study")

    items = [
        {
            "case_id": case.case_id,
            "question": case.question,
            "language": case.language,
        }
        for case in cases
    ]
    items_path = output_dir / "holdout_review_items.jsonl"
    _write_jsonl(items_path, items)
    for reviewer in REVIEWERS:
        _write_jsonl(
            output_dir / f"holdout_{reviewer}.jsonl",
            [_annotation_template(case.case_id) for case in cases],
        )

    manifest = {
        "schema_version": 1,
        "study": "milestone3_frozen_holdout_double_review",
        "cases": len(cases),
        "source_cases": str(cases_path.resolve()),
        "source_cases_sha256": _file_sha256(cases_path),
        "items_sha256": _file_sha256(items_path),
        "reviewers_must_be_independent": True,
        "generated_labels_are_hidden": True,
        "canonical_annotations_written_only_after_adjudication": True,
    }
    (output_dir / "holdout_review_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def holdout_review_status(study_dir: Path) -> dict[str, object]:
    manifest = _load_and_verify_manifest(study_dir)
    expected_ids = _load_item_ids(study_dir, manifest)
    reviewers: dict[str, object] = {}
    for reviewer in REVIEWERS:
        _annotations, summary = _load_review_file(
            study_dir / f"holdout_{reviewer}.jsonl",
            expected_ids=expected_ids,
            complete=False,
        )
        reviewers[reviewer] = summary
    return {
        "cases": len(expected_ids),
        "source_holdout_unchanged": True,
        "ready_to_reconcile": all(
            bool(summary["complete"])
            for summary in reviewers.values()
            if isinstance(summary, dict)
        ),
        "reviewers": reviewers,
    }


def reconcile_holdout_review(study_dir: Path) -> dict[str, object]:
    manifest = _load_and_verify_manifest(study_dir)
    items = _load_items(study_dir, manifest)
    expected_ids = set(items)
    reviewer_a, summary_a = _load_review_file(
        study_dir / "holdout_reviewer_a.jsonl",
        expected_ids=expected_ids,
        complete=True,
    )
    reviewer_b, summary_b = _load_review_file(
        study_dir / "holdout_reviewer_b.jsonl",
        expected_ids=expected_ids,
        complete=True,
    )
    _require_independent_reviewers(reviewer_a, reviewer_b)
    by_a = {annotation.case_id: annotation for annotation in reviewer_a}
    by_b = {annotation.case_id: annotation for annotation in reviewer_b}

    agreements = Counter()
    disagreements: list[dict[str, object]] = []
    for case_id in sorted(expected_ids):
        left = by_a[case_id]
        right = by_b[case_id]
        differing_fields = [
            field
            for field in REVIEW_FIELDS
            if getattr(left, field) != getattr(right, field)
        ]
        for field in REVIEW_FIELDS:
            agreements[field] += int(getattr(left, field) == getattr(right, field))
        if differing_fields:
            disagreements.append(
                {
                    **items[case_id],
                    "differing_fields": differing_fields,
                    "reviewer_a": asdict(left),
                    "reviewer_b": asdict(right),
                    "adjudicator_id": "",
                    "adjudicated_behavior": None,
                    "adjudicated_evidence_available": None,
                    "adjudicated_expected_doc_ids": None,
                    "adjudicated_required_claims": None,
                    "adjudication_notes": "",
                }
            )

    disagreement_path = study_dir / "holdout_review_disagreements.jsonl"
    _write_jsonl(disagreement_path, disagreements)
    total = len(expected_ids)
    exact_agreements = total - len(disagreements)
    report = {
        "schema_version": 1,
        "cases": total,
        "reviewer_a": summary_a,
        "reviewer_b": summary_b,
        "reviewer_a_sha256": _file_sha256(
            study_dir / "holdout_reviewer_a.jsonl"
        ),
        "reviewer_b_sha256": _file_sha256(
            study_dir / "holdout_reviewer_b.jsonl"
        ),
        "source_cases_sha256": manifest["source_cases_sha256"],
        "items_sha256": manifest["items_sha256"],
        "exact_agreements": exact_agreements,
        "exact_agreement_rate": round(exact_agreements / total, 3),
        "behavior_kappa": cohens_kappa(
            [by_a[case_id].expected_behavior or "" for case_id in sorted(expected_ids)],
            [by_b[case_id].expected_behavior or "" for case_id in sorted(expected_ids)],
        ),
        "evidence_kappa": cohens_kappa(
            [str(by_a[case_id].evidence_available) for case_id in sorted(expected_ids)],
            [str(by_b[case_id].evidence_available) for case_id in sorted(expected_ids)],
        ),
        "field_exact_agreement": {
            field: round(agreements[field] / total, 3) for field in REVIEW_FIELDS
        },
        "cases_requiring_adjudication": len(disagreements),
        "disagreements_sha256": _file_sha256(disagreement_path),
    }
    (study_dir / "holdout_review_agreement.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def finalize_holdout_review(
    *,
    study_dir: Path,
    output_path: Path,
    replace: bool = False,
) -> dict[str, object]:
    manifest = _load_and_verify_manifest(study_dir)
    _require_safe_output_path(
        output_path=output_path,
        study_dir=study_dir,
        source_path=Path(str(manifest["source_cases"])),
    )
    _verify_reconciled_inputs(study_dir, manifest)
    items = _load_items(study_dir, manifest)
    expected_ids = set(items)
    reviewer_a, _summary_a = _load_review_file(
        study_dir / "holdout_reviewer_a.jsonl",
        expected_ids=expected_ids,
        complete=True,
    )
    reviewer_b, _summary_b = _load_review_file(
        study_dir / "holdout_reviewer_b.jsonl",
        expected_ids=expected_ids,
        complete=True,
    )
    _require_independent_reviewers(reviewer_a, reviewer_b)
    by_a = {annotation.case_id: annotation for annotation in reviewer_a}
    by_b = {annotation.case_id: annotation for annotation in reviewer_b}
    disagreement_ids = {
        case_id
        for case_id in expected_ids
        if any(
            getattr(by_a[case_id], field) != getattr(by_b[case_id], field)
            for field in REVIEW_FIELDS
        )
    }
    adjudications = _load_adjudications(
        study_dir / "holdout_review_disagreements.jsonl",
        expected_ids=disagreement_ids,
    )

    canonical = []
    for case_id in sorted(expected_ids):
        left = by_a[case_id]
        right = by_b[case_id]
        if case_id in disagreement_ids:
            final = adjudications[case_id]
            adjudicator_id = str(final["adjudicator_id"])
            behavior = str(final["adjudicated_behavior"])
            evidence_available = bool(final["adjudicated_evidence_available"])
            expected_doc_ids = list(final["adjudicated_expected_doc_ids"])
            required_claims = list(final["adjudicated_required_claims"])
            notes = str(final["adjudication_notes"])
        else:
            adjudicator_id = "reviewer-consensus"
            behavior = str(left.expected_behavior)
            evidence_available = bool(left.evidence_available)
            expected_doc_ids = list(left.expected_doc_ids or [])
            required_claims = list(left.required_claims or [])
            notes = "Independent reviewers agreed on every annotation field."
        canonical.append(
            _canonical_annotation(
                left=left,
                right=right,
                adjudicator_id=adjudicator_id,
                behavior=behavior,
                evidence_available=evidence_available,
                expected_doc_ids=expected_doc_ids,
                required_claims=required_claims,
                notes=notes,
            )
        )

    cases = load_eval_cases(Path(str(manifest["source_cases"])))
    _updated, annotation_summary = apply_holdout_annotations(cases, canonical)
    if annotation_summary["ready_for_final_holdout"] is not True:
        raise ValueError("holdout review did not produce complete adjudicated annotations")
    if output_path.exists() and not replace:
        raise ValueError("refusing to replace existing canonical annotations")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, canonical)
    report = {
        "schema_version": 1,
        "cases": len(canonical),
        "ready_for_dataset_sealing": True,
        "source_cases_sha256": manifest["source_cases_sha256"],
        "annotations_path": str(output_path),
        "annotations_sha256": _file_sha256(output_path),
        "annotation_summary": annotation_summary,
    }
    (study_dir / "holdout_review_finalization.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _annotation_template(case_id: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "annotator_id": "",
        "expected_behavior": None,
        "evidence_available": None,
        "expected_doc_ids": None,
        "required_claims": None,
        "rationale": "",
    }


def _load_review_file(
    path: Path,
    *,
    expected_ids: set[str],
    complete: bool,
) -> tuple[list[HoldoutReviewAnnotation], dict[str, object]]:
    annotations = [
        HoldoutReviewAnnotation(**row) for row in _load_jsonl(path)
    ]
    seen: set[str] = set()
    for annotation in annotations:
        annotation.validate(complete=complete)
        if annotation.case_id in seen:
            raise ValueError(f"duplicate holdout annotation case_id: {annotation.case_id}")
        seen.add(annotation.case_id)
    if seen != expected_ids:
        raise ValueError(
            "holdout review IDs do not match study items: "
            f"missing={len(expected_ids - seen)}, unknown={len(seen - expected_ids)}"
        )
    completed = sum(_annotation_is_complete(annotation) for annotation in annotations)
    return annotations, {
        "path": str(path),
        "total": len(annotations),
        "completed": completed,
        "remaining": len(annotations) - completed,
        "complete": completed == len(annotations),
    }


def _annotation_is_complete(annotation: HoldoutReviewAnnotation) -> bool:
    try:
        annotation.validate(complete=True)
    except ValueError:
        return False
    return True


def _require_independent_reviewers(
    reviewer_a: list[HoldoutReviewAnnotation],
    reviewer_b: list[HoldoutReviewAnnotation],
) -> None:
    annotators_a = {annotation.annotator_id for annotation in reviewer_a}
    annotators_b = {annotation.annotator_id for annotation in reviewer_b}
    if len(annotators_a) != 1 or len(annotators_b) != 1:
        raise ValueError("each holdout reviewer file must use exactly one annotator_id")
    if annotators_a == annotators_b:
        raise ValueError("holdout reviews must use different annotator identities")


def _load_adjudications(
    path: Path,
    *,
    expected_ids: set[str],
) -> dict[str, dict[str, object]]:
    if not path.exists() and expected_ids:
        raise ValueError("run holdout review reconciliation before finalization")
    rows = _load_jsonl(path) if path.exists() else []
    by_id = {str(row.get("case_id", "")): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != expected_ids:
        raise ValueError("adjudication IDs do not match holdout review disagreements")
    for case_id, row in by_id.items():
        adjudicator_id = row.get("adjudicator_id")
        behavior = row.get("adjudicated_behavior")
        evidence_available = row.get("adjudicated_evidence_available")
        expected_doc_ids = row.get("adjudicated_expected_doc_ids")
        required_claims = row.get("adjudicated_required_claims")
        notes = row.get("adjudication_notes")
        if not isinstance(adjudicator_id, str) or not adjudicator_id.strip():
            raise ValueError(f"{case_id}: adjudicator_id must be non-empty")
        try:
            ResponseDisposition(str(behavior))
        except ValueError as exc:
            raise ValueError(f"{case_id}: adjudicated_behavior is invalid") from exc
        if not isinstance(evidence_available, bool):
            raise ValueError(
                f"{case_id}: adjudicated_evidence_available must be true or false"
            )
        _validate_string_list(
            case_id,
            "adjudicated_expected_doc_ids",
            expected_doc_ids,
            complete=True,
        )
        _validate_string_list(
            case_id,
            "adjudicated_required_claims",
            required_claims,
            complete=True,
        )
        _validate_evidence_fields(
            case_id,
            evidence_available=evidence_available,
            expected_doc_ids=expected_doc_ids,
            required_claims=required_claims,
        )
        if not isinstance(notes, str) or not notes.strip():
            raise ValueError(f"{case_id}: adjudication_notes must be non-empty")
    return by_id


def _verify_reconciled_inputs(
    study_dir: Path,
    manifest: dict[str, Any],
) -> None:
    report_path = study_dir / "holdout_review_agreement.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("holdout review reconciliation must run before finalization") from exc
    if not isinstance(report, dict):
        raise ValueError("holdout review reconciliation report is invalid")
    expected_hashes = {
        "reviewer_a_sha256": _file_sha256(
            study_dir / "holdout_reviewer_a.jsonl"
        ),
        "reviewer_b_sha256": _file_sha256(
            study_dir / "holdout_reviewer_b.jsonl"
        ),
        "source_cases_sha256": manifest["source_cases_sha256"],
        "items_sha256": manifest["items_sha256"],
    }
    if any(report.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("holdout review inputs changed after reconciliation")


def _require_safe_output_path(
    *,
    output_path: Path,
    study_dir: Path,
    source_path: Path,
) -> None:
    protected_paths = {source_path.resolve()}
    protected_paths.update(
        (study_dir / name).resolve()
        for name in (
            "holdout_review_manifest.json",
            "holdout_review_items.jsonl",
            "holdout_reviewer_a.jsonl",
            "holdout_reviewer_b.jsonl",
            "holdout_review_agreement.json",
            "holdout_review_disagreements.jsonl",
        )
    )
    if output_path.resolve() in protected_paths:
        raise ValueError("output annotations cannot replace a protected holdout review input")


def _canonical_annotation(
    *,
    left: HoldoutReviewAnnotation,
    right: HoldoutReviewAnnotation,
    adjudicator_id: str,
    behavior: str,
    evidence_available: bool,
    expected_doc_ids: list[str],
    required_claims: list[str],
    notes: str,
) -> dict[str, object]:
    return {
        "case_id": left.case_id,
        "annotator_a_id": left.annotator_id,
        "annotator_b_id": right.annotator_id,
        "adjudicator_id": adjudicator_id,
        "annotator_a_behavior": left.expected_behavior,
        "annotator_b_behavior": right.expected_behavior,
        "adjudicated_behavior": behavior,
        "annotator_a_evidence_available": left.evidence_available,
        "annotator_b_evidence_available": right.evidence_available,
        "adjudicated_evidence_available": evidence_available,
        "annotator_a_expected_doc_ids": left.expected_doc_ids,
        "annotator_b_expected_doc_ids": right.expected_doc_ids,
        "adjudicated_expected_doc_ids": expected_doc_ids,
        "annotator_a_required_claims": left.required_claims,
        "annotator_b_required_claims": right.required_claims,
        "adjudicated_required_claims": required_claims,
        "adjudication_notes": notes,
    }


def _load_and_verify_manifest(study_dir: Path) -> dict[str, Any]:
    manifest_path = study_dir / "holdout_review_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot load holdout review manifest") from exc
    if not isinstance(manifest, dict):
        raise ValueError("holdout review manifest must be a JSON object")
    source_path = Path(str(manifest.get("source_cases", "")))
    expected_hash = manifest.get("source_cases_sha256")
    if not source_path.exists() or _file_sha256(source_path) != expected_hash:
        raise ValueError("source holdout has changed since review preparation")
    return manifest


def _load_item_ids(study_dir: Path, manifest: dict[str, Any]) -> set[str]:
    return set(_load_items(study_dir, manifest))


def _load_items(
    study_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, dict[str, object]]:
    path = study_dir / "holdout_review_items.jsonl"
    if _file_sha256(path) != manifest.get("items_sha256"):
        raise ValueError("holdout review items changed after preparation")
    rows = _load_jsonl(path)
    by_id = {str(row.get("case_id", "")): row for row in rows}
    if len(by_id) != len(rows) or len(by_id) != manifest.get("cases"):
        raise ValueError("holdout review items are incomplete or duplicated")
    return by_id


def _validate_string_list(
    case_id: str,
    field_name: str,
    value: object,
    *,
    complete: bool,
) -> None:
    if value is None:
        if complete:
            raise ValueError(f"{case_id}: {field_name} must be a list")
        return
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{case_id}: {field_name} must contain non-empty strings")


def _validate_evidence_fields(
    case_id: str,
    *,
    evidence_available: bool,
    expected_doc_ids: list[str],
    required_claims: list[str],
) -> None:
    if evidence_available and (not expected_doc_ids or not required_claims):
        raise ValueError(
            f"{case_id}: available evidence requires document IDs and required claims"
        )
    if not evidence_available and (expected_doc_ids or required_claims):
        raise ValueError(
            f"{case_id}: unavailable evidence requires empty document IDs and claims"
        )


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSONL from {path}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every JSONL row must be an object")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"cannot hash holdout review artifact: {path}") from exc
