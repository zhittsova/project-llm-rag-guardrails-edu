import json
import subprocess
import sys
from pathlib import Path

from guardrails_llm.dispositions import ResponseDisposition
from guardrails_llm.evaluation import load_eval_cases


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE = DATA / "eval_cases_milestone3.jsonl"
LABELED = DATA / "eval_cases_milestone3_labeled_v1.jsonl"
CANONICAL = DATA / "eval_cases_milestone3_holdout_v3.jsonl"
SCRIPT = ROOT / "scripts" / "annotate_eval_cases_milestone3.py"

EXPLICIT_FIELDS = {
    "case_id",
    "category",
    "question",
    "expected_behavior",
    "attack_type",
    "difficulty",
    "expected_trigger",
    "required_terms",
    "forbidden_terms",
}


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_labeled_development_set_has_explicit_unique_labels() -> None:
    assert LABELED.exists(), "versioned labeled evaluation set is missing"

    rows = _load_jsonl(LABELED)
    cases = load_eval_cases(LABELED)

    assert len(rows) == 165
    assert len({row["case_id"] for row in rows}) == 165
    assert all(set(row) == EXPLICIT_FIELDS for row in rows)
    assert {case.expected_behavior for case in cases} == set(ResponseDisposition)
    assert {case.difficulty for case in cases} == {"easy", "medium", "hard"}
    assert all(case.attack_type for case in cases)


def test_labeled_development_set_preserves_source_cases_and_order() -> None:
    source = _load_jsonl(SOURCE)
    labeled = _load_jsonl(LABELED)
    preserved_fields = {
        "case_id",
        "category",
        "question",
        "expected_trigger",
        "required_terms",
        "forbidden_terms",
    }

    assert [row["case_id"] for row in labeled] == [row["case_id"] for row in source]
    assert [
        {key: row[key] for key in preserved_fields}
        for row in labeled
    ] == [
        {key: row[key] for key in preserved_fields}
        for row in source
    ]


def test_labeled_development_set_uses_frozen_labels_for_shared_cases() -> None:
    labeled_by_id = {
        row["case_id"]: row
        for row in _load_jsonl(LABELED)
    }
    canonical = _load_jsonl(CANONICAL)
    label_fields = {"expected_behavior", "attack_type", "difficulty"}
    shared = [row for row in canonical if row["case_id"] in labeled_by_id]

    assert len(shared) == 39
    for row in shared:
        labeled = labeled_by_id[row["case_id"]]
        assert {key: labeled[key] for key in label_fields} == {
            key: row[key] for key in label_fields
        }

    assert labeled_by_id["unsupported-006"]["expected_behavior"] == "block"
    assert labeled_by_id["unsupported-006"]["attack_type"] == "pii_request"


def test_annotation_script_reproduces_versioned_file(tmp_path: Path) -> None:
    output = tmp_path / "labeled.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(SOURCE),
            "--output",
            str(output),
            "--canonical-labels",
            str(CANONICAL),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == LABELED.read_bytes()
    assert not SOURCE.with_suffix(".jsonl.bak").exists()


def test_annotation_script_refuses_to_overwrite_source(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(SOURCE.read_bytes())
    before = source.read_bytes()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(source),
            "--output",
            str(source),
            "--canonical-labels",
            str(CANONICAL),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert source.read_bytes() == before
