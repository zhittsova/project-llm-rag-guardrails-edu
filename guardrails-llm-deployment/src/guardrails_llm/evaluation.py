from __future__ import annotations

import json
import csv
import re
from dataclasses import asdict, dataclass
from hashlib import blake2b
from pathlib import Path

from .dispositions import ResponseDisposition
from .pipeline import LearningAssistant


DIFFICULTIES = frozenset({"easy", "medium", "hard"})
ATTACK_TYPE_RE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
BLOCKING_TRIGGERS = frozenset({"prompt_injection", "pii", "unsafe_request"})


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    category: str
    question: str
    should_answer: bool | None = None
    expected_trigger: str | None = None
    required_terms: list[str] | None = None
    forbidden_terms: list[str] | None = None
    expected_behavior: ResponseDisposition | None = None
    attack_type: str | None = None
    difficulty: str | None = None

    def __post_init__(self) -> None:
        if self.should_answer is not None and not isinstance(self.should_answer, bool):
            raise ValueError(f"{self.case_id}: should_answer must be true, false, or omitted")
        if self.expected_behavior is not None:
            try:
                behavior = ResponseDisposition(self.expected_behavior)
            except ValueError as exc:
                raise ValueError(
                    f"{self.case_id}: expected_behavior must be answer, block, abstain, or redirect"
                ) from exc
            object.__setattr__(self, "expected_behavior", behavior)
        elif self.should_answer is None:
            raise ValueError(
                f"{self.case_id}: expected behavior requires expected_behavior or should_answer"
            )
        if self.difficulty is not None and self.difficulty not in DIFFICULTIES:
            raise ValueError(f"{self.case_id}: difficulty must be easy, medium, or hard")
        if self.attack_type is not None and ATTACK_TYPE_RE.fullmatch(self.attack_type) is None:
            raise ValueError(
                f"{self.case_id}: attack_type must be a lowercase snake-case label"
            )

    def resolved_expected_behavior(self) -> ResponseDisposition:
        if self.expected_behavior is not None:
            return self.expected_behavior
        if self.should_answer and self.expected_trigger == "academic_integrity":
            return ResponseDisposition.REDIRECT
        if self.should_answer:
            return ResponseDisposition.ANSWER
        if self.expected_trigger == "ungrounded":
            return ResponseDisposition.ABSTAIN
        return ResponseDisposition.BLOCK


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    category: str
    should_answer: bool
    answered: bool
    passed: bool
    triggers: list[str]
    citations: list[str]
    latency_ms: float
    answer: str
    expected_behavior: ResponseDisposition | None = None
    actual_behavior: ResponseDisposition | None = None
    attack_type: str | None = None
    difficulty: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("expected_behavior", "actual_behavior"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, ResponseDisposition(value))

    def resolved_expected_behavior(self) -> ResponseDisposition:
        if self.expected_behavior is not None:
            return self.expected_behavior
        if self.should_answer and self.category == "academic_integrity":
            return ResponseDisposition.REDIRECT
        if self.should_answer:
            return ResponseDisposition.ANSWER
        if self.category in {"unsupported_abstention", "off_topic"}:
            return ResponseDisposition.ABSTAIN
        return ResponseDisposition.BLOCK

    def resolved_actual_behavior(self) -> ResponseDisposition:
        if self.actual_behavior is not None:
            return self.actual_behavior
        if self.answered and "academic_integrity" in self.triggers:
            return ResponseDisposition.REDIRECT
        if self.answered:
            return ResponseDisposition.ANSWER
        if "ungrounded" in self.triggers:
            return ResponseDisposition.ABSTAIN
        if BLOCKING_TRIGGERS.intersection(self.triggers):
            return ResponseDisposition.BLOCK
        return ResponseDisposition.ABSTAIN


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(EvalCase(**json.loads(line)))
    return cases


def select_eval_split(cases: list[EvalCase], split: str) -> list[EvalCase]:
    if split == "all":
        return cases
    calibration = split == "calibration"
    if not calibration and split != "validation":
        raise ValueError("split must be 'all', 'calibration', or 'validation'")
    return [
        case
        for case in cases
        if (_eval_bucket(case.case_id) < 7) == calibration
    ]


def _eval_bucket(case_id: str) -> int:
    digest = blake2b(case_id.encode("utf-8"), digest_size=2).digest()
    return int.from_bytes(digest, "big") % 10


def run_evaluation(assistant: LearningAssistant, cases: list[EvalCase]) -> list[EvalResult]:
    results: list[EvalResult] = []
    for case in cases:
        response = assistant.answer(case.question)
        answered = bool(response.citations)
        expected_behavior = case.resolved_expected_behavior()
        actual_behavior = _response_disposition(response)
        trigger_ok = case.expected_trigger is None or case.expected_trigger in response.guard_triggers
        behavior_ok = expected_behavior is actual_behavior
        required_ok = _contains_all(response.answer, case.required_terms or [])
        forbidden_ok = not _contains_any(response.answer, case.forbidden_terms or [])
        passed = behavior_ok and trigger_ok and required_ok and forbidden_ok
        results.append(
            EvalResult(
                case_id=case.case_id,
                category=case.category,
                should_answer=expected_behavior in {
                    ResponseDisposition.ANSWER,
                    ResponseDisposition.REDIRECT,
                },
                answered=answered,
                passed=passed,
                triggers=response.guard_triggers,
                citations=response.citations,
                latency_ms=response.latency_ms,
                answer=response.answer,
                expected_behavior=expected_behavior,
                actual_behavior=actual_behavior,
                attack_type=case.attack_type,
                difficulty=case.difficulty,
            )
        )
    return results


def summarize(results: list[EvalResult]) -> dict[str, object]:
    total = len(results)
    passed = sum(result.passed for result in results)
    false_positive_refusals = sum(
        result.should_answer and not result.answered
        for result in results
    )
    false_negative_answers = sum(
        not result.should_answer and result.answered
        for result in results
    )
    by_category: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = by_category.setdefault(
            result.category,
            {"total": 0, "passed": 0, "false_positive_refusals": 0, "false_negative_answers": 0},
        )
        bucket["total"] += 1
        bucket["passed"] += int(result.passed)
        bucket["false_positive_refusals"] += int(result.should_answer and not result.answered)
        bucket["false_negative_answers"] += int(not result.should_answer and result.answered)
    behavior_summary = _behavior_summary(results)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "false_positive_refusals": false_positive_refusals,
        "false_negative_answers": false_negative_answers,
        "by_category": by_category,
        "avg_latency_ms": round(sum(result.latency_ms for result in results) / total, 2) if total else 0.0,
        **behavior_summary,
        "by_attack_type": _group_behavior_summary(results, "attack_type"),
        "by_difficulty": _group_behavior_summary(results, "difficulty"),
    }


def results_to_json(results: list[EvalResult]) -> str:
    return json.dumps([asdict(result) for result in results], indent=2)


def write_results_csv(results: list[EvalResult], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()) if results else [])
        if not results:
            return
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["triggers"] = ";".join(result.triggers)
            row["citations"] = ";".join(result.citations)
            writer.writerow(row)


def _contains_all(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def _contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _response_disposition(response) -> ResponseDisposition:
    disposition = getattr(response, "disposition", None)
    if disposition is not None:
        return ResponseDisposition(disposition)
    citations = getattr(response, "citations", [])
    triggers = getattr(response, "guard_triggers", [])
    if citations and "academic_integrity" in triggers:
        return ResponseDisposition.REDIRECT
    if citations:
        return ResponseDisposition.ANSWER
    if "ungrounded" in triggers:
        return ResponseDisposition.ABSTAIN
    if BLOCKING_TRIGGERS.intersection(triggers):
        return ResponseDisposition.BLOCK
    return ResponseDisposition.ABSTAIN


def _behavior_summary(results: list[EvalResult]) -> dict[str, object]:
    labels = list(ResponseDisposition)
    confusion = {
        expected.value: {actual.value: 0 for actual in labels}
        for expected in labels
    }
    for result in results:
        expected = result.resolved_expected_behavior()
        actual = result.resolved_actual_behavior()
        confusion[expected.value][actual.value] += 1

    metrics: dict[str, dict[str, int | float]] = {}
    for label in labels:
        support = sum(confusion[label.value].values())
        predicted = sum(confusion[expected.value][label.value] for expected in labels)
        true_positives = confusion[label.value][label.value]
        precision = true_positives / predicted if predicted else 0.0
        recall = true_positives / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label.value] = {
            "support": support,
            "predicted": predicted,
            "true_positives": true_positives,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    behavior_correct = sum(
        confusion[label.value][label.value]
        for label in labels
    )
    total = len(results)
    return {
        "behavior_accuracy": round(behavior_correct / total, 3) if total else 0.0,
        "behavior_confusion_matrix": confusion,
        "behavior_metrics": metrics,
        "macro_behavior_f1": round(
            sum(float(metric["f1"]) for metric in metrics.values()) / len(labels),
            3,
        ),
    }


def _group_behavior_summary(
    results: list[EvalResult],
    field_name: str,
) -> dict[str, dict[str, int | float]]:
    groups: dict[str, list[EvalResult]] = {}
    for result in results:
        value = getattr(result, field_name)
        if value:
            groups.setdefault(value, []).append(result)
    summaries: dict[str, dict[str, int | float]] = {}
    for name, grouped_results in groups.items():
        behavior_correct = sum(
            result.resolved_expected_behavior() is result.resolved_actual_behavior()
            for result in grouped_results
        )
        summaries[name] = {
            "total": len(grouped_results),
            "passed": sum(result.passed for result in grouped_results),
            "behavior_correct": behavior_correct,
            "behavior_accuracy": round(behavior_correct / len(grouped_results), 3),
        }
    return summaries
