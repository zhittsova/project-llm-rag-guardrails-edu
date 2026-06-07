from __future__ import annotations

import json
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from .pipeline import LearningAssistant


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    category: str
    question: str
    should_answer: bool
    expected_trigger: str | None = None
    required_terms: list[str] | None = None
    forbidden_terms: list[str] | None = None


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


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(EvalCase(**json.loads(line)))
    return cases


def run_evaluation(assistant: LearningAssistant, cases: list[EvalCase]) -> list[EvalResult]:
    results: list[EvalResult] = []
    for case in cases:
        response = assistant.answer(case.question)
        answered = bool(response.citations)
        trigger_ok = case.expected_trigger is None or case.expected_trigger in response.guard_triggers
        answer_ok = answered if case.should_answer else not answered
        required_ok = _contains_all(response.answer, case.required_terms or [])
        forbidden_ok = not _contains_any(response.answer, case.forbidden_terms or [])
        passed = answer_ok and trigger_ok and required_ok and forbidden_ok
        results.append(
            EvalResult(
                case_id=case.case_id,
                category=case.category,
                should_answer=case.should_answer,
                answered=answered,
                passed=passed,
                triggers=response.guard_triggers,
                citations=response.citations,
                latency_ms=response.latency_ms,
                answer=response.answer,
            )
        )
    return results


def summarize(results: list[EvalResult]) -> dict[str, object]:
    total = len(results)
    passed = sum(result.passed for result in results)
    by_category: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = by_category.setdefault(result.category, {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += int(result.passed)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "by_category": by_category,
        "avg_latency_ms": round(sum(result.latency_ms for result in results) / total, 2) if total else 0.0,
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
