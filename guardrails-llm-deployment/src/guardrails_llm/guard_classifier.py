from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .guardrail_policy import GuardrailPolicy


@dataclass(frozen=True)
class GuardClassification:
    label: str
    confidence: float
    explanation: str = ""


class GuardClassifier(Protocol):
    model_name: str

    def classify(self, text: str) -> GuardClassification:
        ...


def should_use_model_classifier(question: str, policy: GuardrailPolicy, triggers: list[str]) -> bool:
    if triggers:
        return False
    lowered = question.lower()
    risky_terms = (
        "answer key",
        "assignment",
        "class marks",
        "confidential",
        "developer message",
        "exam",
        "hidden prompt",
        "instructions",
        "personal",
        "private",
        "roster",
        "student records",
        "submission",
        "system prompt",
    )
    if any(term in lowered for term in risky_terms):
        return True
    return policy.has_near_similarity_trigger(question)
