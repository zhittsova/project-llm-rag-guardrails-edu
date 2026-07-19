from __future__ import annotations

import re
from dataclasses import dataclass, field

from .guardrail_policy import GuardrailPolicy


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    triggers: list[str] = field(default_factory=list)
    message: str | None = None


def input_guard(
    question: str,
    policy: GuardrailPolicy | None = None,
    *,
    include_similarity: bool = True,
) -> GuardResult:
    policy = policy or GuardrailPolicy.default()
    triggers = policy.input_deterministic_triggers(question)
    if include_similarity:
        triggers.extend(policy.input_similarity_triggers(question))
        triggers = list(dict.fromkeys(triggers))

    if policy.blocking_triggers.intersection(triggers):
        return GuardResult(
            allowed=False,
            triggers=triggers,
            message=policy.input_block_message,
        )
    return GuardResult(allowed=True, triggers=triggers)


def output_guard(
    answer: str,
    citations: list[str],
    question_triggers: list[str],
    policy: GuardrailPolicy | None = None,
) -> GuardResult:
    policy = policy or GuardrailPolicy.default()
    triggers: list[str] = []
    if policy.require_citations and not citations:
        triggers.append("ungrounded")
    triggers.extend(policy.output_triggers(answer))
    if "academic_integrity" in question_triggers and _looks_like_full_solution(answer):
        triggers.append("academic_integrity")
    triggers = list(dict.fromkeys(triggers))

    if triggers:
        if triggers == ["ungrounded"]:
            return GuardResult(
                allowed=False,
                triggers=triggers,
                message=policy.ungrounded_message,
            )
        return GuardResult(
            allowed=False,
            triggers=triggers,
            message=policy.output_block_message,
        )
    return GuardResult(allowed=True, triggers=[])


def make_integrity_safe(question: str, policy: GuardrailPolicy | None = None) -> str:
    return (policy or GuardrailPolicy.default()).integrity_safe_message


def sanitize_untrusted_context(text: str, policy: GuardrailPolicy | None = None) -> str:
    policy = policy or GuardrailPolicy.default()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    safe_sentences = [sentence for sentence in sentences if not policy.has_context_injection(sentence)]
    return " ".join(safe_sentences).strip()


def _looks_like_full_solution(answer: str) -> bool:
    lowered = answer.lower()
    return "final answer" in lowered or "submit" in lowered or len(answer.split()) > 180
