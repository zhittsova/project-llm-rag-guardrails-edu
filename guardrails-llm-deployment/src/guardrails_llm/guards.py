from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    triggers: list[str] = field(default_factory=list)
    message: str | None = None


INJECTION_PATTERNS = [
    re.compile(r"\bignore (all )?(previous|above|system|developer) instructions\b", re.I),
    re.compile(r"\breveal\b.*\b(system prompt|hidden prompt|developer message)\b", re.I),
    re.compile(r"\bdisregard\b.*\binstructions\b", re.I),
    re.compile(r"\byou are now\b", re.I),
    re.compile(r"<\s*script\b", re.I),
]

PII_PATTERNS = [
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    re.compile(r"\b(?:\+?\d[\d .()-]{7,}\d)\b"),
    re.compile(r"\b(?:student|matriculation|ssn|passport)\s*(?:id|number|no\.?)\s*[:#]?\s*[a-z0-9-]{4,}\b", re.I),
]

SENSITIVE_DATA_REQUEST_PATTERNS = [
    re.compile(r"\b(private|confidential|internal)\b.*\b(roster|note|data|record|file)\b", re.I),
    re.compile(r"\b(student|professor|teacher|staff)\b.*\b(email addresses?|phone numbers?|grades?|ids?|accommodations?)\b", re.I),
    re.compile(r"\bwhat\b.*\b(email addresses?|phone numbers?|grades?|ids?)\b", re.I),
]

CHEATING_PATTERNS = [
    re.compile(r"\b(write|solve|complete|do)\b.*\b(my|the)\b.*\b(assignment|homework|exam|quiz|graded)\b", re.I),
    re.compile(r"\bgive me\b.*\b(final answer|full solution|submission)\b", re.I),
    re.compile(r"\bplagiar", re.I),
]

UNSAFE_PATTERNS = [
    re.compile(r"\b(self-harm|suicide|kill myself)\b", re.I),
    re.compile(r"\b(make|build)\b.*\b(weapon|bomb|malware)\b", re.I),
]


def input_guard(question: str) -> GuardResult:
    triggers: list[str] = []
    if _matches_any(INJECTION_PATTERNS, question):
        triggers.append("prompt_injection")
    if _matches_any(PII_PATTERNS, question):
        triggers.append("pii")
    if _matches_any(SENSITIVE_DATA_REQUEST_PATTERNS, question):
        triggers.append("pii")
    if _matches_any(CHEATING_PATTERNS, question):
        triggers.append("academic_integrity")
    if _matches_any(UNSAFE_PATTERNS, question):
        triggers.append("unsafe_request")

    blocking = {"prompt_injection", "pii", "unsafe_request"}
    if blocking.intersection(triggers):
        return GuardResult(
            allowed=False,
            triggers=triggers,
            message="I cannot process this request because it may contain unsafe instructions or sensitive data.",
        )
    return GuardResult(allowed=True, triggers=triggers)


def output_guard(answer: str, citations: list[str], question_triggers: list[str]) -> GuardResult:
    triggers: list[str] = []
    if not citations:
        triggers.append("ungrounded")
    if _matches_any(INJECTION_PATTERNS, answer):
        triggers.append("prompt_injection_output")
    if _matches_any(PII_PATTERNS, answer):
        triggers.append("pii_leakage")
    if _matches_any(UNSAFE_PATTERNS, answer):
        triggers.append("unsafe_output")
    if "academic_integrity" in question_triggers and _looks_like_full_solution(answer):
        triggers.append("academic_integrity")

    if triggers:
        if triggers == ["ungrounded"]:
            return GuardResult(
                allowed=False,
                triggers=triggers,
                message="I do not have enough course-grounded evidence to answer that. Please ask about the provided course material.",
            )
        return GuardResult(
            allowed=False,
            triggers=triggers,
            message="I cannot provide that answer in its current form. I can offer high-level guidance or point to relevant course material.",
        )
    return GuardResult(allowed=True, triggers=[])


def make_integrity_safe(question: str) -> str:
    return (
        "I cannot complete graded work for you. I can help you understand the task, outline an approach, "
        "or work through a similar example step by step."
    )


def sanitize_untrusted_context(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    safe_sentences = [sentence for sentence in sentences if not _matches_any(INJECTION_PATTERNS, sentence)]
    return " ".join(safe_sentences).strip()


def _matches_any(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _looks_like_full_solution(answer: str) -> bool:
    lowered = answer.lower()
    return "final answer" in lowered or "submit" in lowered or len(answer.split()) > 180
