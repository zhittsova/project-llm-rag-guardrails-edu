from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .embeddings import HashingEmbedder, cosine_similarity


DEFAULT_INPUT_BLOCK_MESSAGE = (
    "I cannot process this request because it may contain unsafe instructions or sensitive data."
)
DEFAULT_OUTPUT_BLOCK_MESSAGE = (
    "I cannot provide that answer in its current form. I can offer high-level guidance or point to relevant course material."
)
DEFAULT_UNGROUNDED_MESSAGE = (
    "I do not have enough course-grounded evidence to answer that. Please ask about the provided course material."
)
DEFAULT_INTEGRITY_MESSAGE = (
    "I cannot complete graded work for you. I can help you understand the task, outline an approach, "
    "or work through a similar example step by step."
)


@dataclass(frozen=True)
class RegexRule:
    trigger: str
    patterns: tuple[re.Pattern[str], ...]

    @classmethod
    def from_strings(cls, trigger: str, patterns: list[str]) -> RegexRule:
        return cls(trigger=trigger, patterns=tuple(re.compile(pattern, re.I) for pattern in patterns))

    def matches(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.patterns)


@dataclass(frozen=True)
class SimilarityRule:
    trigger: str
    examples: tuple[str, ...]
    threshold: float

    def matches(self, text: str, embedder: HashingEmbedder) -> bool:
        if not text.strip() or not self.examples:
            return False
        query_vector = embedder.embed(text)
        return any(
            cosine_similarity(query_vector, embedder.embed(example)) >= self.threshold
            for example in self.examples
        )


@dataclass(frozen=True)
class GuardrailPolicy:
    input_rules: tuple[RegexRule, ...] = field(default_factory=tuple)
    input_similarity_rules: tuple[SimilarityRule, ...] = field(default_factory=tuple)
    output_rules: tuple[RegexRule, ...] = field(default_factory=tuple)
    context_rules: tuple[RegexRule, ...] = field(default_factory=tuple)
    blocking_triggers: frozenset[str] = frozenset({"prompt_injection", "pii", "unsafe_request"})
    allowed_visibility: frozenset[str] = frozenset({"public"})
    require_citations: bool = True
    input_block_message: str = DEFAULT_INPUT_BLOCK_MESSAGE
    output_block_message: str = DEFAULT_OUTPUT_BLOCK_MESSAGE
    ungrounded_message: str = DEFAULT_UNGROUNDED_MESSAGE
    integrity_safe_message: str = DEFAULT_INTEGRITY_MESSAGE
    similarity_embedder: HashingEmbedder = field(default_factory=HashingEmbedder)

    @classmethod
    def default(cls) -> GuardrailPolicy:
        return DEFAULT_POLICY

    def input_triggers(self, text: str) -> list[str]:
        triggers = [rule.trigger for rule in self.input_rules if rule.matches(text)]
        triggers.extend(
            rule.trigger
            for rule in self.input_similarity_rules
            if rule.matches(text, self.similarity_embedder)
        )
        return _unique(triggers)

    def output_triggers(self, text: str) -> list[str]:
        return _unique([rule.trigger for rule in self.output_rules if rule.matches(text)])

    def has_context_injection(self, text: str) -> bool:
        return any(rule.matches(text) for rule in self.context_rules)


def default_policy_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "guardrail_policy.toml"


def load_guardrail_policy(path: Path) -> GuardrailPolicy:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    extends_default = _as_bool(data.get("extends_default", True), "extends_default")
    base = GuardrailPolicy.default() if extends_default else GuardrailPolicy()

    input_section = _as_table(data.get("input", {}), "input")
    output_section = _as_table(data.get("output", {}), "output")
    context_section = _as_table(data.get("context", {}), "context")
    retrieval_section = _as_table(data.get("retrieval", {}), "retrieval")
    messages_section = _as_table(data.get("messages", {}), "messages")

    return GuardrailPolicy(
        input_rules=base.input_rules + _regex_rules(input_section, "input.rules"),
        input_similarity_rules=base.input_similarity_rules
        + _similarity_rules(input_section, "input.similarity_rules"),
        output_rules=base.output_rules + _regex_rules(output_section, "output.rules"),
        context_rules=base.context_rules + _regex_rules(context_section, "context.rules"),
        blocking_triggers=frozenset(
            _string_list(input_section.get("blocking_triggers", sorted(base.blocking_triggers)), "input.blocking_triggers")
        ),
        allowed_visibility=frozenset(
            _string_list(
                retrieval_section.get("allowed_visibility", sorted(base.allowed_visibility)),
                "retrieval.allowed_visibility",
            )
        ),
        require_citations=_as_bool(output_section.get("require_citations", base.require_citations), "output.require_citations"),
        input_block_message=_as_string(
            messages_section.get("input_block", base.input_block_message),
            "messages.input_block",
        ),
        output_block_message=_as_string(
            messages_section.get("output_block", base.output_block_message),
            "messages.output_block",
        ),
        ungrounded_message=_as_string(
            messages_section.get("ungrounded", base.ungrounded_message),
            "messages.ungrounded",
        ),
        integrity_safe_message=_as_string(
            messages_section.get("integrity_safe", base.integrity_safe_message),
            "messages.integrity_safe",
        ),
    )


def _regex_rules(section: dict[str, object], label: str) -> tuple[RegexRule, ...]:
    raw_rules = section.get("rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError(f"{label} must be a list")
    rules = []
    for index, raw_rule in enumerate(raw_rules):
        rule = _as_table(raw_rule, f"{label}[{index}]")
        trigger = _as_string(rule.get("trigger"), f"{label}[{index}].trigger")
        patterns = _string_list(rule.get("patterns"), f"{label}[{index}].patterns")
        rules.append(RegexRule.from_strings(trigger, patterns))
    return tuple(rules)


def _similarity_rules(section: dict[str, object], label: str) -> tuple[SimilarityRule, ...]:
    raw_rules = section.get("similarity_rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError(f"{label} must be a list")
    rules = []
    for index, raw_rule in enumerate(raw_rules):
        rule = _as_table(raw_rule, f"{label}[{index}]")
        trigger = _as_string(rule.get("trigger"), f"{label}[{index}].trigger")
        examples = tuple(_string_list(rule.get("examples"), f"{label}[{index}].examples"))
        threshold = _as_float(rule.get("threshold", 0.45), f"{label}[{index}].threshold")
        rules.append(SimilarityRule(trigger=trigger, examples=examples, threshold=threshold))
    return tuple(rules)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _as_table(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a table")
    return value


def _as_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _as_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _as_float(value: object, label: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{label} must be a number")
    return float(value)


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{label} must be a list of non-empty strings")
    return value


DEFAULT_POLICY = GuardrailPolicy(
    input_rules=(
        RegexRule.from_strings(
            "prompt_injection",
            [
                r"\bignore (all )?(previous|above|system|developer) instructions\b",
                r"\breveal\b.*\b(system prompt|hidden prompt|developer message)\b",
                r"\bdisregard\b.*\binstructions\b",
                r"\byou are now\b",
                r"<\s*script\b",
            ],
        ),
        RegexRule.from_strings(
            "pii",
            [
                r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
                r"\b(?:\+?\d[\d .()-]{7,}\d)\b",
                r"\b(?:student|matriculation|ssn|passport)\s*(?:id|number|no\.?)\s*[:#]?\s*[a-z0-9-]{4,}\b",
                r"\b(private|confidential|internal)\b.*\b(roster|note|data|record|file)\b",
                r"\b(student|professor|teacher|staff)\b.*\b(email addresses?|phone numbers?|grades?|ids?|accommodations?)\b",
                r"\bwhat\b.*\b(email addresses?|phone numbers?|grades?|ids?)\b",
            ],
        ),
        RegexRule.from_strings(
            "academic_integrity",
            [
                r"\b(write|solve|complete|do)\b.*\b(my|the)\b.*\b(assignment|homework|exam|quiz|graded)\b",
                r"\bgive me\b.*\b(final answer|full solution|submission)\b",
                r"\bplagiar",
            ],
        ),
        RegexRule.from_strings(
            "unsafe_request",
            [
                r"\b(self-harm|suicide|kill myself)\b",
                r"\b(make|build)\b.*\b(weapon|bomb|malware)\b",
            ],
        ),
    ),
    output_rules=(
        RegexRule.from_strings(
            "prompt_injection_output",
            [
                r"\bignore (all )?(previous|above|system|developer) instructions\b",
                r"\breveal\b.*\b(system prompt|hidden prompt|developer message)\b",
                r"\bdisregard\b.*\binstructions\b",
                r"\byou are now\b",
                r"<\s*script\b",
            ],
        ),
        RegexRule.from_strings(
            "pii_leakage",
            [
                r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
                r"\b(?:\+?\d[\d .()-]{7,}\d)\b",
                r"\b(?:student|matriculation|ssn|passport)\s*(?:id|number|no\.?)\s*[:#]?\s*[a-z0-9-]{4,}\b",
            ],
        ),
        RegexRule.from_strings(
            "unsafe_output",
            [
                r"\b(self-harm|suicide|kill myself)\b",
                r"\b(make|build)\b.*\b(weapon|bomb|malware)\b",
            ],
        ),
    ),
    context_rules=(
        RegexRule.from_strings(
            "prompt_injection",
            [
                r"\bignore (all )?(previous|above|system|developer) instructions\b",
                r"\breveal\b.*\b(system prompt|hidden prompt|developer message)\b",
                r"\bdisregard\b.*\binstructions\b",
                r"\byou are now\b",
                r"<\s*script\b",
            ],
        ),
    ),
)
