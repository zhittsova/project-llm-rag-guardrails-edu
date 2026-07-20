from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from .embeddings import HashingEmbedder, TextEmbedder, cosine_similarity
from .guard_text import fuzzy_phrase_matches, guard_text_candidates, normalize_guard_text


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

    def score(self, text: str, embedder: TextEmbedder) -> float:
        if not text.strip() or not self.examples:
            return 0.0
        query_vector = embedder.embed(normalize_guard_text(text))
        return max(
            cosine_similarity(query_vector, embedder.embed(normalize_guard_text(example)))
            for example in self.examples
        )

    def score_vectors(
        self,
        query_vector: list[float],
        example_vectors: tuple[list[float], ...],
    ) -> float:
        if not query_vector or not example_vectors:
            return 0.0
        return max(cosine_similarity(query_vector, vector) for vector in example_vectors)

    def matches(self, text: str, embedder: TextEmbedder) -> bool:
        return self.score(text, embedder) >= self.threshold


@dataclass(frozen=True)
class FuzzyRule:
    trigger: str
    phrases: tuple[str, ...]
    threshold: float

    def matches(self, text: str) -> bool:
        return any(
            fuzzy_phrase_matches(text, phrase, threshold=self.threshold)
            for phrase in self.phrases
        )


@dataclass(frozen=True)
class GuardrailPolicy:
    input_rules: tuple[RegexRule, ...] = field(default_factory=tuple)
    input_similarity_rules: tuple[SimilarityRule, ...] = field(default_factory=tuple)
    input_fuzzy_rules: tuple[FuzzyRule, ...] = field(default_factory=tuple)
    output_rules: tuple[RegexRule, ...] = field(default_factory=tuple)
    output_fuzzy_rules: tuple[FuzzyRule, ...] = field(default_factory=tuple)
    context_rules: tuple[RegexRule, ...] = field(default_factory=tuple)
    context_fuzzy_rules: tuple[FuzzyRule, ...] = field(default_factory=tuple)
    blocking_triggers: frozenset[str] = frozenset({"prompt_injection", "pii", "unsafe_request"})
    allowed_visibility: frozenset[str] = frozenset({"public"})
    require_citations: bool = True
    input_block_message: str = DEFAULT_INPUT_BLOCK_MESSAGE
    output_block_message: str = DEFAULT_OUTPUT_BLOCK_MESSAGE
    ungrounded_message: str = DEFAULT_UNGROUNDED_MESSAGE
    integrity_safe_message: str = DEFAULT_INTEGRITY_MESSAGE
    similarity_embedder: TextEmbedder = field(default_factory=HashingEmbedder)

    @classmethod
    def default(cls) -> GuardrailPolicy:
        return DEFAULT_POLICY

    def input_triggers(self, text: str) -> list[str]:
        return _unique(
            self.input_deterministic_triggers(text)
            + self.input_similarity_triggers(text)
        )

    def input_deterministic_triggers(self, text: str) -> list[str]:
        candidates = guard_text_candidates(text)
        triggers = [
            rule.trigger
            for rule in self.input_rules
            if any(rule.matches(candidate) for candidate in candidates)
        ]
        triggers.extend(rule.trigger for rule in self.input_fuzzy_rules if rule.matches(text))
        return _unique(triggers)

    def input_similarity_triggers(self, text: str) -> list[str]:
        return _unique([
            rule.trigger
            for rule, score in self._similarity_scores(text)
            if score >= rule.threshold
        ])

    def has_near_similarity_trigger(self, text: str, *, margin: float = 0.08) -> bool:
        return any(
            rule.threshold - margin <= score < rule.threshold
            for rule, score in self._similarity_scores(text)
        )

    def _similarity_scores(self, text: str) -> list[tuple[SimilarityRule, float]]:
        if not text.strip() or not self.input_similarity_rules:
            return []
        query_vector = self.similarity_embedder.embed(normalize_guard_text(text))
        return [
            (rule, rule.score_vectors(query_vector, example_vectors))
            for rule, example_vectors in zip(
                self.input_similarity_rules,
                self._similarity_example_vectors,
                strict=True,
            )
        ]

    @cached_property
    def _similarity_example_vectors(self) -> tuple[tuple[list[float], ...], ...]:
        examples = [
            normalize_guard_text(example)
            for rule in self.input_similarity_rules
            for example in rule.examples
        ]
        vectors = iter(self.similarity_embedder.embed_many(examples))
        return tuple(
            tuple(next(vectors) for _example in rule.examples)
            for rule in self.input_similarity_rules
        )

    def output_triggers(self, text: str) -> list[str]:
        candidates = guard_text_candidates(text)
        triggers = [
            rule.trigger
            for rule in self.output_rules
            if any(rule.matches(candidate) for candidate in candidates)
        ]
        triggers.extend(rule.trigger for rule in self.output_fuzzy_rules if rule.matches(text))
        return _unique(triggers)

    def has_context_injection(self, text: str) -> bool:
        candidates = guard_text_candidates(text)
        return any(
            any(rule.matches(candidate) for candidate in candidates)
            for rule in self.context_rules
        ) or any(rule.matches(text) for rule in self.context_fuzzy_rules)


def default_policy_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "guardrail_policy.toml"


def load_guardrail_policy(
    path: Path,
    *,
    similarity_embedder: TextEmbedder | None = None,
) -> GuardrailPolicy:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return guardrail_policy_from_mapping(
        data,
        similarity_embedder=similarity_embedder,
    )


def guardrail_policy_from_mapping(
    data: dict[str, object],
    *,
    similarity_embedder: TextEmbedder | None = None,
) -> GuardrailPolicy:
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
        input_fuzzy_rules=base.input_fuzzy_rules + _fuzzy_rules(input_section, "input.fuzzy_rules"),
        output_rules=base.output_rules + _regex_rules(output_section, "output.rules"),
        output_fuzzy_rules=base.output_fuzzy_rules + _fuzzy_rules(output_section, "output.fuzzy_rules"),
        context_rules=base.context_rules + _regex_rules(context_section, "context.rules"),
        context_fuzzy_rules=base.context_fuzzy_rules + _fuzzy_rules(context_section, "context.fuzzy_rules"),
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
        similarity_embedder=similarity_embedder or base.similarity_embedder,
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


def _fuzzy_rules(section: dict[str, object], label: str) -> tuple[FuzzyRule, ...]:
    raw_rules = section.get("fuzzy_rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError(f"{label} must be a list")
    rules = []
    for index, raw_rule in enumerate(raw_rules):
        rule = _as_table(raw_rule, f"{label}[{index}]")
        trigger = _as_string(rule.get("trigger"), f"{label}[{index}].trigger")
        phrases = tuple(_string_list(rule.get("phrases"), f"{label}[{index}].phrases"))
        threshold = _as_float(rule.get("threshold", 0.88), f"{label}[{index}].threshold")
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"{label}[{index}].threshold must be > 0 and <= 1")
        rules.append(FuzzyRule(trigger=trigger, phrases=phrases, threshold=threshold))
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
    input_fuzzy_rules=(
        FuzzyRule(
            trigger="prompt_injection",
            threshold=0.86,
            phrases=(
                "ignore previous instructions",
                "ignore system instructions",
                "disregard instructions",
                "reveal hidden prompt",
                "reveal system prompt",
                "show hidden prompt",
                "show developer message",
                "print hidden prompt",
                "act as a different assistant",
            ),
        ),
        FuzzyRule(
            trigger="pii",
            threshold=0.88,
            phrases=(
                "student emails",
                "student email addresses",
                "student grades",
                "student ids",
                "student accommodations",
                "class marks",
                "private roster",
                "confidential records",
                "internal student data",
            ),
        ),
        FuzzyRule(
            trigger="academic_integrity",
            threshold=0.86,
            phrases=(
                "final answer",
                "full solution",
                "answer key",
                "graded assignment",
                "complete my assignment",
                "complete graded assignment",
                "solve my homework",
                "write my exam response",
                "final submission",
            ),
        ),
        FuzzyRule(
            trigger="unsafe_request",
            threshold=0.9,
            phrases=(
                "build malware",
                "make a bomb",
                "build a weapon",
            ),
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
    output_fuzzy_rules=(
        FuzzyRule(
            trigger="prompt_injection_output",
            threshold=0.86,
            phrases=(
                "ignore previous instructions",
                "reveal hidden prompt",
                "reveal system prompt",
                "hidden system prompt",
                "developer message",
            ),
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
    context_fuzzy_rules=(
        FuzzyRule(
            trigger="prompt_injection",
            threshold=0.86,
            phrases=(
                "ignore previous instructions",
                "ignore system instructions",
                "disregard instructions",
                "reveal hidden prompt",
                "reveal system prompt",
                "hidden system prompt",
                "developer message",
            ),
        ),
    ),
)
