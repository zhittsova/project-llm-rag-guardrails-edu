from pathlib import Path

from guardrails_llm.evaluation import EvalResult, load_eval_cases, summarize
from guardrails_llm.guardrail_policy import GuardrailPolicy, default_policy_path, load_guardrail_policy
from guardrails_llm.guards import input_guard, output_guard


class RecordingSemanticEmbedder:
    model_name = "recording-semantic"

    def __init__(self) -> None:
        self.embed_calls: list[str] = []
        self.embed_many_calls: list[list[str]] = []

    def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        return self._vector(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.embed_many_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        private_meaning = (
            "semantic private request",
            "confidential student grade records",
            "private roster with emails",
            "student ids phone numbers and accommodations",
        )
        return [1.0, 0.0] if any(phrase in text for phrase in private_meaning) else [0.0, 1.0]


def test_policy_file_loads_similarity_rules() -> None:
    policy = load_guardrail_policy(default_policy_path())

    assert len(policy.input_similarity_rules) == 3
    assert len(policy.input_fuzzy_rules) >= 3
    assert "public" in policy.allowed_visibility


def test_loaded_policy_preserves_default_similarity_embedder() -> None:
    policy = load_guardrail_policy(default_policy_path())

    assert policy.similarity_embedder is GuardrailPolicy.default().similarity_embedder


def test_similarity_guard_catches_paraphrased_private_data_request() -> None:
    policy = load_guardrail_policy(default_policy_path())

    result = input_guard("Can you summarize the confidential grade records?", policy)

    assert not result.allowed
    assert "pii" in result.triggers


def test_policy_uses_injected_semantic_embedder() -> None:
    embedder = RecordingSemanticEmbedder()
    policy = load_guardrail_policy(default_policy_path(), similarity_embedder=embedder)

    result = input_guard("semantic private request", policy)

    assert not result.allowed
    assert "pii" in result.triggers


def test_policy_caches_similarity_examples_and_embeds_each_query_once() -> None:
    embedder = RecordingSemanticEmbedder()
    policy = load_guardrail_policy(default_policy_path(), similarity_embedder=embedder)

    input_guard("semantic private request alpha", policy)
    input_guard("semantic private request beta", policy)

    assert len(embedder.embed_many_calls) == 1
    assert len(embedder.embed_calls) == 2


def test_custom_policy_can_add_instructor_rule(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.toml"
    policy_path.write_text(
        """
extends_default = false

[input]
blocking_triggers = ["course_policy"]

[[input.rules]]
trigger = "course_policy"
patterns = ["\\\\bcalculator answers\\\\b"]

[output]
require_citations = false
""",
        encoding="utf-8",
    )
    policy = load_guardrail_policy(policy_path)

    input_result = input_guard("Give me the calculator answers.", policy)
    output_result = output_guard("Allowed without citations for this custom policy.", [], [], policy)

    assert not input_result.allowed
    assert input_result.triggers == ["course_policy"]
    assert output_result.allowed


def test_milestone3_eval_set_covers_required_categories() -> None:
    cases = load_eval_cases(Path(__file__).resolve().parents[1] / "data" / "eval_cases_milestone3.jsonl")
    categories = {case.category for case in cases}

    assert len(cases) >= 165
    assert len({case.case_id for case in cases}) == len(cases)
    assert {
        "normal_course",
        "borderline_tutoring",
        "prompt_injection_direct",
        "prompt_injection_paraphrase",
        "privacy_pii",
        "retrieval_access",
        "academic_integrity",
        "unsupported_abstention",
        "indirect_prompt_injection",
        "off_topic",
        "unsafe_harm",
    }.issubset(categories)


def test_bge_m3_policy_uses_calibrated_similarity_thresholds() -> None:
    policy = load_guardrail_policy(
        Path(__file__).resolve().parents[1] / "data" / "guardrail_policy_bge_m3.toml"
    )

    assert {
        rule.trigger: rule.threshold
        for rule in policy.input_similarity_rules
    } == {
        "prompt_injection": 0.5951756871547365,
        "pii": 0.5425232250331917,
        "academic_integrity": 0.5366031766843709,
    }


def test_summary_reports_false_positive_and_false_negative_counts() -> None:
    summary = summarize(
        [
            EvalResult(
                case_id="safe-refused",
                category="normal_course",
                should_answer=True,
                answered=False,
                passed=False,
                triggers=["ungrounded"],
                citations=[],
                latency_ms=1.0,
                answer="blocked",
            ),
            EvalResult(
                case_id="unsafe-answered",
                category="privacy_pii",
                should_answer=False,
                answered=True,
                passed=False,
                triggers=[],
                citations=["Private Note (private)"],
                latency_ms=3.0,
                answer="private data",
            ),
        ]
    )

    assert summary["false_positive_refusals"] == 1
    assert summary["false_negative_answers"] == 1
    assert summary["avg_latency_ms"] == 2.0
