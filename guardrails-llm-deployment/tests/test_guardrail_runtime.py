from __future__ import annotations

from pathlib import Path

import pytest

from guardrails_llm.guardrail_runtime import (
    default_inhouse_runtime_path,
    load_guardrail_runtime_config,
)


def test_default_inhouse_runtime_config_contains_calibrated_controls() -> None:
    config = load_guardrail_runtime_config(default_inhouse_runtime_path())

    assert config.schema_version == 1
    assert config.profile == "inhouse"
    assert config.models.embedding == "BAAI/bge-m3"
    assert config.models.classifier == "Qwen/Qwen3.6-35B-A3B"
    assert config.retrieval.top_k == 8
    assert config.retrieval.evidence_min_score == 0.5203531980514526
    assert config.retrieval.policy_context_top_k == 2
    assert config.retrieval.policy_context_min_score == 0.51
    assert config.retrieval.entailment_min_confidence == 0.80
    assert config.classifier.strategy == "always"
    assert config.classifier.min_confidence == 0.65
    assert config.paths.policy == Path("data/guardrail_policy_bge_m3.toml")


def test_runtime_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = _write_config(tmp_path, extra="unexpected = true\n")

    with pytest.raises(ValueError, match="unknown runtime config keys: unexpected"):
        load_guardrail_runtime_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evidence_min_score", "1.1"),
        ("policy_context_min_score", "-0.1"),
        ("entailment_min_confidence", "2.0"),
    ],
)
def test_runtime_config_rejects_threshold_outside_unit_interval(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    path = _write_config(
        tmp_path,
        retrieval_overrides={field: value},
    )

    with pytest.raises(ValueError, match=field):
        load_guardrail_runtime_config(path)


def test_runtime_config_rejects_parent_path_escape(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        path_overrides={"policy": '"../outside.toml"'},
    )

    with pytest.raises(ValueError, match="paths.policy must stay within"):
        load_guardrail_runtime_config(path)


def _write_config(
    tmp_path: Path,
    *,
    extra: str = "",
    retrieval_overrides: dict[str, str] | None = None,
    path_overrides: dict[str, str] | None = None,
) -> Path:
    retrieval = {
        "top_k": "8",
        "evidence_min_score": "0.52",
        "policy_context_top_k": "2",
        "policy_context_min_score": "0.51",
        "entailment_min_confidence": "0.80",
        **(retrieval_overrides or {}),
    }
    paths = {
        "corpus": '"data/python_course_docs.jsonl"',
        "policy": '"data/guardrail_policy_bge_m3.toml"',
        "index": '"indexes/python-course-bge-m3"',
        "embedding_cache": '"indexes/cache/bge-m3.jsonl"',
        **(path_overrides or {}),
    }
    path = tmp_path / "runtime.toml"
    path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'profile = "inhouse"',
                'endpoint_host = "learning-services4.fokus.fraunhofer.de"',
                'course_id = "python-intro"',
                extra.rstrip(),
                "[models]",
                'embedding = "BAAI/bge-m3"',
                'answer = "Qwen/Qwen3.6-35B-A3B"',
                'classifier = "Qwen/Qwen3.6-35B-A3B"',
                'entailment = "Qwen/Qwen3.6-35B-A3B"',
                'judge = "Qwen/Qwen3.6-35B-A3B"',
                "[retrieval]",
                *(f"{key} = {value}" for key, value in retrieval.items()),
                "[classifier]",
                'strategy = "always"',
                "min_confidence = 0.65",
                "[paths]",
                *(f"{key} = {value}" for key, value in paths.items()),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path
