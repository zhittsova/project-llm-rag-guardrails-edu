from __future__ import annotations

import json

import pytest

from guardrails_llm.model_config import (
    MissingModelCredentialError,
    OpenAIModelConfig,
    RemoteModelsNotAllowedError,
    ensure_openai_api_key,
    ensure_remote_models_allowed,
    openai_config_summary,
)


def test_openai_config_summary_reports_key_presence_without_secret(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=secret-test-value\n", encoding="utf-8")

    summary = openai_config_summary(env_file)
    rendered = json.dumps(summary)

    assert summary["api_key_present"] is True
    assert "secret-test-value" not in rendered


def test_remote_model_calls_require_explicit_allow_flag() -> None:
    with pytest.raises(RemoteModelsNotAllowedError, match="--allow-remote-models"):
        ensure_remote_models_allowed(OpenAIModelConfig())


def test_openai_api_key_is_required_for_remote_clients(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    missing_env = tmp_path / ".env"

    with pytest.raises(MissingModelCredentialError, match="OPENAI_API_KEY"):
        ensure_openai_api_key(OpenAIModelConfig(env_file=missing_env, allow_remote_models=True))
