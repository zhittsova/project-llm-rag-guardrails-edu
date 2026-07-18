from __future__ import annotations

import json

import pytest

from guardrails_llm.model_config import (
    MissingModelCredentialError,
    OpenAIModelConfig,
    RemoteModelsNotAllowedError,
    ensure_openai_api_key,
    ensure_remote_models_allowed,
    openai_client_kwargs,
    openai_config_summary,
    resolve_openai_base_url,
    should_use_chat_completions,
)


def test_openai_config_summary_reports_key_presence_without_secret(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=secret-test-value\n", encoding="utf-8")

    summary = openai_config_summary(env_file)
    rendered = json.dumps(summary)

    assert summary["api_key_present"] is True
    assert "secret-test-value" not in rendered


def test_openai_config_summary_reports_api_url_alias_without_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=secret-test-value",
                "OPENAI_API_URL=https://learning.example.edu/litellm/v1",
            ]
        ),
        encoding="utf-8",
    )

    summary = openai_config_summary(env_file)
    rendered = json.dumps(summary)

    assert summary["base_url_present"] is True
    assert summary["base_url_source"] == "OPENAI_API_URL"
    assert summary["base_url_host"] == "learning.example.edu"
    assert "secret-test-value" not in rendered


def test_explicit_openai_base_url_wins_over_api_url_alias(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_BASE_URL=https://official.example.com/v1",
                "OPENAI_API_URL=https://alias.example.com/v1",
            ]
        ),
        encoding="utf-8",
    )

    resolved = resolve_openai_base_url(OpenAIModelConfig(env_file=env_file))

    assert resolved == "https://official.example.com/v1"


def test_openai_compatible_base_url_uses_chat_completions(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_URL=https://learning.example.edu/litellm/v1", encoding="utf-8")

    assert should_use_chat_completions(OpenAIModelConfig(env_file=env_file)) is True


def test_remote_model_calls_require_explicit_allow_flag() -> None:
    with pytest.raises(RemoteModelsNotAllowedError, match="--allow-remote-models"):
        ensure_remote_models_allowed(OpenAIModelConfig())


def test_openai_api_key_is_required_for_remote_clients(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    missing_env = tmp_path / ".env"

    with pytest.raises(MissingModelCredentialError, match="OPENAI_API_KEY"):
        ensure_openai_api_key(OpenAIModelConfig(env_file=missing_env, allow_remote_models=True))


def test_openai_client_uses_bounded_timeout_and_retry_policy(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    config = OpenAIModelConfig(
        env_file=env_file,
        request_timeout_seconds=45.0,
        max_transport_retries=2,
    )

    kwargs = openai_client_kwargs(config)

    assert kwargs["timeout"] == 45.0
    assert kwargs["max_retries"] == 2
