from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_ANSWER_MODEL = "gpt-5.4-mini"
DEFAULT_OPENAI_CLASSIFIER_MODEL = "gpt-5.4-nano"
DEFAULT_OPENAI_JUDGE_MODEL = "gpt-5.4-nano"


class RemoteModelsNotAllowedError(RuntimeError):
    pass


class MissingModelCredentialError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAIModelConfig:
    api_key_env: str = "OPENAI_API_KEY"
    embedding_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL
    answer_model: str = DEFAULT_OPENAI_ANSWER_MODEL
    classifier_model: str = DEFAULT_OPENAI_CLASSIFIER_MODEL
    judge_model: str = DEFAULT_OPENAI_JUDGE_MODEL
    env_file: Path | None = None
    allow_remote_models: bool = False

    @property
    def resolved_env_file(self) -> Path:
        return self.env_file or default_env_path()


def default_env_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".env"


def load_local_env(env_file: Path | None = None) -> Path:
    resolved = env_file or default_env_path()
    if resolved.exists():
        load_dotenv(resolved, override=False)
    return resolved


def openai_config_summary(env_file: Path | None = None) -> dict[str, object]:
    resolved = load_local_env(env_file)
    return {
        "provider": "openai",
        "env_file": str(resolved),
        "env_file_exists": resolved.exists(),
        "api_key_env": "OPENAI_API_KEY",
        "api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "embedding_model": DEFAULT_OPENAI_EMBEDDING_MODEL,
        "answer_model": DEFAULT_OPENAI_ANSWER_MODEL,
        "classifier_model": DEFAULT_OPENAI_CLASSIFIER_MODEL,
        "judge_model": DEFAULT_OPENAI_JUDGE_MODEL,
    }


def ensure_remote_models_allowed(config: OpenAIModelConfig) -> None:
    if not config.allow_remote_models:
        raise RemoteModelsNotAllowedError(
            "Remote model calls are disabled. Re-run with --allow-remote-models "
            "only after approving API usage for this command."
        )


def ensure_openai_api_key(config: OpenAIModelConfig) -> None:
    load_local_env(config.env_file)
    if not os.getenv(config.api_key_env):
        raise MissingModelCredentialError(
            f"{config.api_key_env} is not configured. Set it in the shell environment "
            f"or in {config.resolved_env_file}."
        )
