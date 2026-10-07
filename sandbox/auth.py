"""Codex credential loading. Credential values never enter argv, traces or manifests.

Ported from finautorubric ``final_pipeline/auth.py`` (Codex part only).
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def interpreted_codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser().resolve()


def load_codex_auth(auth_json: Path | None = None, api_key_env: str | None = None):
    """Return credential bytes and mode without leaking credential values."""
    if api_key_env is not None:
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError("set the explicitly selected API key variable: " + api_key_env)
        return json.dumps({"OPENAI_API_KEY": key}).encode(), "api_key"
    path = Path(auth_json) if auth_json is not None else interpreted_codex_home() / "auth.json"
    try:
        auth = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(
            "saved ChatGPT file login not found at " + str(path)
            + "; set cli_auth_credentials_store = \"file\" and run codex login, or use --auth-json"
        ) from exc
    try:
        record = json.loads(auth)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("Codex auth.json is not valid JSON") from exc
    if not isinstance(record, dict) or record.get("OPENAI_API_KEY") or record.get("auth_mode") == "apikey":
        raise ValueError("account mode requires ChatGPT auth; API keys require explicit --api-key-env")
    tokens = record.get("tokens")
    if not isinstance(tokens, dict) or not tokens.get("access_token"):
        raise ValueError("auth.json has no saved ChatGPT account token")
    return auth, "chatgpt"
