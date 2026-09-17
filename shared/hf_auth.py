"""HuggingFace auth helper - reads the token once, never writes it anywhere.

The token is expected in the HF_TOKEN environment variable (set outside the
repo, e.g. `$env:HF_TOKEN = "hf_..."`). It is used only to authorize access to
gated repositories (ai4bharat models, etc.) and is never logged or persisted.
"""
from __future__ import annotations

import os

log_imports: list[str] = []


def hf_token() -> str | None:
    """Return the HF_TOKEN from the environment, or None if unset."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    return token or None


def login_hf() -> None:
    """Best-effort login; no-op and non-fatal when the token is absent."""
    token = hf_token()
    if not token:
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception:
        # Non-fatal: gated-model attempts will fall back gracefully.
        pass