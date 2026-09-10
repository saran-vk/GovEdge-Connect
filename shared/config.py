"""Centralised path + settings loading.

Resolves the repo root so all relative paths in settings.yaml work
regardless of the current working directory.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_settings() -> dict:
    cfg_path = REPO_ROOT / "config" / "settings.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if cfg.get("project_root") is None:
        cfg["project_root"] = str(REPO_ROOT)
    return cfg


def resolve(relative_path: str) -> Path:
    cfg = load_settings()
    root = Path(cfg["project_root"])
    path = Path(relative_path)
    return path if path.is_absolute() else (root / path)
