"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
