from __future__ import annotations

from pathlib import Path

import yaml

from pac_eval.models import GoldSet

_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "gold_set.yaml"


def load_gold_set(path: Path | None = None) -> GoldSet:
    resolved = path or _DEFAULT_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"gold set not found: {resolved}")

    raw = yaml.safe_load(resolved.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"gold set YAML must be a mapping, got {type(raw).__name__}")

    return GoldSet.model_validate(raw)
