"""Shared utility helpers for the sentiment pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_directories(paths: list[Path] | tuple[Path, ...]) -> None:
    """Create directories if they do not already exist."""

    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def current_timestamp() -> str:
    """Return a UTC ISO-8601 timestamp."""

    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    """Convert common scientific Python objects into JSON-safe values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Save a dictionary to disk as formatted JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(data), file, indent=2, ensure_ascii=True, sort_keys=True)


def format_label_distribution(labels: pd.Series) -> dict[str, dict[str, float | int]]:
    """Return label counts and proportions in a compact serializable format."""

    counts = labels.value_counts(dropna=False)
    proportions = labels.value_counts(normalize=True, dropna=False)
    distribution: dict[str, dict[str, float | int]] = {}

    for label, count in counts.items():
        label_key = str(label)
        distribution[label_key] = {
            "count": int(count),
            "proportion": float(round(proportions[label], 6)),
        }

    return distribution
