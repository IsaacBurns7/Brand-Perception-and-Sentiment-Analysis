"""Print accuracy / macro-F1 / weighted-F1 for HF batch CSVs with true_label + predicted_label."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from models.sentiment.evaluate import get_accuracy, get_macro_f1, get_weighted_f1
else:
    from .evaluate import get_accuracy, get_macro_f1, get_weighted_f1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report metrics for a predictions CSV (columns: true_label, predicted_label)."
    )
    parser.add_argument("predictions_csv", type=Path)
    args = parser.parse_args()

    path = args.predictions_csv
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    df = pd.read_csv(path)
    for column in ("true_label", "predicted_label"):
        if column not in df.columns:
            raise SystemExit(f"Missing required column '{column}'. Found: {list(df.columns)}")

    y_true = df["true_label"].astype(str).str.strip().str.lower()
    y_pred = df["predicted_label"].astype(str).str.strip().str.lower()

    print(f"rows: {len(df)}")
    print(f"accuracy: {get_accuracy(y_true, y_pred):.4f}")
    print(f"macro_f1: {get_macro_f1(y_true, y_pred):.4f}")
    print(f"weighted_f1: {get_weighted_f1(y_true, y_pred):.4f}")


if __name__ == "__main__":
    main()
