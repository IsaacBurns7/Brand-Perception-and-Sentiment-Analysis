"""Evaluation helpers for the ABSA pipeline.

Three granularities are measured:

Aspect extraction (ATE)
    Precision / Recall / F1 on aspect-term strings (exact match, case-insensitive).

Sentiment accuracy (APC)
    Among correctly extracted aspects, fraction with correct polarity.

End-to-end triplet F1
    A predicted triplet (aspect, polarity) is correct only if both the
    aspect string and polarity match the gold annotation.

Usage:
    python -m models.absa.evaluate \\
        --predictions artifacts/reports/absa/test_predictions.csv \\
        --gold data/processed/mams_aste/test.dat.aste

The --predictions CSV must contain columns: sentence, aspect, predicted_polarity.
The --gold file is a .dat.aste file (``sentence#### #### ####[(from, to, POL)]``).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import ABSA_REPORT_DIR, MAMS_ASTE_TEST, POLARITY_REVERSE, PROJECT_ROOT

# Matches individual triplet tokens inside the bracket list, e.g. (1, 2, POS)
_TRIPLET_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\w+)\s*\)")


# ---------------------------------------------------------------------------
# Gold annotation parsing
# ---------------------------------------------------------------------------

def _parse_aste_line(line: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse one .dat.aste line into (sentence_text, list of gold triplet dicts).

    Each triplet dict has keys: word_from, word_to, polarity (canonical UPPER code).
    """
    if "#### #### ####" not in line:
        return line.strip(), []

    parts = line.split("#### #### ####", 1)
    sentence = parts[0].strip()
    annotation = parts[1].strip() if len(parts) > 1 else ""

    triplets: list[dict[str, Any]] = []
    for match in _TRIPLET_RE.finditer(annotation):
        word_from, word_to, polarity = int(match.group(1)), int(match.group(2)), match.group(3).upper()
        triplets.append(
            {"word_from": word_from, "word_to": word_to, "polarity": polarity}
        )
    return sentence, triplets


def load_gold_from_aste(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Return {sentence_text: [triplet_dict, ...]} from a .dat.aste file.

    When a sentence appears multiple times (rare), annotations are merged.
    """
    gold: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            sentence, triplets = _parse_aste_line(line)
            if sentence not in gold:
                gold[sentence] = []
            gold[sentence].extend(triplets)
    return gold


def _recover_gold_aspect_term(sentence: str, word_from: int, word_to: int) -> str:
    """Reconstruct the aspect string from word indices in the sentence."""
    tokens = sentence.split()
    span = tokens[word_from : word_to + 1]
    return " ".join(span).lower()


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def _precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    predictions_csv: Path,
    gold_aste: Path,
) -> dict[str, Any]:
    """Compare a predictions CSV against a .dat.aste gold file.

    Returns a nested dict with ATE, APC, and end-to-end (E2E) metrics.

    Predictions CSV schema (required columns):
        sentence            – the input sentence text
        aspect              – predicted aspect term string (may be NaN for no-aspect rows)
        predicted_polarity  – POS | NEG | NEU (may be NaN)
    """
    import pandas as pd

    pred_df = pd.read_csv(predictions_csv)
    required_columns = {"sentence", "aspect", "predicted_polarity"}
    missing = required_columns - set(pred_df.columns)
    if missing:
        raise KeyError(f"Predictions CSV missing columns: {sorted(missing)}")

    gold = load_gold_from_aste(gold_aste)

    # ── ATE counters ────────────────────────────────────────────────────────
    ate_tp = ate_fp = ate_fn = 0
    # ── APC counters ────────────────────────────────────────────────────────
    apc_correct = apc_total = 0
    # ── E2E counters ────────────────────────────────────────────────────────
    e2e_tp = e2e_fp = e2e_fn = 0

    sentences_in_gold = set(gold)
    sentences_in_pred = set(pred_df["sentence"].dropna().unique())

    for sentence in sentences_in_gold | sentences_in_pred:
        gold_triplets = gold.get(sentence, [])

        # Predicted aspects for this sentence (skip NaN rows = no aspect found)
        sent_preds = pred_df[pred_df["sentence"] == sentence].dropna(subset=["aspect"])
        pred_aspects_raw: list[tuple[str, str]] = [
            (str(row["aspect"]).strip().lower(), str(row["predicted_polarity"]).strip().upper())
            for _, row in sent_preds.iterrows()
        ]

        # Recover gold aspect strings from word positions
        gold_aspects: list[tuple[str, str]] = [
            (
                _recover_gold_aspect_term(sentence, t["word_from"], t["word_to"]),
                t["polarity"].upper(),
            )
            for t in gold_triplets
        ]

        gold_aspect_strings = {a for a, _ in gold_aspects}
        pred_aspect_strings = {a for a, _ in pred_aspects_raw}

        # ── ATE ─────────────────────────────────────────────────────────────
        ate_tp += len(gold_aspect_strings & pred_aspect_strings)
        ate_fp += len(pred_aspect_strings - gold_aspect_strings)
        ate_fn += len(gold_aspect_strings - pred_aspect_strings)

        # ── APC: correct-aspect subset ──────────────────────────────────────
        matched_aspects = gold_aspect_strings & pred_aspect_strings
        for aspect in matched_aspects:
            gold_pol = next((p for a, p in gold_aspects if a == aspect), None)
            pred_pol = next((p for a, p in pred_aspects_raw if a == aspect), None)
            if gold_pol is not None and pred_pol is not None:
                apc_total += 1
                if gold_pol == pred_pol:
                    apc_correct += 1

        # ── E2E: both aspect AND polarity must match ─────────────────────────
        gold_e2e = set(gold_aspects)
        pred_e2e = set(pred_aspects_raw)
        e2e_tp += len(gold_e2e & pred_e2e)
        e2e_fp += len(pred_e2e - gold_e2e)
        e2e_fn += len(gold_e2e - pred_e2e)

    ate_metrics = _precision_recall_f1(ate_tp, ate_fp, ate_fn)
    apc_accuracy = round(apc_correct / apc_total, 4) if apc_total > 0 else 0.0
    e2e_metrics = _precision_recall_f1(e2e_tp, e2e_fp, e2e_fn)

    return {
        "aspect_extraction": {
            **ate_metrics,
            "tp": ate_tp,
            "fp": ate_fp,
            "fn": ate_fn,
        },
        "sentiment_accuracy_on_matched_aspects": {
            "accuracy": apc_accuracy,
            "correct": apc_correct,
            "total": apc_total,
        },
        "end_to_end_triplet": {
            **e2e_metrics,
            "tp": e2e_tp,
            "fp": e2e_fp,
            "fn": e2e_fn,
        },
        "sentences_in_gold": len(sentences_in_gold),
        "sentences_in_pred": len(sentences_in_pred),
        "predictions_file": str(predictions_csv),
        "gold_file": str(gold_aste),
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_evaluation(metrics: dict[str, Any]) -> None:
    """Print a formatted evaluation summary to stdout."""
    ate = metrics["aspect_extraction"]
    apc = metrics["sentiment_accuracy_on_matched_aspects"]
    e2e = metrics["end_to_end_triplet"]

    print("\n" + "=" * 60)
    print("ABSA Evaluation Summary")
    print("=" * 60)
    print(f"Gold file  : {metrics['gold_file']}")
    print(f"Predictions: {metrics['predictions_file']}")
    print(f"Sentences in gold : {metrics['sentences_in_gold']}")
    print(f"Sentences in pred : {metrics['sentences_in_pred']}")
    print()
    print("Aspect Term Extraction (ATE)")
    print(f"  Precision : {ate['precision']:.4f}")
    print(f"  Recall    : {ate['recall']:.4f}")
    print(f"  F1        : {ate['f1']:.4f}")
    print(f"  TP/FP/FN  : {ate['tp']} / {ate['fp']} / {ate['fn']}")
    print()
    print("Sentiment Accuracy on Matched Aspects (APC)")
    print(f"  Accuracy  : {apc['accuracy']:.4f}  ({apc['correct']}/{apc['total']})")
    print()
    print("End-to-End Triplet F1 (ATE + APC jointly correct)")
    print(f"  Precision : {e2e['precision']:.4f}")
    print(f"  Recall    : {e2e['recall']:.4f}")
    print(f"  F1        : {e2e['f1']:.4f}")
    print(f"  TP/FP/FN  : {e2e['tp']} / {e2e['fp']} / {e2e['fn']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ABSA predictions against a .dat.aste gold file."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="CSV produced by models.absa.predict (batch command).",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=MAMS_ASTE_TEST,
        help=f"Gold .dat.aste file (default: {MAMS_ASTE_TEST.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the evaluation metrics as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.predictions.exists():
        print(f"Error: predictions file not found: {args.predictions}", file=sys.stderr)
        sys.exit(1)

    if not args.gold.exists():
        print(f"Error: gold file not found: {args.gold}", file=sys.stderr)
        sys.exit(1)

    metrics = evaluate_predictions(
        predictions_csv=args.predictions,
        gold_aste=args.gold,
    )
    print_evaluation(metrics)

    if args.output_json is not None:
        import json

        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, sort_keys=True)
        print(f"\nMetrics saved to: {args.output_json}")


if __name__ == "__main__":
    main()
