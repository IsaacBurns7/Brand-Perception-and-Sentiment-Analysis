"""Evaluate the EMCGCN ABSA model directly on a .dat.aste file.

Uses EMCGCN's own polarity predictions (not RoBERTa stage-2).
Computes ATE, APC, and end-to-end triplet F1.

Usage:
    python scripts/eval_absa_emcgcn.py
    python scripts/eval_absa_emcgcn.py --split test
    python scripts/eval_absa_emcgcn.py --split val --checkpoint artifacts/models/absa_aste/emcgcn_brand_perception_f1_71.71
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT = PROJECT_ROOT / "artifacts/models/absa_aste/emcgcn_brand_perception_f1_71.71"
MAMS_ASTE_DIR = PROJECT_ROOT / "data/processed/mams_aste"

_TRIPLET_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\w+)\s*\)")


# ---------------------------------------------------------------------------
# Parse .dat.aste gold file
# ---------------------------------------------------------------------------

def _parse_gold(path: Path) -> list[dict]:
    """Return list of {sentence, triplets: [(aspect_str, polarity)]} from gold file."""
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if "#### #### ####" not in line:
                records.append({"sentence": line, "triplets": []})
                continue
            parts = line.split("#### #### ####", 1)
            sentence = parts[0].strip()
            tokens = sentence.split()
            annotation = parts[1].strip()
            triplets = []
            for m in _TRIPLET_RE.finditer(annotation):
                w_from, w_to, pol = int(m.group(1)), int(m.group(2)), m.group(3).upper()
                aspect_str = " ".join(tokens[w_from : w_to + 1]).lower()
                triplets.append((aspect_str, pol))
            records.append({"sentence": sentence, "triplets": triplets})
    return records


# ---------------------------------------------------------------------------
# Run EMCGCN inference
# ---------------------------------------------------------------------------

def _predict(sentences: list[str], checkpoint: Path) -> list[list[tuple[str, str]]]:
    """Return predicted [(aspect_str, polarity)] per sentence using EMCGCN.

    Processes one sentence at a time to avoid PyABSA crashing the whole batch
    when a POS tag is not in the model vocabulary.
    """
    from pyabsa import AspectSentimentTripletExtraction as ASTE  # type: ignore[import]

    extractor = ASTE.AspectSentimentTripletExtractor(
        checkpoint=str(checkpoint),
        auto_device=True,
    )

    results: list[list[tuple[str, str]]] = []
    skipped = 0
    for i, sentence in enumerate(sentences):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sentences)} sentences processed ({skipped} skipped) …")
        try:
            raw = extractor.predict(sentence, print_result=False)
            if not isinstance(raw, list):
                raw = [raw]
            triplets_out: list[tuple[str, str]] = []
            for item in raw:
                for t in item.get("Triplets", []) or []:
                    aspect = str(t.get("Aspect", "")).strip().lower()
                    polarity = str(t.get("Sentiment", t.get("Polarity", ""))).strip().upper()
                    polarity = {"POSITIVE": "POS", "NEGATIVE": "NEG", "NEUTRAL": "NEU"}.get(polarity, polarity)
                    if aspect:
                        triplets_out.append((aspect, polarity))
            results.append(triplets_out)
        except Exception:
            # Sentence has unseen POS tags — treat as no aspects predicted
            results.append([])
            skipped += 1

    print(f"  Done. {len(sentences)} sentences, {skipped} skipped due to vocab errors.")
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def evaluate(gold_records: list[dict], pred_triplets: list[list[tuple[str, str]]]) -> dict:
    ate_tp = ate_fp = ate_fn = 0
    apc_correct = apc_total = 0
    e2e_tp = e2e_fp = e2e_fn = 0

    for gold_rec, preds in zip(gold_records, pred_triplets):
        gold = gold_rec["triplets"]
        gold_aspects = {a for a, _ in gold}
        pred_aspects = {a for a, _ in preds}

        # ATE
        ate_tp += len(gold_aspects & pred_aspects)
        ate_fp += len(pred_aspects - gold_aspects)
        ate_fn += len(gold_aspects - pred_aspects)

        # APC (on matched aspects)
        for aspect in gold_aspects & pred_aspects:
            gold_pol = next((p for a, p in gold if a == aspect), None)
            pred_pol = next((p for a, p in preds if a == aspect), None)
            if gold_pol and pred_pol:
                apc_total += 1
                if gold_pol == pred_pol:
                    apc_correct += 1

        # E2E
        gold_set = set(gold)
        pred_set = set(preds)
        e2e_tp += len(gold_set & pred_set)
        e2e_fp += len(pred_set - gold_set)
        e2e_fn += len(gold_set - pred_set)

    return {
        "ate": _prf(ate_tp, ate_fp, ate_fn),
        "apc_accuracy": round(apc_correct / apc_total, 4) if apc_total else 0.0,
        "apc_detail": {"correct": apc_correct, "total": apc_total},
        "e2e": _prf(e2e_tp, e2e_fp, e2e_fn),
        "sentences": len(gold_records),
    }


def print_results(metrics: dict, split: str, gold_path: Path, checkpoint: Path) -> None:
    ate = metrics["ate"]
    e2e = metrics["e2e"]
    apc_acc = metrics["apc_accuracy"]
    apc = metrics["apc_detail"]

    print("\n" + "=" * 62)
    print(f"EMCGCN ABSA Evaluation  —  {split.upper()} split")
    print("=" * 62)
    print(f"Checkpoint : {checkpoint.name}")
    print(f"Gold file  : {gold_path.relative_to(PROJECT_ROOT)}")
    print(f"Sentences  : {metrics['sentences']}")
    print()
    print("Aspect Term Extraction (ATE)")
    print(f"  Precision : {ate['precision']:.4f}")
    print(f"  Recall    : {ate['recall']:.4f}")
    print(f"  F1        : {ate['f1']:.4f}   [TP={ate['tp']} FP={ate['fp']} FN={ate['fn']}]")
    print()
    print("Sentiment Accuracy on Matched Aspects (APC)")
    print(f"  Accuracy  : {apc_acc:.4f}   ({apc['correct']}/{apc['total']})")
    print()
    print("End-to-End Triplet F1  (aspect + polarity both correct)")
    print(f"  Precision : {e2e['precision']:.4f}")
    print(f"  Recall    : {e2e['recall']:.4f}")
    print(f"  F1        : {e2e['f1']:.4f}   [TP={e2e['tp']} FP={e2e['fp']} FN={e2e['fn']}]")
    print("=" * 62)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EMCGCN on MAMS ASTE val/test.")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="Which split to evaluate (default: val).")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT,
                        help=f"Path to EMCGCN checkpoint dir (default: {CHECKPOINT.name}).")
    parser.add_argument("--gold", type=Path, default=None,
                        help="Override gold .dat.aste path directly.")
    args = parser.parse_args()

    gold_path = args.gold or (MAMS_ASTE_DIR / f"{args.split}.dat.aste")
    if not gold_path.exists():
        print(f"Gold file not found: {gold_path}", file=sys.stderr)
        sys.exit(1)
    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading gold annotations from {gold_path.name} …")
    gold_records = _parse_gold(gold_path)
    sentences = [r["sentence"] for r in gold_records]
    print(f"  {len(sentences)} sentences loaded.")

    print(f"Running EMCGCN inference (checkpoint: {args.checkpoint.name}) …")
    pred_triplets = _predict(sentences, args.checkpoint)

    metrics = evaluate(gold_records, pred_triplets)
    print_results(metrics, args.split, gold_path, args.checkpoint)


if __name__ == "__main__":
    main()
