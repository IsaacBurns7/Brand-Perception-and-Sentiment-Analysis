"""ABSA inference: PyABSA span extraction → RoBERTa stage-2 sentiment scoring.

Pipeline per sentence
---------------------
1. PyABSA ATEPC extracts aspect-term spans (text + word positions).
2. Each (sentence, aspect) pair is formatted as ``"{aspect} : {sentence}"``
   and batched through the fine-tuned RoBERTa stage-2 sequence classifier
   stored at artifacts/models/roberta_stage2/.
3. The integer label emitted by RoBERTa (0/1/2) is mapped back to
   negative / neutral / positive.

Usage
-----
    python -m models.absa.predict \\
        --text "The food was amazing but the service was terrible."
    python -m models.absa.predict \\
        --input-path data/processed/mams_aste/test.dat.aste \\
        --output-path artifacts/reports/absa/test_predictions.csv \\
        --atepc-model artifacts/models/absa_aste
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .config import (
    ABSA_MODEL_DIR,
    ABSA_REPORT_DIR,
    EVAL_MANUAL_CSV,
    EVAL_SA_TEST_CSV,
    PROJECT_ROOT,
    ROBERTA_BATCH_SIZE,
    ROBERTA_ID_TO_SENTIMENT,
    ROBERTA_INPUT_TEMPLATE,
    ROBERTA_MAX_LENGTH,
    ROBERTA_SENTIMENT_TO_POLARITY,
    ROBERTA_STAGE2_DIR,
    TEXT_COLUMN_CANDIDATES,
)

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _resolve_device() -> Any:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# RoBERTa sentiment scorer (reusable across calls)
# ---------------------------------------------------------------------------

class _RobertaScorer:
    """Thin wrapper around the fine-tuned RoBERTa stage-2 sequence classifier."""

    def __init__(self, model_path: Path, batch_size: int, max_length: int) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore[import]

        self._device = _resolve_device()
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self._model.to(self._device)
        self._model.eval()

        raw_id2label = self._model.config.id2label
        # Normalise: int key → lowercase string label id
        self._id2label: dict[int, str] = {
            int(k): str(v).lower() for k, v in raw_id2label.items()
        }
        self._batch_size = batch_size
        self._max_length = max_length

    def score(self, texts: list[str]) -> list[str]:
        """Return a sentiment string ('negative'/'neutral'/'positive') for each text."""
        import torch

        predictions: list[str] = []
        with torch.no_grad():
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                encoded = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                logits = self._model(**encoded).logits
                pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
                for pid in pred_ids:
                    raw_label = self._id2label[int(pid)]
                    # Map "0"/"1"/"2" integer-string labels to sentiment words
                    sentiment = ROBERTA_ID_TO_SENTIMENT.get(raw_label, raw_label)
                    predictions.append(sentiment)
        return predictions


# Module-level scorer cache (keyed by resolved model path string)
_SCORER_CACHE: dict[str, _RobertaScorer] = {}


def _get_scorer(
    model_path: Path,
    batch_size: int = ROBERTA_BATCH_SIZE,
    max_length: int = ROBERTA_MAX_LENGTH,
) -> _RobertaScorer:
    key = str(model_path.resolve())
    if key not in _SCORER_CACHE:
        if not model_path.exists():
            raise FileNotFoundError(
                f"RoBERTa stage-2 model not found at '{model_path}'. "
                "Train it first with models/sentiment/train.py."
            )
        _SCORER_CACHE[key] = _RobertaScorer(model_path, batch_size, max_length)
    return _SCORER_CACHE[key]


# ---------------------------------------------------------------------------
# PyABSA span extractor loader
# ---------------------------------------------------------------------------

def _load_atepc_extractor(model_path: Path) -> Any:
    """Load a PyABSA v2 AspectSentimentTripletExtractor from a local checkpoint."""
    try:
        from pyabsa import AspectSentimentTripletExtraction as ASTE  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PyABSA is required for span extraction. "
            "Install with: pip install pyabsa"
        ) from exc

    if not model_path.exists():
        raise FileNotFoundError(
            f"ASTE checkpoint not found: {model_path}. "
            "Train the model first with models/absa/train.py."
        )

    extractor = ASTE.AspectSentimentTripletExtractor(
        checkpoint=str(model_path),
        auto_device=True,
    )
    return extractor


# ---------------------------------------------------------------------------
# Core inference functions
# ---------------------------------------------------------------------------

def extract_spans(
    texts: list[str],
    atepc_model_path: Path = ABSA_MODEL_DIR,
) -> list[dict[str, Any]]:
    """Run PyABSA span extraction and return structured span data per sentence.

    Returns a list of dicts with keys:
        sentence   – original input text
        aspects    – list of aspect term strings
        positions  – list of (word_from, word_to) tuples
    """
    extractor = _load_atepc_extractor(atepc_model_path)
    raw_results = extractor.predict(texts, print_result=False)

    if not isinstance(raw_results, list):
        raw_results = [raw_results]

    outputs: list[dict[str, Any]] = []
    for item in raw_results:
        # PyABSA v2 result dicts use 'sentence' and 'Triplets' keys.
        # Each triplet has 'Aspect', 'Opinion', 'Polarity' — no word positions stored.
        sentence = str(item.get("sentence", ""))
        triplets = item.get("Triplets", []) or []
        aspects: list[str] = [
            str(t.get("Aspect", "")).strip()
            for t in triplets
            if str(t.get("Aspect", "")).strip()
        ]
        # Word positions not available in v2 output; set to sentinel (-1, -1)
        positions: list[tuple[int, int]] = [(-1, -1)] * len(aspects)

        outputs.append(
            {
                "sentence": sentence,
                "aspects": aspects,
                "positions": positions,
            }
        )

    return outputs


def predict_absa(
    texts: list[str],
    atepc_model_path: Path = ABSA_MODEL_DIR,
    roberta_model_path: Path = ROBERTA_STAGE2_DIR,
    roberta_batch_size: int = ROBERTA_BATCH_SIZE,
    roberta_max_length: int = ROBERTA_MAX_LENGTH,
) -> list[dict[str, Any]]:
    """Full ABSA pipeline: span extraction + RoBERTa sentiment scoring.

    Returns one dict per input sentence with keys:
        sentence   – original text
        triplets   – list of {aspect, word_from, word_to, sentiment, polarity} dicts
    """
    # ── 1. Extract spans ────────────────────────────────────────────────────
    span_results = extract_spans(texts, atepc_model_path=atepc_model_path)

    # ── 2. Build flat (sentence, aspect) pairs for batch RoBERTa scoring ───
    scorer = _get_scorer(roberta_model_path, roberta_batch_size, roberta_max_length)

    flat_inputs: list[str] = []
    flat_index: list[tuple[int, int]] = []  # (sentence_idx, aspect_idx)

    for sent_idx, span_item in enumerate(span_results):
        for asp_idx, aspect in enumerate(span_item["aspects"]):
            flat_inputs.append(
                ROBERTA_INPUT_TEMPLATE.format(
                    aspect=aspect,
                    sentence=span_item["sentence"],
                )
            )
            flat_index.append((sent_idx, asp_idx))

    sentiments: list[str] = scorer.score(flat_inputs) if flat_inputs else []

    # ── 3. Reassemble per-sentence results ──────────────────────────────────
    # Pre-populate
    output: list[dict[str, Any]] = [
        {"sentence": item["sentence"], "triplets": []}
        for item in span_results
    ]

    for flat_i, (sent_idx, asp_idx) in enumerate(flat_index):
        span_item = span_results[sent_idx]
        aspect = span_item["aspects"][asp_idx]
        word_from, word_to = span_item["positions"][asp_idx]
        sentiment = sentiments[flat_i]
        polarity = ROBERTA_SENTIMENT_TO_POLARITY.get(sentiment, "NEU")

        output[sent_idx]["triplets"].append(
            {
                "aspect": aspect,
                "word_from": word_from,
                "word_to": word_to,
                "sentiment": sentiment,
                "polarity": polarity,
            }
        )

    return output


def predict_single(
    text: str,
    atepc_model_path: Path = ABSA_MODEL_DIR,
    roberta_model_path: Path = ROBERTA_STAGE2_DIR,
) -> dict[str, Any]:
    """Convenience wrapper for single-sentence ABSA prediction."""
    results = predict_absa(
        [text],
        atepc_model_path=atepc_model_path,
        roberta_model_path=roberta_model_path,
    )
    return results[0]


# ---------------------------------------------------------------------------
# .dat.aste batch inference
# ---------------------------------------------------------------------------

def run_aste_batch_inference(
    input_path: Path,
    output_path: Path,
    atepc_model_path: Path = ABSA_MODEL_DIR,
    roberta_model_path: Path = ROBERTA_STAGE2_DIR,
    roberta_batch_size: int = ROBERTA_BATCH_SIZE,
) -> Path:
    """Score sentences from a .dat.aste file and write predictions to a CSV.

    The input may be a .dat.aste file (with ``#### #### ####`` separators) or a
    plain text file with one sentence per line. True labels from .dat.aste files
    are preserved alongside predictions for downstream evaluation.
    """
    import pandas as pd

    sentences: list[str] = []
    true_labels: list[str] = []  # raw annotation strings, empty for plain text

    with input_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if "#### #### ####" in line:
                parts = line.split("#### #### ####", 1)
                sentences.append(parts[0].strip())
                true_labels.append(parts[1].strip() if len(parts) > 1 else "")
            else:
                sentences.append(line)
                true_labels.append("")

    if not sentences:
        raise ValueError(f"No sentences found in {input_path}.")

    results = predict_absa(
        sentences,
        atepc_model_path=atepc_model_path,
        roberta_model_path=roberta_model_path,
        roberta_batch_size=roberta_batch_size,
    )

    rows: list[dict[str, Any]] = []
    for sent_idx, result in enumerate(results):
        true_label = true_labels[sent_idx] if sent_idx < len(true_labels) else ""
        for triplet in result["triplets"]:
            rows.append(
                {
                    "sentence": result["sentence"],
                    "aspect": triplet["aspect"],
                    "word_from": triplet["word_from"],
                    "word_to": triplet["word_to"],
                    "predicted_sentiment": triplet["sentiment"],
                    "predicted_polarity": triplet["polarity"],
                    "true_annotations": true_label,
                }
            )
        if not result["triplets"]:
            rows.append(
                {
                    "sentence": result["sentence"],
                    "aspect": None,
                    "word_from": None,
                    "word_to": None,
                    "predicted_sentiment": None,
                    "predicted_polarity": None,
                    "true_annotations": true_label,
                }
            )

    output_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Loaded input: {input_path}")
    print(f"Sentences scored: {len(sentences)}")
    print(f"Aspect triplets extracted: {sum(len(r['triplets']) for r in results)}")
    print(f"ATEPC model: {atepc_model_path}")
    print(f"RoBERTa model: {roberta_model_path}")
    print(f"Saved predictions to: {output_path}")
    print("\nFirst 5 rows:")
    print(output_df.head(5).to_string(index=False))

    return output_path


# ---------------------------------------------------------------------------
# CSV batch inference  (eval_sentiment_analysis_test.csv / eval_testdata_manual)
# ---------------------------------------------------------------------------

def _detect_text_column(df: "pd.DataFrame", requested: str | None) -> str:  # type: ignore[name-defined]
    """Return the text column to use, matching the same priority as sentiment/predict.py."""
    import pandas as pd

    if requested is not None:
        if requested not in df.columns:
            raise ValueError(
                f"Requested text column '{requested}' not found. "
                f"Available: {list(df.columns)}"
            )
        return requested
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not detect a text column. Expected one of {TEXT_COLUMN_CANDIDATES}, "
        f"found: {list(df.columns)}"
    )


def run_csv_batch_inference(
    input_path: Path,
    output_path: Path,
    atepc_model_path: Path = ABSA_MODEL_DIR,
    roberta_model_path: Path = ROBERTA_STAGE2_DIR,
    roberta_batch_size: int = ROBERTA_BATCH_SIZE,
    text_column: str | None = None,
) -> Path:
    """Score an eval CSV (``text`` + optional ``label`` columns) and write a
    tidy output CSV with one row per (message, span) pair.

    Output columns
    --------------
    message    – original text from the eval CSV
    span       – aspect term extracted by PyABSA
    sentiment  – negative | neutral | positive  (from RoBERTa stage-2)
    polarity   – NEG | NEU | POS               (shorthand)
    true_label – original integer label from the input CSV (if present, else blank)

    Sentences with no detected aspects appear as a single row with
    span/sentiment/polarity set to None.
    """
    import pandas as pd

    df = pd.read_csv(input_path)
    resolved_col = _detect_text_column(df, text_column)

    messages = df[resolved_col].fillna("").astype(str).str.strip().tolist()
    has_label = "label" in df.columns
    true_labels: list[str] = (
        df["label"].fillna("").astype(str).tolist() if has_label else [""] * len(messages)
    )

    # Filter empty texts but keep index mapping for output alignment
    non_empty_indices = [i for i, m in enumerate(messages) if m]
    non_empty_messages = [messages[i] for i in non_empty_indices]

    if not non_empty_messages:
        raise ValueError(f"No non-empty texts found in column '{resolved_col}' of {input_path}.")

    results = predict_absa(
        non_empty_messages,
        atepc_model_path=atepc_model_path,
        roberta_model_path=roberta_model_path,
        roberta_batch_size=roberta_batch_size,
    )

    rows: list[dict[str, Any]] = []
    for result_idx, result in enumerate(results):
        orig_idx = non_empty_indices[result_idx]
        true_label = true_labels[orig_idx]
        if result["triplets"]:
            for triplet in result["triplets"]:
                rows.append(
                    {
                        "message": result["sentence"],
                        "span": triplet["aspect"],
                        "sentiment": triplet["sentiment"],
                        "polarity": triplet["polarity"],
                        "true_label": true_label,
                    }
                )
        else:
            rows.append(
                {
                    "message": result["sentence"],
                    "span": None,
                    "sentiment": None,
                    "polarity": None,
                    "true_label": true_label,
                }
            )

    if not has_label:
        for row in rows:
            del row["true_label"]

    output_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    total_spans = output_df["span"].notna().sum()
    no_aspect = (output_df["span"].isna()).sum()

    print(f"Loaded dataset  : {input_path}")
    print(f"Text column     : {resolved_col}")
    print(f"Messages scored : {len(non_empty_messages)}")
    print(f"Spans extracted : {total_spans}  (sentences with no aspect: {no_aspect})")
    print(f"ATEPC model     : {atepc_model_path}")
    print(f"RoBERTa model   : {roberta_model_path}")
    print(f"Saved to        : {output_path}")
    print("\nFirst 10 rows:")
    print(output_df.head(10).to_string(index=False))

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ABSA inference: PyABSA span extraction + RoBERTa sentiment scoring."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── single sentence ─────────────────────────────────────────────────────
    p_single = subparsers.add_parser("single", help="Score one sentence and print results.")
    p_single.add_argument("--text", type=str, required=True, help="Input sentence.")
    p_single.add_argument("--atepc-model", type=Path, default=ABSA_MODEL_DIR)
    p_single.add_argument("--roberta-model", type=Path, default=ROBERTA_STAGE2_DIR)

    # ── batch from .dat.aste / plain text ───────────────────────────────────
    p_batch = subparsers.add_parser(
        "batch",
        help="Score a .dat.aste or plain-text file and save a CSV.",
    )
    p_batch.add_argument("--input-path", type=Path, required=True)
    p_batch.add_argument(
        "--output-path",
        type=Path,
        default=ABSA_REPORT_DIR / "batch_predictions.csv",
    )
    p_batch.add_argument("--atepc-model", type=Path, default=ABSA_MODEL_DIR)
    p_batch.add_argument("--roberta-model", type=Path, default=ROBERTA_STAGE2_DIR)
    p_batch.add_argument("--batch-size", type=int, default=ROBERTA_BATCH_SIZE)

    # ── batch from eval CSV (same datasets used for sentiment model eval) ───
    p_csv = subparsers.add_parser(
        "csv-batch",
        help="Score an eval CSV (text + label columns) and write message|span|sentiment rows.",
    )
    p_csv.add_argument(
        "--input-path",
        type=Path,
        default=EVAL_SA_TEST_CSV,
        help=(
            f"Eval CSV to score "
            f"(default: eval_sentiment_analysis_test.csv). "
            f"Also accepts eval_testdata_manual_2009.csv."
        ),
    )
    p_csv.add_argument(
        "--output-path",
        type=Path,
        default=ABSA_REPORT_DIR / "eval_absa_predictions.csv",
    )
    p_csv.add_argument("--atepc-model", type=Path, default=ABSA_MODEL_DIR)
    p_csv.add_argument("--roberta-model", type=Path, default=ROBERTA_STAGE2_DIR)
    p_csv.add_argument("--batch-size", type=int, default=ROBERTA_BATCH_SIZE)
    p_csv.add_argument(
        "--text-column",
        type=str,
        default=None,
        help="Override text column name (auto-detected by default).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "single":
        result = predict_single(
            text=args.text,
            atepc_model_path=args.atepc_model,
            roberta_model_path=args.roberta_model,
        )
        print(f"\nSentence: {result['sentence']}")
        if result["triplets"]:
            for t in result["triplets"]:
                print(
                    f"  Aspect: {t['aspect']!r:20s}  "
                    f"Words [{t['word_from']},{t['word_to']}]  "
                    f"Sentiment: {t['sentiment']} ({t['polarity']})"
                )
        else:
            print("  No aspect terms extracted.")

    elif args.command == "batch":
        run_aste_batch_inference(
            input_path=args.input_path,
            output_path=args.output_path,
            atepc_model_path=args.atepc_model,
            roberta_model_path=args.roberta_model,
            roberta_batch_size=args.batch_size,
        )

    elif args.command == "csv-batch":
        run_csv_batch_inference(
            input_path=args.input_path,
            output_path=args.output_path,
            atepc_model_path=args.atepc_model,
            roberta_model_path=args.roberta_model,
            roberta_batch_size=args.batch_size,
            text_column=args.text_column,
        )


if __name__ == "__main__":
    main()
