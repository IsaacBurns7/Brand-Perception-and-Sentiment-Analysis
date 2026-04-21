"""Inference utilities for the brand sentiment model (sklearn and optional Hugging Face batch)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import (
    DEFAULT_MODEL_PATH,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    TEST_CSV_PATH,
)
from .preprocessing import clean_text

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

_MODEL_CACHE: dict[str, Pipeline] = {}

SKLEARN_BATCH_DEFAULT_INPUT = TEST_CSV_PATH
SKLEARN_BATCH_DEFAULT_OUTPUT = PROCESSED_DATA_DIR / "twitter_sentiment_test_with_predictions.csv"
TEXT_COLUMN_CANDIDATES = ("tweet_text", "clean_comment", "text", "comment")

PROBABILITY_COLUMN_MAP = {
    "Negative emotion": "prob_negative",
    "No emotion toward brand or product": "prob_neutral",
    "Positive emotion": "prob_positive",
}


def load_model(model_path: Path | str = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a trained sentiment model from disk."""

    resolved_path = Path(model_path)
    cache_key = str(resolved_path.resolve())

    if cache_key not in _MODEL_CACHE:
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Model file not found at '{resolved_path}'. Train the model first."
            )
        _MODEL_CACHE[cache_key] = joblib.load(resolved_path)

    return _MODEL_CACHE[cache_key]


def _empty_prediction_result(raw_text: str, cleaned_text_value: str) -> dict[str, Any]:
    """Return a consistent response for invalid or empty inputs."""

    return {
        "input_text": raw_text,
        "cleaned_text": cleaned_text_value,
        "predicted_label": None,
        "confidence": 0.0,
        "class_probabilities": {},
        "error": "Text is empty after cleaning.",
    }


def predict_sentiment(text: str) -> dict[str, Any]:
    """Predict sentiment for a single text input."""

    model = load_model()
    raw_text = text if isinstance(text, str) else ""
    cleaned = clean_text(raw_text)

    if not cleaned:
        return _empty_prediction_result(raw_text=raw_text, cleaned_text_value=cleaned)

    probabilities = model.predict_proba([cleaned])[0]
    class_labels = [str(label) for label in model.classes_]
    class_probabilities = {
        label: float(probability)
        for label, probability in zip(class_labels, probabilities)
    }

    best_label = max(class_probabilities, key=class_probabilities.get)
    confidence = class_probabilities[best_label]

    return {
        "input_text": raw_text,
        "cleaned_text": cleaned,
        "predicted_label": best_label,
        "confidence": float(confidence),
        "class_probabilities": class_probabilities,
        "error": None,
    }


def predict_batch(texts: list[str]) -> list[dict[str, Any]]:
    """Predict sentiment for a batch of texts."""

    model = load_model()
    cleaned_texts = [clean_text(text if isinstance(text, str) else "") for text in texts]

    valid_positions = [index for index, cleaned in enumerate(cleaned_texts) if cleaned]
    results: list[dict[str, Any]] = [
        _empty_prediction_result(raw_text=text if isinstance(text, str) else "", cleaned_text_value=cleaned)
        for text, cleaned in zip(texts, cleaned_texts)
    ]

    if not valid_positions:
        return results

    valid_inputs = [cleaned_texts[index] for index in valid_positions]
    probability_rows = model.predict_proba(valid_inputs)
    class_labels = [str(label) for label in model.classes_]

    for row_index, original_index in enumerate(valid_positions):
        class_probabilities = {
            label: float(probability)
            for label, probability in zip(class_labels, probability_rows[row_index])
        }
        best_label = max(class_probabilities, key=class_probabilities.get)

        results[original_index] = {
            "input_text": texts[original_index] if isinstance(texts[original_index], str) else "",
            "cleaned_text": cleaned_texts[original_index],
            "predicted_label": best_label,
            "confidence": float(class_probabilities[best_label]),
            "class_probabilities": class_probabilities,
            "error": None,
        }

    return results


def detect_text_column(df: pd.DataFrame, requested_column: str | None = None) -> str:
    """Return the text column to use for sklearn batch inference."""

    if requested_column is not None:
        if requested_column not in df.columns:
            raise ValueError(
                f"Requested text column '{requested_column}' was not found. "
                f"Available columns: {list(df.columns)}"
            )
        return requested_column

    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not detect a text column. Expected one of "
        f"{TEXT_COLUMN_CANDIDATES}, but found columns: {list(df.columns)}"
    )


def _base_sklearn_row(original_text: str, cleaned_text_value: str) -> dict[str, Any]:
    return {
        "original_text": original_text,
        "clean_text": cleaned_text_value,
        "predicted_label": None,
        "confidence": 0.0,
        "prob_negative": 0.0,
        "prob_neutral": 0.0,
        "prob_positive": 0.0,
    }


def _apply_probability_columns(
    result: dict[str, Any],
    class_labels: list[str],
    probabilities: list[float],
) -> dict[str, Any]:
    for label, probability in zip(class_labels, probabilities):
        column_name = PROBABILITY_COLUMN_MAP.get(label)
        if column_name is not None:
            result[column_name] = float(probability)
    return result


def run_sklearn_batch_inference(
    input_path: Path = SKLEARN_BATCH_DEFAULT_INPUT,
    output_path: Path = SKLEARN_BATCH_DEFAULT_OUTPUT,
    model_path: Path = DEFAULT_MODEL_PATH,
    text_column: str | None = None,
) -> Path:
    """Run batch inference with a joblib sklearn pipeline and save a CSV with predictions."""

    model = load_model(model_path)
    df = pd.read_csv(input_path)
    resolved_text_column = detect_text_column(df, requested_column=text_column)

    original_texts = df[resolved_text_column].fillna("").astype(str).tolist()
    cleaned_texts = [clean_text(text) for text in original_texts]

    results = [
        _base_sklearn_row(original_text=text, cleaned_text_value=cleaned)
        for text, cleaned in zip(original_texts, cleaned_texts)
    ]
    valid_indices = [index for index, cleaned in enumerate(cleaned_texts) if cleaned]

    if valid_indices:
        valid_cleaned_texts = [cleaned_texts[index] for index in valid_indices]
        probability_rows = model.predict_proba(valid_cleaned_texts)
        class_labels = [str(label) for label in model.classes_]

        for row_index, original_index in enumerate(valid_indices):
            probabilities = probability_rows[row_index]
            predicted_idx = int(probabilities.argmax())
            predicted_label = class_labels[predicted_idx]

            results[original_index]["predicted_label"] = predicted_label
            results[original_index]["confidence"] = float(probabilities[predicted_idx])
            _apply_probability_columns(
                results[original_index],
                class_labels=class_labels,
                probabilities=probabilities.tolist(),
            )

    output_df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    skipped_count = len(df) - len(valid_indices)

    print(f"Loaded dataset: {input_path}")
    print(f"Using text column: {resolved_text_column}")
    print(f"Loaded model: {model_path}")
    print(f"Total rows processed: {len(df)}")
    print(f"Rows skipped due to empty cleaned text: {skipped_count}")
    print(f"Saved predictions to: {output_path}")
    print("\nFirst 5 predictions:")
    print(output_df.head(5).to_string(index=False))

    return output_path


def _resolve_hf_device() -> Any:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_hf_input_dataframe(input_path: Path, *, text_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required_columns = {text_column, label_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {sorted(missing_columns)}")

    working = df.loc[:, [text_column, label_column]].copy()
    working[text_column] = working[text_column].fillna("").astype(str).str.strip()
    working[label_column] = working[label_column].fillna("").astype(str).str.strip().str.lower()
    working = working[(working[text_column] != "") & (working[label_column] != "")].reset_index(drop=True)
    return working


def _load_hf_input_text_only(input_path: Path, *, text_column: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if text_column not in df.columns:
        raise KeyError(f"Missing text column '{text_column}'. Found: {list(df.columns)}")

    working = df.loc[:, [text_column]].copy()
    working[text_column] = working[text_column].fillna("").astype(str).str.strip()
    working = working[working[text_column] != ""].reset_index(drop=True)
    return working


def _build_hf_id2label(model: Any) -> dict[int, str]:
    raw_id2label = model.config.id2label
    return {int(key): str(value).lower() for key, value in raw_id2label.items()}


def run_hf_batch_inference(
    *,
    model_path: Path | str,
    input_path: Path,
    output_path: Path,
    text_column: str = "text",
    label_column: str = "label",
    batch_size: int = 32,
    include_true_labels: bool = True,
) -> Path:
    """Run batch inference with a Hugging Face sequence classifier (local dir or model id on the Hub)."""

    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # pyright: ignore[reportMissingImports]
    import torch

    if include_true_labels:
        df = _load_hf_input_dataframe(input_path, text_column=text_column, label_column=label_column)
    else:
        df = _load_hf_input_text_only(input_path, text_column=text_column)

    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="distilbert_stage2")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, subfolder="distilbert_stage2")
    id2label = _build_hf_id2label(model)
    device = _resolve_hf_device()

    model.to(device)
    model.eval()

    predicted_labels: list[str] = []
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch_texts = df[text_column].iloc[start : start + batch_size].tolist()
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            predicted_ids = logits.argmax(dim=-1).detach().cpu().tolist()
            predicted_labels.extend(id2label[int(label_id)] for label_id in predicted_ids)

    if include_true_labels:
        output_df = pd.DataFrame(
            {
                "text": df[text_column],
                "true_label": df[label_column],
                "predicted_label": predicted_labels,
            }
        )
    else:
        output_df = pd.DataFrame(
            {
                "text": df[text_column],
                "predicted_label": predicted_labels,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Loaded model: {model_path}")
    print(f"Loaded dataset: {input_path}")
    print(f"Rows scored: {len(output_df)}")
    print(f"Device: {device}")
    print(f"Saved predictions to: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inference: sklearn joblib pipeline or Hugging Face classifier."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_sklearn = subparsers.add_parser(
        "sklearn-batch",
        help="Score a CSV with the TF-IDF + logistic regression joblib model.",
    )
    p_sklearn.add_argument("--input-path", type=Path, default=SKLEARN_BATCH_DEFAULT_INPUT)
    p_sklearn.add_argument("--output-path", type=Path, default=SKLEARN_BATCH_DEFAULT_OUTPUT)
    p_sklearn.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    p_sklearn.add_argument("--text-column", type=str, default=None)

    p_hf = subparsers.add_parser(
        "hf-batch",
        help="Score a CSV with a saved Hugging Face sequence classification model directory.",
    )
    p_hf.add_argument("--model-path", type=Path, required=True)
    p_hf.add_argument("--input-path", type=Path, required=True)
    p_hf.add_argument("--output-path", type=Path, required=True)
    p_hf.add_argument("--text-column", type=str, default="text")
    p_hf.add_argument("--label-column", type=str, default="label")
    p_hf.add_argument(
        "--omit-labels",
        action="store_true",
        help="Only require a text column; output text + predicted_label (no true_label).",
    )
    p_hf.add_argument("--batch-size", type=int, default=32)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "sklearn-batch":
        run_sklearn_batch_inference(
            input_path=args.input_path,
            output_path=args.output_path,
            model_path=args.model_path,
            text_column=args.text_column,
        )
    elif args.command == "hf-batch":
        run_hf_batch_inference(
            model_path=args.model_path,
            input_path=args.input_path,
            output_path=args.output_path,
            text_column=args.text_column,
            label_column=args.label_column,
            batch_size=args.batch_size,
            include_true_labels=not args.omit_labels,
        )


if __name__ == "__main__":
    main()
