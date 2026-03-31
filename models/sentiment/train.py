"""Train and persist the baseline brand sentiment classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import (
    CLEAN_TEXT_COLUMN,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    LABEL_COLUMN,
    LOGISTIC_REGRESSION_CONFIG,
    MODEL_ARTIFACTS_DIR,
    MODEL_NAME,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    REPORT_ARTIFACTS_DIR,
    TEST_SIZE,
    TEXT_COLUMN,
    TFIDF_CONFIG,
    TRAIN_CSV_PATH,
)
from .evaluate import get_classification_report, serialize_evaluation_results
from .preprocessing import preprocess_dataframe
from .utils import current_timestamp, ensure_directories, format_label_distribution, save_json


def build_pipeline() -> Pipeline:
    """Create the baseline TF-IDF + Logistic Regression pipeline."""

    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
            ("classifier", LogisticRegression(**LOGISTIC_REGRESSION_CONFIG)),
        ]
    )


def train_model(
    dataset_path: Path = TRAIN_CSV_PATH,
    model_output_path: Path = DEFAULT_MODEL_PATH,
    metadata_output_path: Path = DEFAULT_METADATA_PATH,
) -> dict[str, Any]:
    """Train the model, save artifacts, and return training metadata."""

    ensure_directories((MODEL_ARTIFACTS_DIR, REPORT_ARTIFACTS_DIR, PROCESSED_DATA_DIR))

    raw_df = pd.read_csv(dataset_path)
    raw_row_count = len(raw_df)

    cleaned_df = preprocess_dataframe(raw_df, text_col=TEXT_COLUMN, label_col=LABEL_COLUMN)
    cleaned_row_count = len(cleaned_df)

    labeled_df = cleaned_df.dropna(subset=[LABEL_COLUMN]).reset_index(drop=True)
    labeled_row_count = len(labeled_df)

    if labeled_df.empty:
        raise ValueError("No labeled rows remain after preprocessing.")

    if labeled_df[LABEL_COLUMN].nunique() < 2:
        raise ValueError("At least two sentiment classes are required for training.")

    X = labeled_df[CLEAN_TEXT_COLUMN].astype(str)
    y = labeled_df[LABEL_COLUMN].astype(str)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)

    label_order = sorted(y.unique())
    evaluation_summary = serialize_evaluation_results(y_val, y_val_pred, labels=label_order)

    joblib.dump(pipeline, model_output_path)

    metadata = {
        "timestamp": current_timestamp(),
        "model_name": MODEL_NAME,
        "dataset_path": str(Path(dataset_path)),
        "text_column": TEXT_COLUMN,
        "label_column": LABEL_COLUMN,
        "row_counts": {
            "raw_rows": raw_row_count,
            "rows_after_preprocessing": cleaned_row_count,
            "rows_used_for_training": labeled_row_count,
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
        },
        "label_distribution_after_preprocessing": format_label_distribution(
            labeled_df[LABEL_COLUMN]
        ),
        "model_parameters": {
            "tfidf": TFIDF_CONFIG,
            "logistic_regression": LOGISTIC_REGRESSION_CONFIG,
            "random_seed": RANDOM_SEED,
            "test_size": TEST_SIZE,
        },
        "validation_metrics": evaluation_summary,
    }
    save_json(metadata, metadata_output_path)

    print(f"Loaded dataset: {dataset_path}")
    print(f"Raw rows: {raw_row_count}")
    print(f"Rows after preprocessing: {cleaned_row_count}")
    print(f"Rows used for training: {labeled_row_count}")
    print(f"Train rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"\nAccuracy: {evaluation_summary['accuracy']:.4f}")
    print(f"Macro F1: {evaluation_summary['macro_f1']:.4f}")
    print(f"Weighted F1: {evaluation_summary['weighted_f1']:.4f}")
    print("\nClassification Report:")
    print(get_classification_report(y_val, y_val_pred))
    print("\nConfusion Matrix:")
    for row in evaluation_summary["confusion_matrix"]:
        print(row)
    print(f"\nSaved model to: {model_output_path}")
    print(f"Saved metadata to: {metadata_output_path}")

    return metadata


def parse_args() -> argparse.Namespace:
    """Parse optional command line arguments for training."""

    parser = argparse.ArgumentParser(description="Train the brand sentiment baseline model.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=TRAIN_CSV_PATH,
        help="Path to the labeled training CSV.",
    )
    parser.add_argument(
        "--model-output-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model artifact.",
    )
    parser.add_argument(
        "--metadata-output-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Where to save training metadata JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for training."""

    args = parse_args()
    train_model(
        dataset_path=args.dataset_path,
        model_output_path=args.model_output_path,
        metadata_output_path=args.metadata_output_path,
    )


if __name__ == "__main__":
    main()
