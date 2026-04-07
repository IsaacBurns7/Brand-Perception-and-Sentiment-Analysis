"""Train and persist the baseline brand sentiment classifier or fine-tune DeBERTa on GoEmotions."""

from __future__ import annotations

import argparse
import inspect
import math
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import (
    CLEAN_TEXT_COLUMN,
    DEBERTA_GOEMOTIONS_MODEL_DIR,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    GOEMOTIONS_LABEL2ID,
    GOEMOTIONS_SEQUENCE_LABELS,
    GOEMOTIONS_TEST_CSV_PATH,
    HF_FINETUNE_MODEL_NAMES,
    LABEL_COLUMN,
    LOGISTIC_REGRESSION_CONFIG,
    MODEL_ARTIFACTS_DIR,
    MODEL_NAME,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    REPORT_ARTIFACTS_DIR,
    TEST_SIZE,
    TEXT_COLUMN,
    TFIDF_CONFIG,
    TRAIN_CSV_PATH,
)
from .evaluate import get_classification_report, serialize_evaluation_results
from .model_factory import get_model
from .prepare_datasets import build_stage_processed_csvs
from .preprocessing import preprocess_dataframe
from .utils import current_timestamp, ensure_directories, format_label_distribution, save_json

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))


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


class EncodedTextDataset(torch.utils.data.Dataset):
    """Tokenized rows for Hugging Face `Trainer`."""

    def __init__(self, encodings: dict[str, list[Any]], label_ids: list[int]) -> None:
        self.encodings = encodings
        self.label_ids = label_ids

    def __len__(self) -> int:
        return len(self.label_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.label_ids[index]
        return item


def _load_goemotions_labeled(
    path: Path,
    *,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in (text_column, label_column):
        if column not in df.columns:
            raise KeyError(f"Missing column '{column}' in {path}. Found: {list(df.columns)}")

    working = df[[text_column, label_column]].copy()
    working[text_column] = working[text_column].fillna("").astype(str).str.strip()
    working[label_column] = working[label_column].fillna("").astype(str).str.strip().str.lower()
    working = working[(working[text_column] != "") & (working[label_column] != "")]
    valid = set(GOEMOTIONS_LABEL2ID)
    unexpected = set(working[label_column].unique()) - valid
    if unexpected:
        raise ValueError(f"Unexpected labels in {path}: {sorted(unexpected)}")
    return working.reset_index(drop=True)


def _hf_compute_metrics(eval_pred: Any) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    logits, label_ids = eval_pred
    predicted_ids = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(label_ids, predicted_ids)),
        "macro_f1": float(f1_score(label_ids, predicted_ids, average="macro")),
        "weighted_f1": float(f1_score(label_ids, predicted_ids, average="weighted")),
    }


def train_deberta_goemotions(
    *,
    input_path: Path,
    output_dir: Path,
    test_path: Path,
    text_column: str,
    label_column: str,
    max_length: int,
    batch_size: int,
    epochs: float,
    learning_rate: float,
    weight_decay: float = 0.01,
    logging_steps: int = 100,
    warmup_ratio: float = 0.06,
    max_train_rows: int | None = None,
    max_eval_rows: int | None = None,
) -> dict[str, Any]:
    """Fine-tune DeBERTa-v3-base on GoEmotions-style CSV (negative / neutral / positive)."""

    import torch
    from transformers import (  # pyright: ignore[reportMissingImports]
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(RANDOM_SEED)

    train_df = _load_goemotions_labeled(input_path, text_column=text_column, label_column=label_column)
    eval_df = _load_goemotions_labeled(test_path, text_column=text_column, label_column=label_column)
    if max_train_rows is not None:
        train_df = train_df.head(max_train_rows).reset_index(drop=True)
    if max_eval_rows is not None:
        eval_df = eval_df.head(max_eval_rows).reset_index(drop=True)

    label_names = list(GOEMOTIONS_SEQUENCE_LABELS)
    spec = get_model("deberta", label_names=label_names)
    tokenizer = spec.tokenizer
    model = spec.estimator

    train_enc = tokenizer(
        train_df[text_column].tolist(),
        truncation=True,
        max_length=max_length,
    )
    eval_enc = tokenizer(
        eval_df[text_column].tolist(),
        truncation=True,
        max_length=max_length,
    )
    train_labels = train_df[label_column].map(GOEMOTIONS_LABEL2ID).astype(int).tolist()
    eval_labels = eval_df[label_column].map(GOEMOTIONS_LABEL2ID).astype(int).tolist()

    train_dataset = EncodedTextDataset(train_enc, train_labels)
    eval_dataset = EncodedTextDataset(eval_enc, eval_labels)

    steps_per_epoch = max(1, math.ceil(len(train_df) / batch_size))
    total_steps = max(1, math.ceil(steps_per_epoch * epochs))
    warmup_steps = max(1, int(round(total_steps * warmup_ratio)))

    ensure_directories((output_dir,))

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_steps": logging_steps,
        "save_total_limit": 2,
        "report_to": "none",
        "fp16": bool(torch.cuda.is_available()),
        "dataloader_pin_memory": bool(torch.cuda.is_available()),
        "seed": RANDOM_SEED,
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    if (
        "use_mps_device" in parameters
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        kwargs["use_mps_device"] = True

    training_args = TrainingArguments(**kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": _hf_compute_metrics,
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = trainer.evaluate()
    print("\nDeBERTa GoEmotions — eval on test split:")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    print(f"\nSaved model and tokenizer to: {output_dir}")

    return {
        "output_dir": str(output_dir),
        "train_rows": len(train_df),
        "eval_rows": len(eval_df),
        "label2id": dict(GOEMOTIONS_LABEL2ID),
        "metrics": metrics,
    }


def _build_int_label_maps(y: pd.Series) -> tuple[dict[int, int], dict[int, str], list[int]]:
    """Map raw integer labels (e.g. 0,2 or 0,1,2) to contiguous model ids 0..k-1."""

    uniq = sorted(y.dropna().unique().astype(int).tolist())
    raw_to_model = {v: i for i, v in enumerate(uniq)}
    id2label = {i: str(uniq[i]) for i in range(len(uniq))}
    return raw_to_model, id2label, uniq


def train_hf_stage_csv(
    *,
    model_name: str,
    input_path: Path,
    output_dir: Path,
    text_column: str,
    label_column: str,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    model_path: Path | None = None,
    max_length: int = 128,
    weight_decay: float = 0.01,
    val_size: float = 0.1,
    logging_steps: int = 200,
    warmup_ratio: float = 0.06,
    gradient_accumulation_steps: int = 1,
    use_mps: bool = True,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Fine-tune a HF encoder on CSV with integer labels (binary 0/2 or 3-class 0/1/2)."""

    import torch
    from transformers import (  # pyright: ignore[reportMissingImports]
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(RANDOM_SEED)

    df = pd.read_csv(input_path, low_memory=False)
    for col in (text_column, label_column):
        if col not in df.columns:
            raise KeyError(f"Missing '{col}' in {input_path}. Columns: {list(df.columns)}")

    work = df[[text_column, label_column]].copy()
    work[text_column] = work[text_column].fillna("").astype(str).str.strip()
    work[label_column] = pd.to_numeric(work[label_column], errors="coerce")
    work = work.dropna(subset=[label_column])
    work[label_column] = work[label_column].astype(int)
    work = work[work[text_column] != ""].reset_index(drop=True)

    if work.empty:
        raise ValueError(f"No labeled rows after cleaning: {input_path}")

    raw_to_model, id2label, uniq_raw = _build_int_label_maps(work[label_column])
    num_labels = len(uniq_raw)
    y_model = work[label_column].map(raw_to_model).astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        work[text_column].tolist(),
        y_model.tolist(),
        test_size=val_size,
        random_state=RANDOM_SEED,
        stratify=y_model,
    )

    label_names = [str(u) for u in uniq_raw]

    if model_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        label2id_cfg = {id2label[i]: i for i in id2label}
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id_cfg,
            ignore_mismatched_sizes=True,
        )
    else:
        spec = get_model(model_name, label_names=label_names)
        tokenizer = spec.tokenizer
        model = spec.estimator

    train_enc = tokenizer(X_train, truncation=True, max_length=max_length)
    val_enc = tokenizer(X_val, truncation=True, max_length=max_length)
    train_dataset = EncodedTextDataset(train_enc, y_train)
    eval_dataset = EncodedTextDataset(val_enc, y_val)

    steps_per_epoch = max(
        1, math.ceil(len(X_train) / (batch_size * max(1, gradient_accumulation_steps)))
    )
    total_steps = max(1, math.ceil(steps_per_epoch * epochs))
    warmup_steps = max(1, int(round(total_steps * warmup_ratio)))

    ensure_directories((output_dir,))

    _mps_on = (
        use_mps
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    eval_bs = min(batch_size, 8) if _mps_on else batch_size

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": eval_bs,
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_steps": logging_steps,
        "save_total_limit": 2,
        "report_to": "none",
        "fp16": bool(torch.cuda.is_available()),
        "dataloader_pin_memory": bool(torch.cuda.is_available()),
        "seed": RANDOM_SEED,
    }
    if gradient_accumulation_steps > 1:
        kwargs["gradient_accumulation_steps"] = gradient_accumulation_steps

    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    if "use_mps_device" in parameters and _mps_on:
        kwargs["use_mps_device"] = True

    training_args = TrainingArguments(**kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": _hf_compute_metrics,
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    resume_kw: str | bool | None = None
    if resume_from_checkpoint:
        token = resume_from_checkpoint.strip().lower()
        if token in ("true", "1", "yes", "last"):
            resume_kw = True
        else:
            resume_kw = str(Path(resume_from_checkpoint).expanduser())

    trainer.train(resume_from_checkpoint=resume_kw)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = trainer.evaluate()
    print(f"\n{model_name} — validation metrics:")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    print(f"\nSaved model and tokenizer to: {output_dir}")

    return {
        "output_dir": str(output_dir),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "raw_labels": uniq_raw,
        "num_labels": num_labels,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    """Parse optional command line arguments for training."""

    parser = argparse.ArgumentParser(
        description="Train sklearn baseline or fine-tune DeBERTa on GoEmotions CSV."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sklearn",
        help="sklearn | deberta (with --dataset goemotions) | "
        + ", ".join(HF_FINETUNE_MODEL_NAMES)
        + " (with --input-path for stage CSVs).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help='Required for DeBERTa: use "goemotions".',
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Training CSV for DeBERTa (e.g. goemotions_train.csv).",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=None,
        help="Eval CSV for DeBERTa during training (default: goemotions_test.csv from config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="HF output directory (DeBERTa GoEmotions default or required for stage CSV training).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional: load tokenizer/model from this dir (e.g. stage-1 checkpoint for stage 2).",
    )
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument(
        "--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=1e-5
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="DeBERTa smoke test: cap training rows (default: use full CSV).",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=None,
        help="DeBERTa smoke test: cap eval rows (default: use full CSV).",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=TRAIN_CSV_PATH,
        help="Path to the labeled training CSV (sklearn only).",
    )
    parser.add_argument(
        "--model-output-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained sklearn model artifact.",
    )
    parser.add_argument(
        "--metadata-output-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Where to save training metadata JSON (sklearn only).",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="HF stage training only: skip rebuilding stage1/stage2/eval CSVs before training.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        "--gradient_accumulation_steps",
        dest="gradient_accumulation_steps",
        type=int,
        default=1,
        help="HF stage training: accumulate gradients (lower memory; effective batch = batch_size × this).",
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        help="HF stage training: use CPU instead of Apple MPS (slower, avoids MPS OOM).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        metavar="PATH_OR_TRUE",
        help="HF stage training: resume from checkpoint-* dir, or 'true' / 'last' for latest under --output-dir.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for training."""

    args = parse_args()
    model_key = args.model.strip().lower()

    if model_key == "deberta" and args.dataset == "goemotions":
        if args.input_path is None:
            print("Error: --input-path is required for DeBERTa GoEmotions.", file=sys.stderr)
            sys.exit(1)
        output_dir = args.output_dir if args.output_dir is not None else DEBERTA_GOEMOTIONS_MODEL_DIR
        test_path = args.test_path if args.test_path is not None else GOEMOTIONS_TEST_CSV_PATH
        train_deberta_goemotions(
            input_path=args.input_path,
            output_dir=output_dir,
            test_path=test_path,
            text_column=args.text_column,
            label_column=args.label_column,
            max_length=args.max_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_train_rows=args.max_train_rows,
            max_eval_rows=args.max_eval_rows,
        )
        return

    if model_key in {m.lower() for m in HF_FINETUNE_MODEL_NAMES}:
        if args.input_path is None:
            print(
                f"Error: --input-path is required when --model is {model_key} (stage CSV).",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.output_dir is None:
            print("Error: --output-dir is required for stage CSV fine-tuning.", file=sys.stderr)
            sys.exit(1)
        if not args.skip_prepare:
            build_stage_processed_csvs()
        train_hf_stage_csv(
            model_name=model_key,
            input_path=args.input_path,
            output_dir=args.output_dir,
            text_column=args.text_column,
            label_column=args.label_column,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            model_path=args.model_path,
            max_length=args.max_length,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_mps=not args.no_mps,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        return

    if model_key == "sklearn":
        train_model(
            dataset_path=args.dataset_path,
            model_output_path=args.model_output_path,
            metadata_output_path=args.metadata_output_path,
        )
        return

    print(
        f"Error: unsupported --model {args.model!r}. "
        f"Use sklearn, deberta+--dataset goemotions, or one of: {', '.join(HF_FINETUNE_MODEL_NAMES)}.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
