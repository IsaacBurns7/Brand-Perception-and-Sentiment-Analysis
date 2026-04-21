"""Build standardized stage-1 (binary) and stage-2 (3-class) training CSVs plus eval sets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DATASETS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, TRAIN_CSV_PATH

SENTIMENT140_TRAIN = DATASETS_DIR / "sentiment140" / "training.1600000.processed.noemoticon.csv"
REDDIT_ML_READY = DATASETS_DIR / "reddit_comments" / "reddit_comments_ml_ready.csv"
SA_DIR = DATASETS_DIR / "sentiment-analysis"
SA_TRAIN = SA_DIR / "train.csv"
SA_TEST = SA_DIR / "test.csv"
TESTDATA_MANUAL = SA_DIR / "testdata.manual.2009.06.14.csv"
TWITTER_TRAIN = TRAIN_CSV_PATH

STAGE1_OUT = PROCESSED_DATA_DIR / "stage1_pretrain.csv"
STAGE2_OUT = PROCESSED_DATA_DIR / "stage2_finetune.csv"
EVAL_SA_TEST_OUT = PROCESSED_DATA_DIR / "eval_sentiment_analysis_test.csv"
EVAL_MANUAL_OUT = PROCESSED_DATA_DIR / "eval_testdata_manual_2009.csv"

NO_HEADER_6 = ["polarity", "id", "date", "query", "user", "text"]

TWITTER_LABEL_TO_ID = {
    "negative emotion": 0,
    "no emotion toward brand or product": 1,
    "positive emotion": 2,
}
EXCLUDED_TWITTER_LABEL = "i can't tell"


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False, **kwargs)


def _standardize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def load_sentiment140(path: Path = SENTIMENT140_TRAIN) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = _read_csv(path, header=None, names=NO_HEADER_6, on_bad_lines="skip")
    text = _standardize_text(df["text"])
    pol = pd.to_numeric(df["polarity"], errors="coerce")
    out = pd.DataFrame({"text": text, "label": pol})
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    out = out[out["label"].isin((0, 4))]
    out["label"] = out["label"].map({0: 0, 4: 2})
    out = out[out["text"] != ""].reset_index(drop=True)
    return out


def load_reddit(path: Path = REDDIT_ML_READY) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = _read_csv(path, on_bad_lines="skip")
    text_col = "clean_text" if "clean_text" in df.columns else None
    if text_col is None:
        for c in df.columns:
            if "text" in c.lower() and "id" not in c.lower():
                text_col = c
                break
    if text_col is None or "label" not in df.columns:
        raise KeyError(f"Expected clean_text/text + label in {path}; got {list(df.columns)}")
    text = _standardize_text(df[text_col])
    lab = pd.to_numeric(df["label"], errors="coerce")
    out = pd.DataFrame({"text": text, "label": lab})
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    out = out[out["label"].isin((0, 4))]
    out["label"] = out["label"].map({0: 0, 4: 2})
    out = out[out["text"] != ""].reset_index(drop=True)
    return out


def load_sentiment_analysis_train(path: Path = SA_TRAIN) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = _read_csv(path, on_bad_lines="skip")
    if "text" not in df.columns or "sentiment" not in df.columns:
        raise KeyError(f"Expected text + sentiment in {path}; got {list(df.columns)}")
    text = _standardize_text(df["text"])
    raw = df["sentiment"].astype(str).str.strip().str.lower()
    mapped = raw.map({"negative": 0, "neutral": 1, "positive": 2})
    out = pd.DataFrame({"text": text, "label": mapped})
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    out = out[out["text"] != ""].reset_index(drop=True)
    return out


def load_twitter_train(path: Path = TWITTER_TRAIN) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = _read_csv(path, on_bad_lines="skip")
    text_col = "tweet_text"
    label_col = "is_there_an_emotion_directed_at_a_brand_or_product"
    if text_col not in df.columns or label_col not in df.columns:
        raise KeyError(f"Expected {text_col} + {label_col} in {path}; got {list(df.columns)}")
    text = _standardize_text(df[text_col])
    raw = df[label_col].fillna("").astype(str).str.strip().str.lower()
    mask_ex = raw == EXCLUDED_TWITTER_LABEL
    mapped = raw.map(TWITTER_LABEL_TO_ID)
    out = pd.DataFrame({"text": text, "label": mapped})
    out = out[~mask_ex]
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    out = out[out["text"] != ""].reset_index(drop=True)
    return out


def load_sentiment_analysis_test(path: Path = SA_TEST) -> pd.DataFrame:
    return load_sentiment_analysis_train(path)


def load_testdata_manual(path: Path = TESTDATA_MANUAL) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = _read_csv(path, header=None, names=NO_HEADER_6, on_bad_lines="skip")
    text = _standardize_text(df["text"])
    pol = pd.to_numeric(df["polarity"], errors="coerce")
    out = pd.DataFrame({"text": text, "label": pol})
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    out = out[out["label"].isin((0, 2, 4))]
    out["label"] = out["label"].map({0: 0, 2: 1, 4: 2})
    out = out[out["text"] != ""].reset_index(drop=True)
    return out


def _print_dist(name: str, s: pd.Series) -> None:
    print(f"\n{name} — rows: {len(s)}")
    print(s.value_counts().sort_index())


def print_pipeline_commands() -> None:
    models = ("distilbert", "roberta", "deberta", "bertweet")
    print("\n" + "=" * 72)
    print("TRAINING (run from project root)")
    print("Note: each train run rebuilds stage1/stage2/eval CSVs first unless you pass --skip-prepare.")
    print("=" * 72)
    for m in models:
        print(f"\n##########################\n# {m}\n##########################\n")
        print("Stage 1:")
        print(
            f"python -m models.sentiment.train \\\n"
            f"  --model {m} \\\n"
            f"  --input-path data/processed/stage1_pretrain.csv \\\n"
            f"  --text-column text \\\n"
            f"  --label-column label \\\n"
            f"  --epochs 1 \\\n"
            f"  --batch_size 16 \\\n"
            f"  --learning_rate 2e-5 \\\n"
            f"  --output-dir artifacts/models/{m}_stage1\n"
        )
        print("Stage 2:")
        print(
            f"python -m models.sentiment.train \\\n"
            f"  --model {m} \\\n"
            f"  --input-path data/processed/stage2_finetune.csv \\\n"
            f"  --text-column text \\\n"
            f"  --label-column label \\\n"
            f"  --epochs 3 \\\n"
            f"  --batch_size 16 \\\n"
            f"  --learning_rate 1e-5 \\\n"
            f"  --model-path artifacts/models/{m}_stage1 \\\n"
            f"  --output-dir artifacts/models/{m}_stage2\n"
        )

    print("\n" + "=" * 72)
    print("EVALUATION (after stage-2 checkpoints exist)")
    print("=" * 72)
    eval_sets = (
        ("data/processed/eval_sentiment_analysis_test.csv", "eval_sa_test"),
        ("data/processed/eval_testdata_manual_2009.csv", "eval_manual_2009"),
    )
    for m in models:
        for csv_path, tag in eval_sets:
            pred_path = f"artifacts/reports/{m}_stage2_{tag}_predictions.csv"
            print(f"\n# {m} — {tag}")
            print(
                f"python -m models.sentiment.predict hf-batch \\\n"
                f"  --model-path artifacts/models/{m}_stage2 \\\n"
                f"  --input-path {csv_path} \\\n"
                f"  --output-path {pred_path} \\\n"
                f"  --text-column text \\\n"
                f"  --label-column label \\\n"
                f"  --batch-size 32\n"
            )
            print(f"python -m models.sentiment.hf_predictions_metrics {pred_path}\n")


def build_stage_processed_csvs() -> None:
    """Load raw sources, standardize labels, write stage1/stage2 train CSVs and eval CSVs."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing datasets (standardizing sources → stage1/stage2 + eval CSVs)…")
    s140 = load_sentiment140()
    reddit = load_reddit()
    sa_train = load_sentiment_analysis_train()
    tw = load_twitter_train()

    stage1 = pd.concat([s140, reddit], ignore_index=True)
    stage1 = stage1.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    stage1 = stage1[stage1["label"].isin((0, 2))]

    stage2 = pd.concat([sa_train, tw], ignore_index=True)
    stage2 = stage2.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    stage2 = stage2[stage2["label"].isin((0, 1, 2))]

    stage1.to_csv(STAGE1_OUT, index=False)
    stage2.to_csv(STAGE2_OUT, index=False)

    print(f"\nWrote {STAGE1_OUT} ({len(stage1):,} rows)")
    _print_dist("Stage 1 label", stage1["label"])
    print(f"\nWrote {STAGE2_OUT} ({len(stage2):,} rows)")
    _print_dist("Stage 2 label", stage2["label"])

    ev_test = load_sentiment_analysis_test()
    ev_manual = load_testdata_manual()
    ev_test.to_csv(EVAL_SA_TEST_OUT, index=False)
    ev_manual.to_csv(EVAL_MANUAL_OUT, index=False)
    print(f"\nWrote {EVAL_SA_TEST_OUT} ({len(ev_test):,} rows)")
    _print_dist("Eval sentiment-analysis test", ev_test["label"])
    print(f"\nWrote {EVAL_MANUAL_OUT} ({len(ev_manual):,} rows)")
    _print_dist("Eval testdata.manual", ev_manual["label"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare stage1/stage2 CSVs and eval sets.")
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="After building CSVs, print train + eval shell commands.",
    )
    parser.add_argument(
        "--commands-only",
        action="store_true",
        help="Only print train + eval commands (skip CSV writes).",
    )
    args = parser.parse_args()

    if args.commands_only:
        print_pipeline_commands()
        return

    build_stage_processed_csvs()

    if args.print_commands:
        print_pipeline_commands()


if __name__ == "__main__":
    main()
