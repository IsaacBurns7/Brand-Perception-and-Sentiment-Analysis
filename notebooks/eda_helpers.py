"""Shared helpers for dataset EDA notebooks (pandas + matplotlib only)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\u2600-\u26FF\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)

SENTIMENT140_COLUMNS = ["polarity", "tweet_id", "date", "query", "user", "text"]


def resolve_project_root() -> Path:
    here = Path.cwd().resolve()
    if (here / "data" / "datasets").is_dir():
        return here
    if (here.parent / "data" / "datasets").is_dir():
        return here.parent
    raise FileNotFoundError(
        "Could not find data/datasets. Run the notebook from the repo root or notebooks/."
    )


def datasets_root(project_root: Path | None = None) -> Path:
    root = project_root or resolve_project_root()
    return root / "data" / "datasets"


def eda_summary_dir(project_root: Path | None = None) -> Path:
    root = project_root or resolve_project_root()
    d = root / "data" / "processed" / "eda_summaries"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_csv_safe(path: Path, *, max_rows: int | None = None, **kwargs: Any) -> pd.DataFrame:
    # Do not limit rows by default — full dataset needed for accurate EDA
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    read_kw: dict[str, Any] = {"low_memory": False, **kwargs}
    if max_rows is not None:
        read_kw["nrows"] = max_rows
    try:
        return pd.read_csv(path, encoding="utf-8", **read_kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **read_kw)


def load_sentiment140(path: Path, *, max_rows: int | None = None) -> pd.DataFrame:
    return read_csv_safe(
        path,
        max_rows=max_rows,
        header=None,
        names=SENTIMENT140_COLUMNS,
        on_bad_lines="skip",
    )


def char_word_lengths(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = s.fillna("").astype(str)
    char_len = s.str.len()
    word_len = s.str.split().str.len().clip(lower=0)
    return char_len, word_len


def text_pattern_stats(text_series: pd.Series) -> dict[str, float]:
    s = text_series.fillna("").astype(str)
    nonempty = s.str.len() > 0
    if not nonempty.any():
        return {"url_pct": 0.0, "mention_pct": 0.0, "hashtag_pct": 0.0, "emoji_pct": 0.0}
    sub = s[nonempty]
    return {
        "url_pct": 100.0 * sub.str.contains(URL_RE, regex=True).mean(),
        "mention_pct": 100.0 * sub.str.contains(MENTION_RE, regex=True).mean(),
        "hashtag_pct": 100.0 * sub.str.contains(HASHTAG_RE, regex=True).mean(),
        "emoji_pct": 100.0 * sub.str.contains(EMOJI_RE, regex=True).mean(),
    }


def plot_label_counts(counts: pd.Series, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.sort_index().plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("label")
    ax.set_ylabel("count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_hist_length(lengths: pd.Series, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    clip = lengths.quantile(0.99) if len(lengths) else 0
    lengths.clip(upper=clip).hist(bins=40, ax=ax)
    ax.set_title(title + " (clipped at 99th pct for display)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.show()


def duplicate_text_pct(text_series: pd.Series) -> float:
    s = text_series.fillna("").astype(str)
    if len(s) == 0:
        return 0.0
    return 100.0 * float(s.duplicated().sum()) / float(len(s))


def null_fraction_df(df: pd.DataFrame) -> float:
    if df.size == 0:
        return 0.0
    return float(df.isna().to_numpy().mean())


def interpret_reddit_labels(unique_vals: list[Any]) -> str:
    u = {str(v) for v in unique_vals if pd.notna(v)}
    if u <= {"0", "1"}:
        return "Binary string labels 0/1: map to neg/pos; confirm polarity from data card."
    if u <= {"0", "4"}:
        return "Labels 0 and 4 only in sample: matches Sentiment140-style neg (0) / pos (4); file may be block-sorted—scan full CSV for class 2 (neutral)."
    if u <= {"0", "2", "4"}:
        return "Sentiment140-style 0/2/4: remap to 0/1/2 for 3-class training."
    return f"Inspect documentation; unique in sample: {sorted(u)[:10]}"


def save_eda_summary(slug: str, payload: dict[str, Any], project_root: Path | None = None) -> Path:
    root = project_root or resolve_project_root()
    out = eda_summary_dir(root) / f"{slug}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def load_eda_summary(slug: str, project_root: Path | None = None) -> dict[str, Any] | None:
    root = project_root or resolve_project_root()
    path = eda_summary_dir(root) / f"{slug}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
