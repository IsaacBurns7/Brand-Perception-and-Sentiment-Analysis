"""
Reusable EDA utilities for the Brand Sentiment dataset.
All official outputs are saved under EDA/output/.
"""

import os
import re
from collections import Counter
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from nltk.corpus import stopwords

    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = set(ENGLISH_STOP_WORDS)

OUTPUT_DIR = "EDA/output"
DEFAULT_PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
PLOTS_DIR = DEFAULT_PLOTS_DIR
TEXT_DIR = os.path.join(OUTPUT_DIR, "text")
REPORT_PATH = os.path.join(TEXT_DIR, "eda_report.txt")

MAIN_SENTIMENTS = [
    "Positive emotion",
    "Negative emotion",
    "No emotion toward brand or product",
]

SENTIMENT_FILE_SUFFIXES = {
    "Positive emotion": "positive",
    "Negative emotion": "negative",
    "No emotion toward brand or product": "neutral",
}


def ensure_output_dirs():
    """Create output directories used by the EDA workflow."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)


def set_plots_dir(directory):
    """Point plot saving to a notebook-specific directory."""
    global PLOTS_DIR
    PLOTS_DIR = directory
    ensure_output_dirs()
    return PLOTS_DIR


def set_report_path(filename):
    """Point report writing to a specific file inside EDA/output/text/."""
    global REPORT_PATH
    ensure_output_dirs()
    REPORT_PATH = os.path.join(TEXT_DIR, filename)
    return REPORT_PATH


def reset_report():
    """Clear the report file so a notebook run starts fresh."""
    ensure_output_dirs()
    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write("")


def write_report(text):
    """Append formatted text to the EDA report file."""
    ensure_output_dirs()
    with open(REPORT_PATH, "a", encoding="utf-8") as report_file:
        report_file.write(f"{text.rstrip()}\n\n")


def write_section(title):
    """Write a clearly separated section header to the report file."""
    header = "=" * 20
    write_report(f"{header}\n{title.upper()}\n{header}")


def infer_text_and_label_columns(df):
    """Infer likely text and label columns and write the selection to the report."""
    columns = list(df.columns)
    lower_map = {column: column.lower() for column in columns}

    text_keywords = ["tweet", "text", "review", "sentence", "content", "comment", "message", "post"]
    label_keywords = ["label", "sentiment", "target", "class", "polarity"]

    text_candidates = [
        column for column in columns if any(keyword in lower_map[column] for keyword in text_keywords)
    ]
    if not text_candidates:
        object_columns = [
            column for column in columns
            if df[column].dtype == "object" and df[column].fillna("").astype(str).str.len().mean() > 20
        ]
        text_candidates = object_columns or [columns[-1]]

    label_candidates = [
        column for column in columns if any(keyword in lower_map[column] for keyword in label_keywords)
    ]
    if not label_candidates:
        low_cardinality_columns = []
        for column in columns:
            unique_count = df[column].nunique(dropna=True)
            if 1 < unique_count <= min(20, max(2, int(len(df) * 0.05))):
                low_cardinality_columns.append(column)
        label_candidates = low_cardinality_columns or [columns[1] if len(columns) > 1 else columns[0]]

    text_col = max(
        text_candidates,
        key=lambda column: df[column].fillna("").astype(str).str.len().mean(),
    )
    label_col = min(
        label_candidates,
        key=lambda column: df[column].nunique(dropna=True),
    )

    write_report(
        "Inferred schema for this dataset:\n"
        f"Text column: {text_col}\n"
        f"Label column: {label_col}\n"
        f"Text candidates considered: {text_candidates}\n"
        f"Label candidates considered: {label_candidates}"
    )
    return text_col, label_col


def dataset_overview_detailed(df):
    """Write shape, columns, info, and describe(include='all')."""
    buffer = StringIO()
    df.info(buf=buffer)
    write_report(
        f"Shape: {df.shape}\n\n"
        f"Column names: {list(df.columns)}\n\n"
        "--- df.info() ---\n"
        f"{buffer.getvalue().rstrip()}\n\n"
        "--- df.describe(include='all') ---\n"
        f"{df.describe(include='all').to_string()}"
    )
    return {
        "shape": df.shape,
        "columns": list(df.columns),
    }


def load_data(filepath):
    """Load CSV into a pandas DataFrame."""
    ensure_output_dirs()
    return pd.read_csv(filepath)


def _string_series(df, column):
    return df[column].fillna("").astype(str)


def _normalize_text(text, keep_hashtags=False):
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+|\{link\}", " ", text)
    text = re.sub(r"@\w+|@mention", " ", text)
    if not keep_hashtags:
        text = re.sub(r"#\w+", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
    else:
        text = re.sub(r"[^a-z0-9#\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize_text(text):
    cleaned_text = _normalize_text(text)
    return cleaned_text.split() if cleaned_text else []


def _tokens_to_series(tokens, top_n):
    if not tokens:
        return pd.Series(dtype=int)
    counts = Counter(tokens)
    return pd.Series(counts).sort_values(ascending=False).head(top_n)


def _plot_barh(series, title, xlabel, plot_path, figsize=(10, 6)):
    ensure_output_dirs()
    plt.figure(figsize=figsize)
    if series.empty:
        plt.text(0.5, 0.5, "No data available", ha="center", va="center")
        plt.axis("off")
    else:
        plot_series = series.sort_values(ascending=True)
        plt.barh(plot_series.index.astype(str), plot_series.values)
        plt.xlabel(xlabel)
        plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


def _uppercase_ratio(text):
    text = "" if pd.isna(text) else str(text)
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    uppercase_letters = sum(char.isupper() for char in letters)
    return uppercase_letters / len(letters)


def _stopword_ratio(text):
    tokens = _tokenize_text(text)
    if not tokens:
        return 0.0
    stopword_count = sum(token in STOP_WORDS for token in tokens)
    return stopword_count / len(tokens)


def _only_placeholders_after_cleaning(text):
    if pd.isna(text):
        return False
    raw_text = str(text).strip()
    if not raw_text:
        return False
    stripped = re.sub(
        r"http\S+|www\.\S+|\{link\}|@\w+|@mention|#\w+",
        " ",
        raw_text,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(r"[\W_]+", " ", stripped).strip()
    return stripped == ""


def _top_ngrams_from_texts(texts, top_n=20):
    cleaned_texts = [_normalize_text(text) for text in texts]
    cleaned_texts = [text for text in cleaned_texts if text]
    if not cleaned_texts:
        return pd.Series(dtype=int)

    try:
        vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words="english")
        matrix = vectorizer.fit_transform(cleaned_texts)
    except ValueError:
        return pd.Series(dtype=int)

    ngram_counts = np.asarray(matrix.sum(axis=0)).ravel()
    feature_names = vectorizer.get_feature_names_out()
    return (
        pd.Series(ngram_counts, index=feature_names)
        .sort_values(ascending=False)
        .head(top_n)
    )


def dataset_overview(df):
    """Write a compact dataset overview useful for modeling setup."""
    buffer = StringIO()
    df.info(buf=buffer)
    overview_text = "\n".join(
        [
            f"Shape: {df.shape}",
            "",
            f"Column names: {list(df.columns)}",
            "",
            "--- df.info() ---",
            buffer.getvalue().rstrip(),
        ]
    )
    write_report(overview_text)


def missing_value_summary(df):
    """Write missing value counts and proportions per column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        write_report("No missing values.")
    else:
        write_report(
            "Missing value counts:\n"
            f"{missing.to_string()}\n\n"
            "Missing value proportions:\n"
            f"{(missing / len(df)).to_string()}"
        )
    return missing


def duplicate_summary(df, text_col):
    """Write duplicate counts and unique text ratio."""
    n_duplicates = int(df.duplicated().sum())
    n_text_duplicates = int(df[text_col].duplicated().sum())
    unique_ratio = df[text_col].nunique() / max(len(df), 1)
    summary = pd.DataFrame(
        {
            "metric": ["duplicate_rows", "duplicate_texts", "unique_text_ratio"],
            "value": [n_duplicates, n_text_duplicates, round(unique_ratio, 4)],
        }
    )
    write_report(
        f"Duplicate rows: {n_duplicates}\n"
        f"Duplicate texts (in '{text_col}'): {n_text_duplicates}\n"
        f"Unique text ratio: {unique_ratio:.4f}"
    )
    return summary


def sentiment_distribution(df, sentiment_col):
    """Create and save a count plot of the target distribution."""
    ensure_output_dirs()
    plot_path = os.path.join(PLOTS_DIR, "sentiment_distribution.png")
    order = df[sentiment_col].value_counts(dropna=False).index
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, y=sentiment_col, order=order)
    plt.title("Sentiment distribution")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    write_report(
        "Sentiment counts:\n"
        f"{df[sentiment_col].value_counts(dropna=False).to_string()}\n\n"
        f"Saved plot: {plot_path}"
    )
    return plot_path


def text_length_features(df, text_col):
    """Add text_length and word_count columns to df in place."""
    text_series = _string_series(df, text_col)
    df["text_length"] = text_series.str.len()
    df["word_count"] = text_series.str.split().str.len()
    write_report(
        "Text length statistics:\n"
        f"{df['text_length'].describe().to_string()}\n\n"
        "Word count statistics:\n"
        f"{df['word_count'].describe().to_string()}"
    )
    return df


def plot_text_length_distribution(df, text_col):
    """Create and save a histogram of text length."""
    ensure_output_dirs()
    if "text_length" not in df.columns:
        text_length_features(df, text_col)
    plot_path = os.path.join(PLOTS_DIR, "text_length_distribution.png")
    plt.figure(figsize=(8, 5))
    plt.hist(df["text_length"], bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Text length (characters)")
    plt.ylabel("Frequency")
    plt.title("Distribution of text length")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    write_report(f"Saved plot: {plot_path}")
    return plot_path


def clean_and_tokenize(df, text_col):
    """Return a flattened list of basic-cleaned tokens."""
    tokens = []
    for text in df[text_col].fillna(""):
        tokens.extend(_tokenize_text(text))
    write_report(f"Generated {len(tokens)} tokens from column '{text_col}'.")
    return tokens


def word_frequency(tokens):
    """Return top 20 words and the vocabulary size."""
    top20 = _tokens_to_series(tokens, top_n=20)
    vocab_size = len(set(tokens))
    write_report(
        "Top 20 words:\n"
        f"{top20.to_string()}\n\n"
        f"Vocabulary size: {vocab_size}"
    )
    return top20, vocab_size


def vocabulary_statistics(tokens):
    """Write compact vocabulary statistics without noisy token frequency tables."""
    token_count = len(tokens)
    vocab_size = len(set(tokens))
    type_token_ratio = vocab_size / token_count if token_count else 0.0
    write_report(
        f"Token count: {token_count}\n"
        f"Vocabulary size: {vocab_size}\n"
        f"Type-token ratio: {type_token_ratio:.4f}"
    )
    return pd.DataFrame(
        {
            "metric": ["token_count", "vocabulary_size", "type_token_ratio"],
            "value": [token_count, vocab_size, round(type_token_ratio, 4)],
        }
    )


def tfidf_diagnostics(df, text_col):
    """Build a TF-IDF matrix and write sparsity diagnostics."""
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    matrix = vectorizer.fit_transform(df[text_col].fillna(""))
    n_docs, n_features = matrix.shape
    sparsity = 1.0 - (matrix.nnz / (n_docs * n_features))
    nnz_per_doc = np.array(matrix.getnnz(axis=1))
    write_report(
        f"TF-IDF matrix shape: {matrix.shape}\n"
        f"Sparsity: {sparsity}\n"
        "Active features per document statistics:\n"
        "mean: {:.2f}\nstd: {:.2f}\nmin: {}\n25%: {:.2f}\nmedian: {:.2f}\n75%: {:.2f}\nmax: {}".format(
            nnz_per_doc.mean(),
            nnz_per_doc.std(),
            nnz_per_doc.min(),
            np.percentile(nnz_per_doc, 25),
            np.median(nnz_per_doc),
            np.percentile(nnz_per_doc, 75),
            nnz_per_doc.max(),
        )
    )
    return nnz_per_doc


def plot_tfidf_feature_distribution(nnz_per_doc):
    """Create and save a histogram of active TF-IDF features."""
    ensure_output_dirs()
    plot_path = os.path.join(PLOTS_DIR, "tfidf_feature_distribution.png")
    plt.figure(figsize=(8, 5))
    plt.hist(nnz_per_doc, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Number of active TF-IDF features per document")
    plt.ylabel("Frequency")
    plt.title("Distribution of TF-IDF feature activation per document")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    write_report(f"Saved plot: {plot_path}")
    return plot_path


def rare_class_recommendation(df, sentiment_col, rare_label="I can't tell", threshold_pct=2.0):
    """Write a recommendation for handling a rare target class."""
    percentages = df[sentiment_col].value_counts(normalize=True, dropna=False) * 100
    rare_pct = float(percentages.get(rare_label, 0.0))
    if rare_pct == 0.0:
        recommendation = f"'{rare_label}' is not present in the dataset."
    elif rare_pct < threshold_pct:
        recommendation = (
            f"'{rare_label}' accounts for only {rare_pct:.2f}% of rows, so it is extremely rare. "
            "Consider dropping it or merging it into a broader neutral or uncertain class before modeling."
        )
    else:
        recommendation = (
            f"'{rare_label}' accounts for {rare_pct:.2f}% of rows. "
            "Keep it only if your modeling setup can support a small minority class."
        )
    write_report(f"Rare class recommendation: {recommendation}")
    return recommendation


def label_cleanup_analysis(df, sentiment_col):
    """Write class counts, percentages, and imbalance notes for the target column."""
    counts = df[sentiment_col].value_counts(dropna=False)
    percentages = (counts / len(df) * 100).round(2)
    summary = pd.DataFrame({"count": counts, "percentage": percentages})

    max_class = summary["percentage"].idxmax()
    max_pct = summary.loc[max_class, "percentage"]
    min_class = summary["percentage"].idxmin()
    min_pct = summary.loc[min_class, "percentage"]

    imbalance_message = (
        f"Class imbalance note: '{max_class}' is the largest class at {max_pct:.2f}%, "
        f"while '{min_class}' is the smallest class at {min_pct:.2f}%."
    )
    if max_pct >= 50:
        imbalance_message += (
            " The dataset is meaningfully imbalanced, so class weights or resampling may help."
        )

    rare_message = ""
    if "I can't tell" in summary.index:
        rare_pct = summary.loc["I can't tell", "percentage"]
        rare_message = (
            f"'I can't tell' appears in only {rare_pct:.2f}% of rows, "
            "which makes it an extremely rare class."
        )

    write_report(
        "Target class distribution:\n"
        f"{summary.to_string()}\n\n"
        f"{imbalance_message}\n"
        f"{rare_message}".rstrip()
    )
    recommendation = rare_class_recommendation(df, sentiment_col)
    return summary, recommendation


def entity_column_analysis(df, entity_col):
    """Analyze the brand or entity column and save a top-entity plot."""
    ensure_output_dirs()
    missing_count = int(df[entity_col].isna().sum())
    unique_count = int(df[entity_col].dropna().nunique())
    top_entities = df[entity_col].dropna().value_counts().head(20)
    plot_series = df[entity_col].dropna().value_counts().head(15)
    plot_path = os.path.join(PLOTS_DIR, "top_entities.png")
    _plot_barh(plot_series, "Top entities", "Count", plot_path)

    summary_table = top_entities.rename_axis("entity").reset_index(name="count")
    write_report(
        f"Missing values in '{entity_col}': {missing_count}\n"
        f"Unique non-null entities: {unique_count}\n\n"
        "Top 20 entities:\n"
        f"{top_entities.to_string()}\n\n"
        f"Saved plot: {plot_path}"
    )
    return summary_table, plot_path


def sentiment_by_entity(df, entity_col, sentiment_col, top_n=10):
    """Create a crosstab and stacked bar chart for top entities by sentiment."""
    ensure_output_dirs()
    top_entities = df[entity_col].dropna().value_counts().head(top_n).index
    subset = df[df[entity_col].isin(top_entities)].copy()
    crosstab = pd.crosstab(subset[entity_col], subset[sentiment_col])
    crosstab = crosstab.loc[top_entities]

    plot_path = os.path.join(PLOTS_DIR, "sentiment_by_entity.png")
    crosstab.plot(kind="bar", stacked=True, figsize=(12, 6))
    plt.title("Sentiment distribution for top entities")
    plt.xlabel("Entity")
    plt.ylabel("Tweet count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    write_report(
        "Sentiment by top entities crosstab:\n"
        f"{crosstab.to_string()}\n\n"
        f"Saved plot: {plot_path}"
    )
    return crosstab, plot_path


def class_wise_text_length_analysis(df, text_col, sentiment_col):
    """Summarize text length by sentiment and save a boxplot."""
    ensure_output_dirs()
    if "text_length" not in df.columns or "word_count" not in df.columns:
        text_length_features(df, text_col)

    summary = df.groupby(sentiment_col)["text_length"].agg(["mean", "median", "std"])
    plot_path = os.path.join(PLOTS_DIR, "text_length_by_sentiment.png")
    plt.figure(figsize=(10, 6))
    order = df[sentiment_col].value_counts().index
    sns.boxplot(data=df, x=sentiment_col, y="text_length", order=order)
    plt.title("Text length by sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Text length (characters)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    write_report(
        "Class-wise text length summary:\n"
        f"{summary.to_string()}\n\n"
        f"Saved plot: {plot_path}"
    )
    return summary, plot_path


def word_frequency_by_sentiment(df, text_col, sentiment_col):
    """Write top words per main sentiment and save one plot for each class."""
    ensure_output_dirs()
    results = {}
    plot_paths = {}

    for sentiment in MAIN_SENTIMENTS:
        texts = df.loc[df[sentiment_col] == sentiment, text_col].fillna("")
        tokens = []
        for text in texts:
            tokens.extend(_tokenize_text(text))
        top_words = _tokens_to_series(tokens, top_n=15)
        results[sentiment] = top_words

        suffix = SENTIMENT_FILE_SUFFIXES[sentiment]
        plot_path = os.path.join(PLOTS_DIR, f"top_words_{suffix}.png")
        _plot_barh(top_words, f"Top words - {sentiment}", "Count", plot_path)
        plot_paths[sentiment] = plot_path

        report_text = (
            f"Top words for {sentiment}:\n"
            f"{top_words.to_string()}\n\n"
            f"Saved plot: {plot_path}"
        )
        write_report(report_text)

    return results, plot_paths


def extract_hashtags(text):
    """Extract lowercase hashtags from a single tweet."""
    text = "" if pd.isna(text) else str(text).lower()
    return re.findall(r"#\w+", text)


def hashtag_mention_link_analysis(df, text_col):
    """Write social marker usage stats and save a top-hashtag plot."""
    ensure_output_dirs()
    text_series = _string_series(df, text_col)

    has_hashtag = text_series.str.contains(r"#\w+", na=False)
    has_mention = text_series.str.contains(r"@\w+|@mention", na=False)
    has_link = text_series.str.contains(r"\{link\}|http", case=False, na=False)
    has_rt = text_series.str.contains(r"\brt\b", case=False, na=False)

    summary = pd.DataFrame(
        {
            "count": [
                int(has_hashtag.sum()),
                int(has_mention.sum()),
                int(has_link.sum()),
                int(has_rt.sum()),
            ],
            "proportion": [
                has_hashtag.mean(),
                has_mention.mean(),
                has_link.mean(),
                has_rt.mean(),
            ],
        },
        index=["contains_hashtag", "contains_mention", "contains_link", "contains_rt"],
    )

    hashtags = []
    for text in text_series:
        hashtags.extend(extract_hashtags(text))
    top_hashtags = _tokens_to_series(hashtags, top_n=20)

    plot_path = os.path.join(PLOTS_DIR, "top_hashtags.png")
    _plot_barh(top_hashtags.head(15), "Top hashtags", "Count", plot_path)

    write_report(
        "Hashtag, mention, link, and retweet usage:\n"
        f"{summary.to_string()}\n\n"
        "Top 20 hashtags:\n"
        f"{top_hashtags.to_string()}\n\n"
        f"Saved plot: {plot_path}"
    )
    return summary, top_hashtags, plot_path


def add_text_noise_features(df, text_col):
    """Add punctuation and surface-form noise features to df in place."""
    text_series = _string_series(df, text_col)
    df["exclamation_count"] = text_series.str.count("!")
    df["question_count"] = text_series.str.count(r"\?")
    df["uppercase_ratio"] = text_series.apply(_uppercase_ratio)
    df["digit_count"] = text_series.str.count(r"\d")
    return df


def add_stopword_ratio(df, text_col):
    """Add a stopword_ratio feature to df in place."""
    df["stopword_ratio"] = _string_series(df, text_col).apply(_stopword_ratio)
    return df


def punctuation_noise_analysis(df, text_col, sentiment_col):
    """Summarize punctuation and noise features overall and by sentiment."""
    add_text_noise_features(df, text_col)
    overall_summary = df[["exclamation_count", "question_count", "uppercase_ratio", "digit_count"]].agg(
        ["mean", "median", "std", "min", "max"]
    )
    by_sentiment = df.groupby(sentiment_col)[
        ["exclamation_count", "question_count", "uppercase_ratio", "digit_count"]
    ].mean()

    write_report(
        "Overall punctuation and text-noise feature summary:\n"
        f"{overall_summary.to_string()}\n\n"
        "Average punctuation and text-noise features by sentiment:\n"
        f"{by_sentiment.to_string()}"
    )
    return overall_summary, by_sentiment


def low_information_tweet_analysis(df, text_col):
    """Report empty and low-information tweets and write a modeling recommendation."""
    if "text_length" not in df.columns or "word_count" not in df.columns:
        text_length_features(df, text_col)

    text_series = _string_series(df, text_col)
    length_zero = int((df["text_length"] == 0).sum())
    word_count_le_two = int((df["word_count"] <= 2).sum())
    placeholder_only = int(text_series.apply(_only_placeholders_after_cleaning).sum())

    recommendation = (
        "Recommendation: drop rows with empty text, review tweets with two or fewer words, "
        "and consider removing tweets that reduce to only mentions, links, or hashtags after cleaning."
    )

    summary = pd.DataFrame(
        {"count": [length_zero, word_count_le_two, placeholder_only]},
        index=["length_zero", "word_count_le_two", "only_placeholders_after_cleaning"],
    )

    write_report(
        "Short, empty, and low-information tweet summary:\n"
        f"{summary.to_string()}\n\n"
        f"{recommendation}"
    )
    return summary, recommendation


def ngram_analysis(df, text_col, sentiment_col):
    """Write top bigrams overall and for positive and negative classes."""
    overall = _top_ngrams_from_texts(df[text_col].fillna(""), top_n=20)
    positive = _top_ngrams_from_texts(
        df.loc[df[sentiment_col] == "Positive emotion", text_col].fillna(""),
        top_n=15,
    )
    negative = _top_ngrams_from_texts(
        df.loc[df[sentiment_col] == "Negative emotion", text_col].fillna(""),
        top_n=15,
    )

    write_report(
        "Top 20 bigrams overall:\n"
        f"{overall.to_string()}\n\n"
        "Top 15 bigrams for Positive emotion:\n"
        f"{positive.to_string()}\n\n"
        "Top 15 bigrams for Negative emotion:\n"
        f"{negative.to_string()}"
    )
    return {"overall": overall, "positive": positive, "negative": negative}


def feature_correlation_analysis(df, text_col):
    """Create a numeric feature correlation matrix and save a heatmap."""
    ensure_output_dirs()
    if "text_length" not in df.columns or "word_count" not in df.columns:
        text_length_features(df, text_col)
    add_stopword_ratio(df, text_col)
    add_text_noise_features(df, text_col)

    feature_df = df[
        [
            "text_length",
            "word_count",
            "stopword_ratio",
            "uppercase_ratio",
            "exclamation_count",
            "question_count",
            "digit_count",
        ]
    ].copy()
    correlation = feature_df.corr().round(3)

    plot_path = os.path.join(PLOTS_DIR, "feature_correlation_heatmap.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation heatmap for engineered text features")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    write_report(
        "Feature correlation matrix:\n"
        f"{correlation.to_string()}\n\n"
        f"Saved plot: {plot_path}"
    )
    return correlation, plot_path


def outlier_inspection(df, text_col, sentiment_col, sample_size=5):
    """Write examples of very short and very long tweets based on quantiles."""
    if "text_length" not in df.columns:
        text_length_features(df, text_col)

    lower_bound = df["text_length"].quantile(0.01)
    upper_bound = df["text_length"].quantile(0.99)

    shortest = (
        df.loc[df["text_length"] <= lower_bound, [text_col, sentiment_col, "text_length"]]
        .sort_values("text_length", ascending=True)
        .head(sample_size)
    )
    longest = (
        df.loc[df["text_length"] >= upper_bound, [text_col, sentiment_col, "text_length"]]
        .sort_values("text_length", ascending=False)
        .head(sample_size)
    )

    write_report(
        f"Lower outlier threshold (1st percentile): {lower_bound}\n"
        f"Upper outlier threshold (99th percentile): {upper_bound}\n\n"
        "Sample shortest tweets:\n"
        f"{shortest.to_string(index=False)}\n\n"
        "Sample longest tweets:\n"
        f"{longest.to_string(index=False)}"
    )
    return shortest, longest


def preprocessing_recommendations(df, text_col, sentiment_col, entity_col):
    """Write a final recommendation section based on the observed EDA patterns."""
    recommendations = []

    if int(df[text_col].isna().sum()) > 0:
        recommendations.append("- Drop rows with null tweet_text before modeling.")

    if "I can't tell" in set(df[sentiment_col].dropna()):
        rare_pct = (df[sentiment_col] == "I can't tell").mean() * 100
        if rare_pct < 2:
            recommendations.append("- Consider removing or merging 'I can't tell' because it is extremely rare.")

    class_distribution = df[sentiment_col].value_counts(normalize=True)
    if not class_distribution.empty and class_distribution.max() >= 0.5:
        recommendations.append("- Handle class imbalance with class weights, stratified evaluation, or resampling.")

    duplicate_texts = int(df[text_col].duplicated().sum())
    if duplicate_texts > 0:
        recommendations.append("- Drop duplicate tweets if you want cleaner model evaluation and less repeated signal.")

    text_series = _string_series(df, text_col)
    if text_series.str.contains(r"@\w+|@mention|\{link\}|http", case=False, na=False).any():
        recommendations.append("- Remove or normalize placeholders such as @mention and {link} during preprocessing.")

    if text_series.str.contains(r"#\w+", na=False).any():
        recommendations.append("- Consider keeping hashtags as features or extracting them into separate engineered variables.")

    placeholder_only = int(text_series.apply(_only_placeholders_after_cleaning).sum())
    if placeholder_only > 0:
        recommendations.append("- Review tweets that collapse to only mentions, links, or hashtags after cleaning and consider dropping them.")

    if int(df[entity_col].isna().sum()) > 0:
        recommendations.append("- Treat the brand or entity column as optional metadata since many rows are missing that value.")

    if not recommendations:
        recommendations.append("- No urgent preprocessing issues were detected, but standard text normalization is still recommended.")

    write_section("Preprocessing Recommendations")
    write_report("\n".join(recommendations))
    return recommendations


def target_variable_analysis(df, sentiment_col, plot_filename="sentiment_distribution.png"):
    """Write class counts and percentages, assess balance, and save a label plot."""
    ensure_output_dirs()
    counts = df[sentiment_col].value_counts(dropna=False)
    percentages = (counts / len(df) * 100).round(2)
    summary = pd.DataFrame({"count": counts, "percentage": percentages})

    max_pct = float(percentages.max()) if not percentages.empty else 0.0
    balance_note = (
        "The dataset appears imbalanced."
        if max_pct >= 60
        else "The dataset is moderately imbalanced."
        if max_pct >= 50
        else "The dataset appears relatively balanced."
    )

    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, y=sentiment_col, order=counts.index)
    plt.title("Sentiment distribution")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    write_report(
        "Target class distribution:\n"
        f"{summary.to_string()}\n\n"
        f"Balance assessment: {balance_note}\n"
        f"Saved plot: {plot_path}"
    )
    return summary, balance_note, plot_path


def plot_text_length_distribution_named(df, text_col, plot_filename):
    """Save the text-length histogram using a custom filename."""
    ensure_output_dirs()
    if "text_length" not in df.columns:
        text_length_features(df, text_col)
    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    plt.figure(figsize=(8, 5))
    plt.hist(df["text_length"], bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Text length (characters)")
    plt.ylabel("Frequency")
    plt.title("Distribution of text length")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    write_report(f"Saved plot: {plot_path}")
    return plot_path


def class_wise_text_length_analysis_named(df, text_col, sentiment_col, plot_filename):
    """Summarize text length by class and save the boxplot with a custom filename."""
    ensure_output_dirs()
    if "text_length" not in df.columns or "word_count" not in df.columns:
        text_length_features(df, text_col)

    summary = df.groupby(sentiment_col)["text_length"].agg(["mean", "median", "std"])
    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    plt.figure(figsize=(10, 6))
    order = df[sentiment_col].value_counts().index
    sns.boxplot(data=df, x=sentiment_col, y="text_length", order=order)
    plt.title("Text length by sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Text length (characters)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    write_report(
        "Class-wise text length summary:\n"
        f"{summary.to_string()}\n\n"
        f"Saved plot: {plot_path}"
    )
    return summary, plot_path


def top_words_by_class(df, text_col, sentiment_col, top_n=20):
    """Write the top words overall and for each label class."""
    overall_tokens = clean_and_tokenize(df, text_col)
    overall_top_words = _tokens_to_series(overall_tokens, top_n=top_n)
    per_class = {}

    report_blocks = [
        "Top 20 words overall:",
        overall_top_words.to_string(),
    ]

    for label in df[sentiment_col].dropna().unique():
        tokens = []
        for text in df.loc[df[sentiment_col] == label, text_col].fillna(""):
            tokens.extend(_tokenize_text(text))
        top_words = _tokens_to_series(tokens, top_n=top_n)
        per_class[label] = top_words
        report_blocks.extend(
            [
                "",
                f"Top {top_n} words for class '{label}':",
                top_words.to_string(),
            ]
        )

    write_report("\n".join(report_blocks))
    return overall_top_words, per_class


def plot_meaningful_top_words(overall_top_words, plot_filename="top_words.png", top_n=15):
    """Save one overall top-words plot when the words are informative enough."""
    ensure_output_dirs()
    if overall_top_words.empty:
        write_report("Top-word plot skipped because no tokens were available.")
        return None, "No tokens available."

    top_slice = overall_top_words.head(top_n)
    dominant_stopwords = sum(word in STOP_WORDS for word in top_slice.index)
    if dominant_stopwords >= max(5, top_n // 2):
        note = "Top words are dominated by stopwords, so the overall top-words plot was skipped."
        write_report(note)
        return None, note

    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    _plot_barh(top_slice, "Top words", "Count", plot_path)
    write_report(f"Saved plot: {plot_path}")
    return plot_path, "Plot created."


def text_noise_quality_summary(df, text_col):
    """Write proportions of links, mentions, hashtags, digits, and low-information texts."""
    text_series = _string_series(df, text_col)
    if "word_count" not in df.columns:
        text_length_features(df, text_col)

    summary = pd.DataFrame(
        {
            "count": [
                int(text_series.str.contains(r"\{link\}|http", case=False, na=False).sum()),
                int(text_series.str.contains(r"@\w+|@mention", na=False).sum()),
                int(text_series.str.contains(r"#\w+", na=False).sum()),
                int(text_series.str.contains(r"\d", na=False).sum()),
                int((text_series.str.strip() == "").sum()),
                int((df["word_count"] <= 2).sum()),
            ],
            "proportion": [
                text_series.str.contains(r"\{link\}|http", case=False, na=False).mean(),
                text_series.str.contains(r"@\w+|@mention", na=False).mean(),
                text_series.str.contains(r"#\w+", na=False).mean(),
                text_series.str.contains(r"\d", na=False).mean(),
                (text_series.str.strip() == "").mean(),
                (df["word_count"] <= 2).mean(),
            ],
        },
        index=[
            "contains_link",
            "contains_mention",
            "contains_hashtag",
            "contains_digit",
            "empty_text",
            "very_short_text_le_2_words",
        ],
    )
    write_report(f"Text noise and quality summary:\n{summary.to_string()}")
    return summary


def plot_tfidf_feature_distribution_named(nnz_per_doc, plot_filename):
    """Save the TF-IDF feature histogram with a custom filename."""
    ensure_output_dirs()
    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    plt.figure(figsize=(8, 5))
    plt.hist(nnz_per_doc, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Number of active TF-IDF features per document")
    plt.ylabel("Frequency")
    plt.title("Distribution of TF-IDF feature activation per document")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    write_report(f"Saved plot: {plot_path}")
    return plot_path


def feature_correlation_analysis_named(df, text_col, plot_filename):
    """Create a correlation matrix using the binary-workflow feature set."""
    ensure_output_dirs()
    if "text_length" not in df.columns or "word_count" not in df.columns:
        text_length_features(df, text_col)
    add_text_noise_features(df, text_col)

    feature_df = df[
        [
            "text_length",
            "word_count",
            "uppercase_ratio",
            "exclamation_count",
            "question_count",
            "digit_count",
        ]
    ].copy()
    correlation = feature_df.corr().round(3)

    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    plt.figure(figsize=(9, 7))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation heatmap for engineered text features")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    write_report(
        "Feature correlation matrix:\n"
        f"{correlation.to_string()}\n\n"
        f"Saved plot: {plot_path}"
    )
    return correlation, plot_path


def binary_eda_summary(df, text_col, sentiment_col, nnz_per_doc):
    """Write a concise end-of-report summary for the binary sentiment dataset."""
    mean_length = df["text_length"].mean() if "text_length" in df.columns else np.nan
    label_distribution = df[sentiment_col].value_counts(normalize=True)
    max_pct = float(label_distribution.max() * 100) if not label_distribution.empty else 0.0
    balance_note = "balanced" if max_pct < 60 else "imbalanced"
    length_note = "mostly short" if mean_length < 140 else "mixed-length"
    cleanliness_note = "fairly clean" if df[text_col].isna().sum() <= 1 and df.duplicated().sum() < 50 else "needs cleaning"
    tfidf_note = (
        "TF-IDF looks appropriate because documents activate a manageable number of sparse features."
        if float(np.mean(nnz_per_doc)) < 100
        else "TF-IDF may still work, but feature density is higher than expected for short-form text."
    )

    summary = (
        f"Dataset size: {df.shape[0]} rows and {df.shape[1]} columns.\n"
        f"Label balance: the dataset appears {balance_note}.\n"
        f"Text length: tweets are {length_note} on average (mean length {mean_length:.2f} characters).\n"
        f"Data quality: the dataset looks {cleanliness_note} based on missingness and duplicates.\n"
        f"TF-IDF suitability: {tfidf_note}\n"
        "Recommendations: drop null or empty texts, deduplicate if needed, normalize links and mentions, and compare TF-IDF baselines with simple linear models before moving to heavier models."
    )
    write_report(summary)
    return summary
