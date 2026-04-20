import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
from sqlalchemy import create_engine

from pipeline.storage import normalize_processed_documents


# CONFIG



BASE_DIR = Path(__file__).resolve().parent

# Database configuration 
# Example PostgreSQL: "postgresql://user:password@localhost:5432/mydatabase"
# Example SQLite: "sqlite:///local_data.db"
DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING", "sqlite:///default.db")
TARGET_TABLE = "sentiment_data" # Change this to your actual database table name

OUTPUT_CSV = BASE_DIR / "daily_results.csv"
OUTPUT_SENTIMENT_PLOT = BASE_DIR / "sentiment_over_time.png"
OUTPUT_CHANGEPOINT_PLOT = BASE_DIR / "changepoint_detection.png"

ROLLING_WINDOW_DAYS = 7
CHANGEPOINT_PENALTY = 3



# HELPERS


def load_data(engine) -> pd.DataFrame:
    """
    Load data directly from the database table. 
    Assumes the table strictly follows the defined schema.
    """
    print(f"Connecting to database and querying '{TARGET_TABLE}'...")
    
    # Query only the columns we need to save memory
    query = f"""
        SELECT 
            doc_id, text, brand, sentiment, 
            sentiment_label, topic, source, created_utc 
        FROM {TARGET_TABLE}
    """
    
    df = pd.read_sql(query, con=engine)

    if df.empty:
        raise ValueError(f"The table '{TARGET_TABLE}' is empty or could not be read.")

    # Parse actual timestamps to ensure Pandas treats them as datetime objects
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")

    missing_time_count = df["created_utc"].isna().sum()
    if missing_time_count > 0:
        print(f"Dropping {missing_time_count} rows with invalid or missing created_utc.")

    df = df.dropna(subset=["created_utc"]).copy()

    if df.empty:
        raise ValueError("No valid rows remain after parsing created_utc.")

    return df


def ensure_processed_documents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize the canonical processed-document shape.
    """
    normalized = normalize_processed_documents(df)
    if normalized.empty:
        raise ValueError("No valid rows remain after parsing created_utc.")

    return normalized


def build_daily_aggregation(
    processed_documents: pd.DataFrame,
    *,
    rolling_window_days: int = ROLLING_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Aggregate to daily buckets and reindex to continuous daily dates so
    rolling windows are based on calendar days not just observed rows.
    """
    df = ensure_processed_documents(processed_documents)
    df["day"] = df["created_utc"].dt.floor("D")

    daily = (
        df.groupby("day", as_index=False)
        .agg(
            doc_count=("doc_id", "count"),
            avg_sentiment=("sentiment", "mean"),
        )
        .sort_values("day")
    )

    # Reindex to all calendar days between min and max
    full_range = pd.date_range(
        start=daily["day"].min(),
        end=daily["day"].max(),
        freq="D",
        tz="UTC",
    )

    daily = (
        daily.set_index("day")
        .reindex(full_range)
        .rename_axis("day")
        .reset_index()
    )

    # Fill missing days for counts
    daily["doc_count"] = daily["doc_count"].fillna(0).astype(int)
    
    # For changepoint detection: Forward fill the previous day's sentiment to 
    # prevent artificial drops to 0.0 on days with zero documents.
    # The trailing fillna(0.0) catches the edge case where the very first day is missing.
    daily["avg_sentiment_filled"] = daily["avg_sentiment"].ffill().fillna(0.0)

    # Rolling averages (leaving base avg_sentiment as NaN for missing days so it doesn't skew the mean)
    daily["sentiment_7d"] = (
        daily["avg_sentiment"]
        .rolling(window=rolling_window_days, min_periods=1)
        .mean()
    )
    daily["count_7d"] = (
        daily["doc_count"]
        .rolling(window=rolling_window_days, min_periods=1)
        .mean()
    )

    return daily


def detect_changepoints(
    daily: pd.DataFrame,
    *,
    penalty: int | float = CHANGEPOINT_PENALTY,
) -> list[pd.Timestamp]:
    """
    Run changepoint detection on the daily sentiment signal.
    """
    signal = daily["avg_sentiment_filled"].to_numpy(dtype=float).reshape(-1, 1)

    if len(signal) < 3:
        return []

    model = rpt.Pelt(model="l2").fit(signal)
    breakpoints = model.predict(pen=penalty)

    cp_idx = [bp - 1 for bp in breakpoints[:-1] if 0 < bp <= len(daily)]
    cp_dates = daily.iloc[cp_idx]["day"].tolist()

    return cp_dates


def plot_sentiment(daily: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(daily["day"], daily["avg_sentiment"], label="Daily sentiment", marker='o', markersize=4, linestyle='-', alpha=0.5)
    plt.plot(daily["day"], daily["sentiment_7d"], label="7-day rolling average", linewidth=2)
    plt.title("Sentiment Over Time")
    plt.xlabel("Day")
    plt.ylabel("Average Sentiment")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_SENTIMENT_PLOT, dpi=300)
    plt.close()


def plot_changepoints(daily: pd.DataFrame, cp_dates: list[pd.Timestamp]) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(daily["day"], daily["avg_sentiment_filled"], label="Filled Daily Sentiment", color='gray', alpha=0.7)

    for cp in cp_dates:
        plt.axvline(cp, color='red', linestyle="--", alpha=0.8, label='Changepoint' if cp == cp_dates[0] else "")

    plt.title("Changepoint Detection (PELT)")
    plt.xlabel("Day")
    plt.ylabel("Average Sentiment")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_CHANGEPOINT_PLOT, dpi=300)
    plt.close()


def run_temporal_aggregation(
    processed_documents: pd.DataFrame,
    *,
    rolling_window_days: int = ROLLING_WINDOW_DAYS,
    changepoint_penalty: int | float = CHANGEPOINT_PENALTY,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    daily = build_daily_aggregation(
        processed_documents,
        rolling_window_days=rolling_window_days,
    )
    cp_dates = detect_changepoints(daily, penalty=changepoint_penalty)
    return daily, cp_dates



# MAIN


def main() -> None:
    # Initialize the database connection
    engine = create_engine(DB_CONNECTION_STRING)
    
    df = load_data(engine)

    print("\nLoaded data preview:")
    print(df.head())

    daily, cp_dates = run_temporal_aggregation(df)

    print("\nDaily aggregation preview:")
    print(daily.head())

    print("\nChangepoints:")
    if cp_dates:
        for cp in cp_dates:
            print(cp.strftime('%Y-%m-%d'))
    else:
        print("No changepoints detected.")

    plot_sentiment(daily)
    plot_changepoints(daily, cp_dates)

    # Save only the user-facing columns
    output_df = daily[
        ["day", "doc_count", "avg_sentiment", "sentiment_7d", "count_7d"]
    ].copy()
    output_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved daily results to: {OUTPUT_CSV}")
    print(f"Saved sentiment plot to: {OUTPUT_SENTIMENT_PLOT}")
    print(f"Saved changepoint plot to: {OUTPUT_CHANGEPOINT_PLOT}")


if __name__ == "__main__":
    main()
