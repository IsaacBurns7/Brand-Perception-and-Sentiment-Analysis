from __future__ import annotations

import os
from pathlib import Path
import duckdb

import pandas as pd

from pipeline.document_builder import (
    build_processed_documents,
    build_stub_sentiment_output,
    build_stub_topic_output,
)
from pipeline.preprocessing import clean_documents

from main import (
    clean_reddit_data_to_curated
)


# Avoid matplotlib cache warnings during temporal_aggregation import.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

from temporal_aggregation import run_temporal_aggregation

def gather_data() -> pd.DataFrame:
    master_db_path = Path("./services/reddit_data/master.db")
    query = """
        WITH cleaned AS (
            SELECT text,
                string_split(trim(regexp_replace(text, '\\s+', ' ', 'g')), ' ') AS tokens
            FROM reddit_scrape_raw
        )
        SELECT text
        FROM cleaned
        WHERE array_length(tokens) > 10;
    """

    con = duckdb.connect(str(master_db_path))

    df = con.execute(query).df()

    con.close()
    return df

def build_sample_raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "url": "https://news.example.com/acme-earnings-2026-04-18",
                "title": "Acme posts stronger Q1 revenue as product demand improves",
                "description": "Executives said enterprise renewals remained stable.",
                "full_content": (
                    "Acme reported better-than-expected first-quarter revenue on April 18, 2026. "
                    "Management said demand improved across software subscriptions and support services, "
                    "while analysts noted steadier retention among large enterprise customers."
                ),
                "source_name": "Reuters",
                "published_at": "2026-04-18T13:15:00Z",
            },
            {
                "url": "https://news.example.com/acme-shipping-delays-2026-04-18",
                "title": "Customers flag Acme shipping delays after warehouse outage",
                "description": "Some orders were delayed for two days.",
                "full_content": (
                    "Customers on social platforms described delayed deliveries after an outage at a regional "
                    "warehouse. Acme said it restored operations and was working through a backlog of orders."
                ),
                "source_name": "CNBC",
                "published_at": "2026-04-18T20:40:00Z",
            },
            {
                "url": "https://news.example.com/acme-ai-partnership-2026-04-19",
                "title": "Acme announces AI partnership with major cloud provider",
                "description": "The deal focuses on analytics and customer support tools.",
                "full_content": (
                    "Acme announced a multiyear partnership to expand AI-based analytics and automate customer "
                    "support workflows. Executives said the agreement should reduce response times and improve "
                    "internal productivity."
                ),
                "source_name": "The Verge",
                "published_at": "2026-04-19T15:05:00Z",
            },
            {
                "url": "https://news.example.com/acme-pricing-2026-04-20",
                "title": "Analysts watch Acme pricing changes ahead of summer launch",
                "description": "Investors are looking for margin impact.",
                "full_content": (
                    "Analysts said Acme's upcoming pricing changes could support margins if demand remains solid. "
                    "Several firms kept a neutral stance while waiting for more guidance on the summer product launch."
                ),
                "source_name": "Bloomberg",
                "published_at": "2026-04-20T11:00:00Z",
            },
        ]
    )


def print_frame(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(df.to_string(index=False))


def main() -> None:
    #gather raw data 
    raw_rows = gather_data()
    print(raw_rows)
    #generate embed table
    
    #run RAG according to pre-defined user query  

    clean_docs = clean_documents(raw_rows)
    print(clean_docs)
    sentiment_output = build_stub_sentiment_output(clean_docs)
        #connect absa model and sentiment model 
    # topic_output = build_stub_topic_output(clean_docs)
        #connect bertopic rating - 08_confidence
        #add LLM inference of topics 
    # processed_docs = build_processed_documents(
    #     clean_docs,
    #     sentiment_output=sentiment_output,
    #     topic_output=topic_output,
    # )
    # daily_aggregation, changepoints = run_temporal_aggregation(processed_docs)

    # print_frame("Raw Sample Rows", raw_rows)
    # print_frame("Cleaned Documents", clean_docs)
    # print_frame("Processed Documents", processed_docs)
    # print_frame(
    #     "Daily Aggregation Summary",
    #     daily_aggregation[["day", "doc_count", "avg_sentiment", "sentiment_7d", "count_7d"]],
    # )

    # print("\nChangepoints")
    # if changepoints:
    #     for changepoint in changepoints:
    #         print(changepoint.isoformat())
    # else:
    #     print("No changepoints detected.")


if __name__ == "__main__":
    main()
