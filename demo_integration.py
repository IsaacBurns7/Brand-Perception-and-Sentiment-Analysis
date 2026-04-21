from __future__ import annotations

import os
from pathlib import Path
import duckdb
import joblib

import pandas as pd

from pipeline.document_builder import (
    build_processed_documents,
    build_stub_sentiment_output,
    build_stub_topic_output,
    build_topic_output
)
from pipeline.preprocessing import clean_documents

from main import (
    clean_reddit_data_to_curated
)
from models.sentiment.predict import (
    run_hf_batch_inference
)


# Avoid matplotlib cache warnings during temporal_aggregation import.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

from temporal_aggregation import run_temporal_aggregation

def gather_data() -> pd.DataFrame:
    master_db_path = Path("./services/reddit_data/master.db")
    query = """
        SELECT
            text,
            CASE
                WHEN createdAt IS NULL THEN NULL
                WHEN CAST(createdAt AS BIGINT) >= 1000000000000
                    THEN to_timestamp(CAST(createdAt AS DOUBLE) / 1000.0)
                ELSE to_timestamp(CAST(createdAt AS DOUBLE))
            END AS created_utc
        FROM reddit_scrape_raw
        WHERE text IS NOT NULL
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
    # print(raw_rows)
    #generate embed table
    #run RAG according to pre-defined user query  

    clean_docs = clean_documents(raw_rows)
    # print(clean_docs)
    #write clean docs to some csv 

    sentiment_output = build_stub_sentiment_output(clean_docs)
    args = {
        "model_path": "deepakm10/brand-perception-models",
        "input_path": Path("services/reddit_data/data/reddit-scraper-pro/2026-04-13_18-15.csv"),
        "output_path": Path("inference.csv"),
        "text_column": "text",
        "label_column": "sentiment",
        "batch_size": 50,
    }
    info = run_hf_batch_inference(
        model_path=args["model_path"],
        input_path=args["input_path"],
        output_path=args["output_path"],
        text_column=args["text_column"],
        label_column=args["label_column"],
        batch_size=args["batch_size"],
        include_true_labels=not args.get("omit_labels", True),
    )
    # print(info)
        # load output file into DF     
    sentiment_output = pd.read_csv(args["output_path"])
    # print(sentiment_output)
        #write sentiment output to db (ignore for now)

    # topic_output = build_stub_topic_output(clean_docs)
    topic_output = build_topic_output(clean_docs)
    # print(topic_output)

    # clustered = clean_docs.merge(topic_output, on="doc_id", how="left")

    # clusters = (
    #     clustered.groupby("topic", dropna=False)
    #     .agg(
    #         doc_count=("doc_id", "count"),
    #         doc_ids=("doc_id", list),
    #         sample_text=("text", lambda s: s.iloc[0] if len(s) else "")
    #     )
    #     .reset_index()
    #     .sort_values("doc_count", ascending=False)
    # )

    # print(clusters)

    processed_docs = build_processed_documents(
        clean_docs,
        sentiment_output=sentiment_output,
        topic_output=topic_output,
        absa_enabled=True
    )
    processed_docs.to_csv(Path("processed_documents.csv"), index=False)
    #current at 
    #doc_id,text,brand,aspect,sentiment,sentiment_label,topic,source,created_utc
    #6,a family of two maybe,unknown,general,0.0,neutral,82,unknown,2025-07-23 22:56:16+00:00
    #fix brand(probably NER coercion), aspect(ABSA model), sentiment(use ABSA model), sentiment_label(use ABSA model), source="reddit"

    # build_daily_aggregation expects processed-document schema columns:
    # doc_id, text, brand, aspect, sentiment, sentiment_label, topic, source, created_utc.
    # Aggregation itself uses created_utc (time bucket), doc_id (count), and sentiment (mean).
    daily_aggregation, changepoints = run_temporal_aggregation(processed_docs)
    aggregation_output_path = Path("temporal_aggregation_daily.csv")
    aggregation_output_path.parent.mkdir(parents=True, exist_ok=True)
    daily_aggregation.to_csv(aggregation_output_path, index=False)
    print(f"Saved temporal aggregation CSV: {aggregation_output_path}")

    # print_frame("Raw Sample Rows", raw_rows)
    # print_frame("Cleaned Documents", clean_docs)
    # print_frame("Processed Documents", processed_docs)
    # print_frame(
    #     "Daily Aggregation Summary",
    #     daily_aggregation[["day", "doc_count", "avg_sentiment", "sentiment_7d", "count_7d"]],
    # )

    print("\nChangepoints")
    if changepoints:
        for changepoint in changepoints:
            print(changepoint.isoformat())
    else:
        print("No changepoints detected.")


if __name__ == "__main__":
    main()
