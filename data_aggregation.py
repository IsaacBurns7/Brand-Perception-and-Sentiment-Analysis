import os
import logging
import pandas as pd
from sqlalchemy import create_engine, text

# --- CONFIGURATION ---
DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING", "postgresql://user:password@localhost:5432/mydatabase")
TARGET_TABLE = "aggregated_documents"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def build_aggregated_table(engine) -> pd.DataFrame:
    """
    Joins the core articles table with the outputs from NER, Sentiment, 
    and Topic Modeling to create a flattened document-level view.
    """
    logger.info("Executing aggregation join query...")
    
    # This query assumes the downstream models (sentiment, topics) output to 
    # their respective tables linking back to the article_url (doc_id).
    # We do an INNER JOIN on brands (since we only care about documents with brands)
    # and LEFT JOINS on sentiment/topics in case some documents failed those steps.
    query = """
        SELECT 
            a.url AS doc_id, 
            a.full_content AS text, 
            b.canonical_name AS brand, 
            'general' AS aspect,
            s.sentiment_score AS sentiment, 
            s.sentiment_label AS sentiment_label, 
            t.topic_name AS topic, 
            a.source_name AS source, 
            a.published_at AS created_utc
        FROM articles a
        
        -- 1. Link NER Brands
        INNER JOIN document_brand_mentions dbm ON a.url = dbm.article_url
        INNER JOIN brands b ON dbm.brand_id = b.id
        
        -- 2. Link Sentiment Analysis (Assuming a document_sentiment table)
        LEFT JOIN document_sentiment s ON a.url = s.article_url
        
        -- 3. Link Topic Modeling (Assuming a document_topics table)
        LEFT JOIN document_topics t ON a.url = t.article_url
    """
    
    df = pd.read_sql(query, con=engine)
    logger.info(f"Aggregated {len(df)} raw rows from the database.")
    
    return df

def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures data types match the temporal_aggregation.py expectations 
    and drops rows missing critical time-series components.
    """
    logger.info("Cleaning and normalizing aggregated data...")
    
    # Ensure created_utc is a valid datetime object
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")
    
    # Drop rows where we don't have a valid time or doc_id to aggregate on
    missing_time = df["created_utc"].isna().sum()
    if missing_time > 0:
        logger.warning(f"Dropping {missing_time} rows with invalid or missing 'created_utc'.")
        
    df = df.dropna(subset=["created_utc", "doc_id", "brand"]).copy()
    
    # Fill missing sentiment/topic fields with defaults if downstream models failed
    df["aspect"] = df["aspect"].fillna("general")
    df["sentiment"] = df["sentiment"].fillna(0.0).astype(float)
    df["sentiment_label"] = df["sentiment_label"].fillna("neutral")
    df["topic"] = df["topic"].fillna("unclassified")
    
    return df

def write_to_warehouse(df: pd.DataFrame, engine):
    """
    Writes the final flattened dataframe to the target analytics table.
    """
    logger.info(f"Writing {len(df)} rows to target table: '{TARGET_TABLE}'...")
    
    # Write to DB. Using if_exists='replace' to overwrite the table completely 
    # on each full aggregation run, or 'append' if you are doing daily batches.
    df.to_sql(TARGET_TABLE, con=engine, if_exists="replace", index=False)
    
    # Create an index on the new table to speed up the temporal aggregation queries
    with engine.connect() as conn:
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{TARGET_TABLE}_time ON {TARGET_TABLE} (created_utc);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{TARGET_TABLE}_brand ON {TARGET_TABLE} (brand);"))
        conn.commit()
        
    logger.info("Aggregation complete and indexes created.")

def main():
    engine = create_engine(DB_CONNECTION_STRING)
    
    try:
        raw_df = build_aggregated_table(engine)
        if raw_df.empty:
            logger.error("The aggregation query returned 0 rows. Check your source tables.")
            return
            
        clean_df = clean_and_normalize(raw_df)
        write_to_warehouse(clean_df, engine)
        
    except Exception as e:
        logger.error(f"Data aggregation failed: {e}")

if __name__ == "__main__":
    main()
