from apify_client import ApifyClient
import os 
import pandas as pd 
import duckdb
from datetime import datetime
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# now = datetime.now()
# timestamp = now.strftime("%Y-%m-%d_%H-%M")
# data_dir = Path(os.getenv("DATA_ROOT", "./data")) / "reddit-scraper-pro"
# csv_path = data_dir / f"{timestamp}.csv"
# csv_path = data_dir / f"2026-04-07_17-21.csv"
# database_file = os.getenv("DATABASE_FILE", "master.db")
# table_name = os.getenv("DATABASE_TABLE", "reddit_scrape_raw")
# max_items = int(os.getenv("MAX_ITEMS", "1000"))
# end_page = int(os.getenv("END_PAGE", "10"))

# def reddit_api_scraper():
#     # Initialize the ApifyClient with your API token
#     apify_api_token = os.getenv("APIFY_API_TOKEN")
#     if not apify_api_token:
#         raise ValueError("APIFY_API_TOKEN is required")

#     client = ApifyClient(apify_api_token)

#     # Prepare the Actor input
#     run_input = {
#         "startUrls": [
#             "https://www.reddit.com/r/pasta",
#             # "https://www.reddit.com/hot",
#             # "https://www.reddit.com/best",
#             # "https://www.reddit.com/user/nationalgeographic",
#             # "https://www.reddit.com/user/lukaskrivka/comments",
#             # "https://www.reddit.com/r/redditisfun/comments/13wxepd/rif_dev_here_reddits_api_changes_will_likely_kill",
#         ],
#         "search": "lasagna",
#         "searchMode": "link",
#         "sort": "relevance",
#         "time": "all",
#         "includeComments": False,
#         "maxItems": max_items,
#         "endPage": end_page,
#         "extendOutputFunction": "($) => { return {} }",
#         "customMapFunction": "(object) => { return {...object} }",
#         "proxy": {
#             "useApifyProxy": True,
#             "apifyProxyGroups": ["RESIDENTIAL"],
#         },
#     }

#     # Run the Actor and wait for it to finish
#     run = client.actor("jwR5FKaWaGSmkeq2b").call(run_input=run_input)

#     res = []
#     # Fetch and print Actor results from the run's dataset (if there are any)
#     for item in client.dataset(run["defaultDatasetId"]).iterate_items():
#         res.append(item)

#     df = pd.DataFrame(res)
#     data_dir.mkdir(parents=True, exist_ok=True)
#     df.to_csv(csv_path, index=False)   
#     return 0

def write_to_db(csv_path, database_file):
    # 1. Setup paths
    csv_path = Path(csv_path)
    database_file = Path(database_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {csv_path}")

    # 2. Connect to DuckDB
    # This automatically creates the .db file if it doesn't exist
    if database_file.parent.as_posix() not in ("", "."):
        database_file.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database_file.as_posix())

    try:
        src = csv_path.as_posix().replace("'", "''")
        src_count = con.execute(
            f"SELECT COUNT(*) FROM read_csv_auto('{src}')"
        ).fetchone()[0]

        # Define a stable staging schema used by this pipeline.
        con.execute("""
            CREATE TABLE IF NOT EXISTS reddit_staging (
                id VARCHAR PRIMARY KEY,
                createdAt TIMESTAMP,
                title TEXT,
                text TEXT,
                score INTEGER,
                subreddit VARCHAR,
                author VARCHAR,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        before_count = con.execute("SELECT COUNT(*) FROM reddit_staging").fetchone()[0]

        # Use src.<column> so selected names map to the CSV columns you expect.
        result = con.execute("""
            INSERT INTO reddit_staging (id, createdAt, title, text, score, subreddit, author)
            SELECT 
                src.id, 
                to_timestamp(src.createdAt::BIGINT), 
                src.title, 
                src.text, 
                src.score, 
                src.subreddit, 
                src.author
            FROM read_csv_auto(?) AS src
            ON CONFLICT (id) DO NOTHING
            RETURNING id
        """, [str(csv_path)]).fetchall()

        after_count = con.execute("SELECT COUNT(*) FROM reddit_staging").fetchone()[0]
        inserted_count = after_count - before_count

        if inserted_count != len(result):
            raise RuntimeError(
                "DuckDB insert verification failed "
                f"(inserted_delta={inserted_count}, returned_rows={len(result)})"
            )

        print(f"Successfully inserted: {inserted_count} rows into reddit_staging")
        print(f"Source rows seen in CSV: {src_count}")

        return {
            "status": "ok",
            "csv_path": str(csv_path),
            "database_file": database_file.as_posix(),
            "table_name": "reddit_staging",
            "csv_rows": int(src_count),
            "inserted_rows": int(inserted_count),
            "table_rows": int(after_count),
        }
    except Exception as exc:
        raise RuntimeError(
            f"write_to_db failed for {csv_path} into {database_file}: {exc}"
        ) from exc
    finally:
        con.close()

#reads table into csv_path 
def read_from_db(database_file, table, csv_path):
    # 1. Setup paths and basic validation
    database_file = Path(database_file)
    csv_path = Path(csv_path)
    if not database_file.exists():
        raise FileNotFoundError(f"Database not found: {database_file}")

    # 2. Ensure output directory exists
    if csv_path.parent.as_posix() not in ("", "."):
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Connect and export
    con = duckdb.connect(database_file)
    try:
        table_exists = con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            """,
            [table],
        ).fetchone()[0] > 0

        if not table_exists:
            raise ValueError(f"Table does not exist: {table}")

        src_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        dst = csv_path.as_posix().replace("'", "''")

        # Export table rows to CSV with a header row.
        con.execute(
            f"COPY (SELECT * FROM {table}) "
            f"TO '{dst}' (FORMAT CSV, HEADER, DELIMITER ',')"
        )

        if not csv_path.exists():
            raise RuntimeError(f"CSV output was not created: {csv_path}")

        dst_count = con.execute(
            f"SELECT COUNT(*) FROM read_csv_auto('{dst}')"
        ).fetchone()[0]

        if src_count != dst_count:
            raise RuntimeError(
                f"Row count mismatch after CSV export (table={src_count}, csv={dst_count})"
            )

        return {
            "status": "ok",
            "database_file": database_file.as_posix(),
            "table_name": table,
            "csv_path": str(csv_path),
            "table_rows": int(src_count),
            "csv_rows": int(dst_count),
        }
    except Exception as exc:
        raise RuntimeError(
            f"read_from_db failed for table {table} from {database_file}: {exc}"
        ) from exc
    finally:
        con.close()

if __name__ == "__main__":
    # exit_code = reddit_api_scraper()  # costs a small amount of $
    local_db = Path("./master.db")
    data_dir = Path(os.getenv("DATA_ROOT", "./data")) / "reddit-scraper-pro"
    csv_path = data_dir / f"2026-04-07_17-21.csv"

    print("\n[TEST 1] write_to_db happy path")
    try:
        write_result = write_to_db(csv_path, local_db)
        print(write_result)
    except Exception as exc:
        print(f"[TEST 1] failed: {exc}")

    print("\n[TEST 2] read_from_db happy path")
    try:
        read_result = read_from_db(local_db, "reddit_staging", csv_path)
        print(read_result)
    except Exception as exc:
        print(f"[TEST 2] failed: {exc}")

    print("\n[TEST 3] read_from_db missing table (expected failure)")
    try:
        read_from_db(local_db, "reddit_staging_missing", data_dir / "should_not_exist.csv")
        print("[TEST 3] unexpected success")
    except Exception as exc:
        print(f"[TEST 3] expected failure: {exc}")
