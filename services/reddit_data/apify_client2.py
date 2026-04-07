from apify_client import ApifyClient
from dotenv import load_dotenv
import os 
from pathlib import Path
import pandas as pd 
import duckdb
from datetime import datetime
import re

load_dotenv()

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M")
print("Timestamp: ", timestamp)
data_dir = Path(os.getenv("DATA_ROOT", "./data")) / "reddit-scraper-pro"
csv_path = data_dir / f"{timestamp}.csv"
database_file = os.getenv("DATABASE_FILE", "master.db")
table_name = os.getenv("DATABASE_TABLE", "reddit_scrape_raw")
max_items = int(os.getenv("MAX_ITEMS", "1000"))
end_page = int(os.getenv("END_PAGE", "10"))


def _validate_identifier(identifier: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier}")
    return identifier

def reddit_api_scraper():
    # Initialize the ApifyClient with your API token
    apify_api_token = os.getenv("APIFY_API_TOKEN")
    if not apify_api_token:
        raise ValueError("APIFY_API_TOKEN is required")

    client = ApifyClient(apify_api_token)

    # Prepare the Actor input
    run_input = {
        "startUrls": [
            "https://www.reddit.com/r/pasta",
            # "https://www.reddit.com/hot",
            # "https://www.reddit.com/best",
            # "https://www.reddit.com/user/nationalgeographic",
            # "https://www.reddit.com/user/lukaskrivka/comments",
            # "https://www.reddit.com/r/redditisfun/comments/13wxepd/rif_dev_here_reddits_api_changes_will_likely_kill",
        ],
        "search": "lasagna",
        "searchMode": "link",
        "sort": "relevance",
        "time": "all",
        "includeComments": False,
        "maxItems": max_items,
        "endPage": end_page,
        "extendOutputFunction": "($) => { return {} }",
        "customMapFunction": "(object) => { return {...object} }",
        "proxy": {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"],
        },
    }

    # Run the Actor and wait for it to finish
    run = client.actor("jwR5FKaWaGSmkeq2b").call(run_input=run_input)

    res = []
    # Fetch and print Actor results from the run's dataset (if there are any)
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        res.append(item)

    df = pd.DataFrame(res)
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)   
    return 0

def write_to_db(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {csv_path}")
    if not database_file:
        raise ValueError("DATABASE_FILE is required")

    db_path = Path(database_file)
    if db_path.parent.as_posix() not in ("", "."):
        db_path.parent.mkdir(parents=True, exist_ok=True)

    safe_table = _validate_identifier(table_name)
    pq_path = csv_path.with_suffix(".parquet")
    con = duckdb.connect(db_path.as_posix())
    try:
        src = csv_path.as_posix().replace("'", "''")
        dst = pq_path.as_posix().replace("'", "''")

        src_count = con.execute(
            f"SELECT COUNT(*) FROM read_csv_auto('{src}')"
        ).fetchone()[0]

        table_exists = con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            """,
            [safe_table],
        ).fetchone()[0] > 0

        if not table_exists:
            con.execute(
                f"CREATE TABLE {safe_table} AS SELECT * FROM read_csv_auto('{src}') LIMIT 0"
            )

        before_count = con.execute(f"SELECT COUNT(*) FROM {safe_table}").fetchone()[0]

        # BY NAME guards against column-order mismatches between CSV and table.
        con.execute(
            f"INSERT INTO {safe_table} BY NAME SELECT * FROM read_csv_auto('{src}')"
        )

        after_count = con.execute(f"SELECT COUNT(*) FROM {safe_table}").fetchone()[0]
        inserted_count = after_count - before_count

        if inserted_count != src_count:
            raise RuntimeError(
                "DuckDB insert verification failed "
                f"(expected inserted={src_count}, actual inserted={inserted_count})"
            )

        con.execute(
            f"COPY (SELECT * FROM read_csv_auto('{src}')) "
            f"TO '{dst}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )

        if not pq_path.exists():
            raise RuntimeError(f"Parquet output was not created: {pq_path}")

        dst_count = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{dst}')"
        ).fetchone()[0]

        if src_count != dst_count:
            raise RuntimeError(
                f"Row count mismatch after parquet write (csv={src_count}, parquet={dst_count})"
            )

        return {
            "status": "ok",
            "csv_path": str(csv_path),
            "database_file": db_path.as_posix(),
            "table_name": safe_table,
            "parquet_path": str(pq_path),
            "csv_rows": int(src_count),
            "inserted_rows": int(inserted_count),
            "table_rows": int(after_count),
            "parquet_rows": int(dst_count),
        }
    except Exception as exc:
        raise RuntimeError(f"write_to_db failed for {csv_path}: {exc}") from exc
    finally:
        con.close()

if __name__ == "__main__":
    exit_code = reddit_api_scraper()
    result = write_to_db(csv_path)
    print(result)