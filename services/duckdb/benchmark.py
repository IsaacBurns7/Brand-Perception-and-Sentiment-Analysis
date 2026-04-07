import duckdb
import pandas as pd
import polars as pl
import pyarrow.csv as pv
import time
import os

# --- CONFIGURATION: Point these to your actual local files ---
TOPIC_TERMS_CSV = 'topics_info.csv' 
TOPIC_MATRIX_CSV = 'documents_info.csv'
# -------------------------------------------------------------

def print_results(title, results, csv_path):
    csv_size = os.path.getsize(csv_path) / 1e6
    print(f"\n{'='*20} {title} {'='*20}")
    print(f"File: {csv_path} ({csv_size:.2f} MB)")
    print(f"{'Method':<25} | {'Load/Plan (s)':<15} | {'Query (s)':<15}")
    print("-" * 60)
    for method, times in results.items():
        print(f"{method:<25} | {times[0]:.4f}          | {times[1]:.4f}")

def benchmark_topic_terms(csv_path):
    if not os.path.exists(csv_path):
        print(f"Skipping: {csv_path} not found.")
        return

    pq_path = csv_path.replace('.csv', '.parquet')
    con = duckdb.connect(':memory:')
    results = {}

    # 1. DuckDB Native
    s = time.time(); con.execute(f"CREATE OR REPLACE TABLE t1 AS SELECT * FROM '{csv_path}'"); w = time.time()-s
    s = time.time(); con.execute("SELECT COUNT(*) FROM t1 WHERE topic_terms LIKE '%energy%'").fetchone(); r = time.time()-s
    results['1. DuckDB Native'] = (w, r)

    # 2. Polars Lazy (Streaming)
    s = time.time(); lf = pl.scan_csv(csv_path); w = time.time()-s
    s = time.time(); lf.filter(pl.col("topic_terms").str.contains("energy")).select(pl.len()).collect(); r = time.time()-s
    results['2. Polars Lazy'] = (w, r)

    # 3. Pandas (PyArrow Engine)
    s = time.time(); df = pd.read_csv(csv_path, engine='pyarrow'); con.register("t_pd", df); w = time.time()-s
    s = time.time(); con.execute("SELECT COUNT(*) FROM t_pd WHERE topic_terms LIKE '%energy%'").fetchone(); r = time.time()-s
    results['3. Pandas (Arrow)'] = (w, r)

    # 4. Parquet (Zero-Copy Conversion & Read)
    s = time.time(); con.execute(f"COPY (SELECT * FROM '{csv_path}') TO '{pq_path}' (FORMAT PARQUET)"); w = time.time()-s
    s = time.time(); con.execute(f"SELECT COUNT(*) FROM '{pq_path}' WHERE topic_terms LIKE '%energy%'").fetchone(); r = time.time()-s
    results['4. Parquet (Conv+Read)'] = (w, r)

    print_results("TOPIC TERMS BENCHMARK", results, csv_path)

def benchmark_topic_matrix(csv_path):
    if not os.path.exists(csv_path):
        print(f"Skipping: {csv_path} not found.")
        return

    pq_path = csv_path.replace('.csv', '.parquet')
    con = duckdb.connect(':memory:')
    results = {}

    # 1. DuckDB Native
    s = time.time(); con.execute(f"CREATE OR REPLACE TABLE m1 AS SELECT * FROM '{csv_path}'"); w = time.time()-s
    s = time.time(); con.execute("SELECT AVG(topic_0), AVG(topic_1) FROM m1").fetchone(); r = time.time()-s
    results['1. DuckDB Native'] = (w, r)

    # 2. Polars Lazy
    s = time.time(); lf = pl.scan_csv(csv_path); w = time.time()-s
    s = time.time(); lf.select([pl.col("topic_0").mean(), pl.col("topic_1").mean()]).collect(); r = time.time()-s
    results['2. Polars Lazy'] = (w, r)

    # 3. Raw PyArrow
    s = time.time(); table = pv.read_csv(csv_path); con.register("m_arrow", table); w = time.time()-s
    s = time.time(); con.execute("SELECT AVG(topic_0), AVG(topic_1) FROM m_arrow").fetchone(); r = time.time()-s
    results['3. Raw PyArrow'] = (w, r)

    # 4. Parquet
    s = time.time(); con.execute(f"COPY (SELECT * FROM '{csv_path}') TO '{pq_path}' (FORMAT PARQUET)"); w = time.time()-s
    s = time.time(); con.execute(f"SELECT AVG(topic_0), AVG(topic_1) FROM '{pq_path}'").fetchone(); r = time.time()-s
    results['4. Parquet (Conv+Read)'] = (w, r)

    print_results("TOPIC MATRIX BENCHMARK", results, csv_path)

if __name__ == "__main__":
    benchmark_topic_terms(TOPIC_TERMS_CSV)
    benchmark_topic_matrix(TOPIC_MATRIX_CSV)