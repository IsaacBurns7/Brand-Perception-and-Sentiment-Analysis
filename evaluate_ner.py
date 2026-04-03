"""
NER Evaluation
==============
Benchmarks model backends on a sample corpus drawn from the same
news / article domain as rating.csv and news_dailyworker output.

The evaluation style mirrors LDA.py: a stage() context manager for
timings and structured printed output, consistent with the project's
existing dev workflow.

Usage
-----
    python evaluate_ner.py                          # built-in corpus
    python evaluate_ner.py --csv data/rating.csv    # score against real data
    python evaluate_ner.py --backend rules          # single backend
    python evaluate_ner.py --verbose                # show per-sample detail

CSV mode
--------
  If the CSV contains a column named `ner_gold` (comma-separated brand names)
  precision / recall / F1 are computed against those labels.

  If no `ner_gold` column is present the script runs the pipeline and
  prints coverage stats: articles processed, total brands found, and the
  top-20 brands by mention count.

  Text is read from the first available column in priority order:
    article → full_content → content → title+description
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from contextlib import contextmanager

from ner_pipeline import NERPipeline


# ── Timing helper (matches LDA.py) ───────────────────────────────────────────

@contextmanager
def stage(name: str):
    start = time.perf_counter()
    print(f"[timing] start: {name}", flush=True)
    try:
        yield
    finally:
        print(f"[timing] end:   {name} ({time.perf_counter()-start:.2f}s)", flush=True)


# ── Evaluation corpus ─────────────────────────────────────────────────────────
# Articles are representative of the news_dailyworker domain:
# finance/tech/consumer brands across the sources already scraped
# (Forbes, BBC, Wired, Verge, CNN, Bloomberg, etc.)

EVAL_CORPUS = [
    {
        "text": (
            "Apple Inc. reported a record quarter driven by iPhone 15 sales. "
            "CEO Tim Cook said demand in Asia remains strong. Analysts at Goldman Sachs "
            "raised their price target for Apple. Microsoft Corp also beat earnings "
            "estimates, helped by Azure cloud growth. Alphabet Inc., Google's parent, "
            "posted mixed results due to YouTube ad slowdown."
        ),
        "gold": ["Apple Inc.", "Microsoft Corp", "Alphabet Inc.", "Google", "Goldman Sachs"],
        "source": "Forbes",
        "category": "technology",
    },
    {
        "text": (
            "Tesla's Cybertruck finally reached customers after years of delays. "
            "Rival EV makers Rivian Automotive Inc. and Lucid Motors have struggled. "
            "Ford Motor Company expanded its EV lineup with the F-150 Lightning. "
            "General Motors said it would restart Chevy Bolt production. "
            "Volkswagen AG and BMW AG are also ramping European EV output."
        ),
        "gold": ["Tesla", "Rivian Automotive Inc.", "Lucid Motors", "Ford Motor Company",
                 "General Motors", "Volkswagen AG", "BMW AG"],
        "source": "The Verge",
        "category": "automotive",
    },
    {
        "text": (
            "OpenAI launched GPT-4 Turbo at its first developer conference. "
            "Anthropic, the rival AI safety startup backed by Google and Amazon, "
            "released Claude 2.1 the same week. Hugging Face published a new "
            "open-weight model. Microsoft, which has invested $13 billion in OpenAI, "
            "said Copilot would be integrated across its Office 365 suite."
        ),
        "gold": ["OpenAI", "Anthropic", "Google", "Amazon", "Hugging Face", "Microsoft"],
        "source": "Wired",
        "category": "artificial intelligence",
    },
    {
        "text": (
            "Nike Inc. reported a slowdown in North America as consumers trade down. "
            "Adidas AG posted its first annual loss in decades after ending the Yeezy "
            "partnership. Lululemon Athletica Inc. bucked the trend with double-digit "
            "growth. Under Armour, Inc. cut its full-year outlook. "
            "Walmart and Target both reported inventory improvements heading into Q4."
        ),
        "gold": ["Nike Inc.", "Adidas AG", "Lululemon Athletica Inc.",
                 "Under Armour, Inc.", "Walmart", "Target"],
        "source": "BBC News",
        "category": "retail",
    },
    {
        "text": (
            "Pfizer Inc. said its COVID vaccine revenue would decline sharply in 2024. "
            "Moderna posted a wider-than-expected loss. Johnson & Johnson spun off "
            "its consumer health unit as Kenvue Inc. AstraZeneca PLC received FDA "
            "approval for a new lung cancer drug. Merck & Co. defended its Keytruda "
            "patent against generic challengers."
        ),
        "gold": ["Pfizer Inc.", "Moderna", "Johnson & Johnson", "Kenvue Inc.",
                 "AstraZeneca PLC", "Merck & Co."],
        "source": "Reuters",
        "category": "pharmaceuticals",
    },
]


# ── Metrics ───────────────────────────────────────────────────────────────────

def _canonical(name: str) -> str:
    return re.sub(r'[^a-z0-9 ]', '', name.lower()).strip()


def evaluate(pipeline: NERPipeline, corpus: list[dict], verbose: bool = False) -> dict:
    tp = fp = fn = 0
    latencies = []

    for i, sample in enumerate(corpus):
        gold = {_canonical(g) for g in sample["gold"]}

        t0     = time.perf_counter()
        result = pipeline.run(
            sample["text"],
            doc_id      = f"eval_{i}",
            source_name = sample.get("source"),
            category    = sample.get("category"),
        )
        latencies.append(time.perf_counter() - t0)

        pred = {_canonical(e.canonical_name) for e in result.brand_entities}

        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

        if verbose:
            print(f"\n  Sample {i+1} [{sample.get('source')} / {sample.get('category')}]")
            print(f"    Gold:      {sorted(gold)}")
            print(f"    Predicted: {sorted(pred)}")
            if gold - pred: print(f"    Missed:    {sorted(gold - pred)}")
            if pred - gold: print(f"    Extra:     {sorted(pred - gold)}")

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "precision":      round(prec, 4),
        "recall":         round(rec,  4),
        "f1":             round(f1,   4),
        "tp": tp, "fp": fp, "fn": fn,
        "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 1),
    }


# ── CSV evaluation ────────────────────────────────────────────────────────────

def _pick_text_column(df) -> str:
    """Return the first available text column in richness priority order."""
    for col in ["article", "full_content", "content", "text"]:
        if col in df.columns:
            return col
    # Minimal fallback: title
    if "title" in df.columns:
        return "title"
    return df.columns[0]


def evaluate_csv(pipeline: NERPipeline, csv_path: str, verbose: bool = False) -> None:
    """
    Run the pipeline over a CSV file and report results.

    If the CSV has a ``ner_gold`` column (comma-separated brand names) precision,
    recall, and F1 are computed. Otherwise coverage stats are printed.
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas is required for CSV evaluation: pip install pandas")
        return

    df = pd.read_csv(csv_path)
    print(f"\nLoaded CSV: {csv_path}  ({len(df)} rows, columns: {list(df.columns)})")

    text_col = _pick_text_column(df)
    print(f"Using text column: '{text_col}'")

    has_gold = "ner_gold" in df.columns

    # Build corpus list so we can reuse the existing evaluate() logic when
    # gold labels are present.
    corpus = []
    for _, row in df.iterrows():
        text = str(row.get(text_col, "") or "")
        if len(text) < 200:
            title = str(row.get("title", "") or "")
            desc  = str(row.get("description", "") or "")
            text  = " ".join(filter(None, [title, desc, text]))

        entry = {
            "text":     text,
            "source":   str(row.get("source_name", row.get("source", ""))),
            "category": str(row.get("category", "")),
        }
        if has_gold:
            raw_gold = str(row.get("ner_gold", "") or "")
            entry["gold"] = [g.strip() for g in raw_gold.split(",") if g.strip()]
        corpus.append(entry)

    if has_gold:
        print("\nGold labels found in 'ner_gold' column — computing P/R/F1\n" + "=" * 72)
        metrics = evaluate(pipeline, corpus, verbose=verbose)
        print(f"\n  Result  ->  {_fmt(metrics)}\n")
    else:
        print("\nNo 'ner_gold' column — running pipeline and reporting coverage stats\n" + "=" * 72)
        brand_counts: dict[str, int] = defaultdict(int)
        docs_with_brands = 0
        total_mentions   = 0

        with stage(f"NER over {len(corpus)} rows"):
            for i, sample in enumerate(corpus):
                result = pipeline.run(
                    sample["text"],
                    doc_id      = f"csv_{i}",
                    source_name = sample.get("source"),
                    category    = sample.get("category"),
                )
                if result.brand_entities:
                    docs_with_brands += 1
                for ent in result.brand_entities:
                    brand_counts[ent.canonical_name] += ent.mention_count
                    total_mentions += ent.mention_count

                if verbose and result.brand_entities:
                    print(f"  Row {i+1}: {result.brand_names()}")

        coverage = docs_with_brands / len(corpus) * 100 if corpus else 0
        print(f"\n  Articles processed : {len(corpus)}")
        print(f"  Articles with brands: {docs_with_brands}  ({coverage:.1f}%)")
        print(f"  Total brand mentions : {total_mentions}")
        print(f"  Unique brands found  : {len(brand_counts)}")
        print("\n  Top 20 brands by mention count:")
        for brand, count in sorted(brand_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"    {count:>5}  {brand}")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Evaluate NER backends")
    p.add_argument("--csv",     metavar="PATH",
                   help="Path to CSV file (e.g. data/rating.csv). "
                        "Include a 'ner_gold' column (comma-separated brands) for F1 scoring; "
                        "omit it to get coverage stats instead.")
    p.add_argument("--backend", choices=["spacy", "rules", "both"], default="both")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # ── CSV path ──────────────────────────────────────────────────────────────
    if args.csv:
        if args.backend in ("both", "spacy"):
            with stage("spaCy + Rules (combined)"):
                pipeline = NERPipeline(combine_rules=True)
            evaluate_csv(pipeline, args.csv, verbose=args.verbose)
        else:
            with stage("Rules only"):
                pipeline = NERPipeline(combine_rules=False)
                pipeline._spacy._available = False
                pipeline.model_used        = pipeline._rules.name
            evaluate_csv(pipeline, args.csv, verbose=args.verbose)
        return

    # ── Built-in corpus ───────────────────────────────────────────────────────
    corpus = EVAL_CORPUS
    print(f"\nEvaluating on {len(corpus)} articles\n" + "=" * 72)

    results = {}

    if args.backend in ("both", "spacy"):
        with stage("spaCy + Rules (combined)"):
            pipeline = NERPipeline(combine_rules=True)
            results["spaCy+Rules"] = evaluate(pipeline, corpus, args.verbose)
        print(f"  spaCy+Rules  ->  {_fmt(results['spaCy+Rules'])}")

    if args.backend in ("both", "rules"):
        with stage("Rules only"):
            # Force rules-only by temporarily marking spaCy unavailable
            pipeline = NERPipeline(combine_rules=False)
            pipeline._spacy._available = False
            pipeline.model_used        = pipeline._rules.name
            results["Rules-only"] = evaluate(pipeline, corpus, args.verbose)
        print(f"  Rules-only   ->  {_fmt(results['Rules-only'])}")

    print("\n" + "=" * 72)
    best = max(results, key=lambda k: results[k]["f1"])
    print(f"  Best: {best}  (F1={results[best]['f1']:.3f})\n")


def _fmt(m: dict) -> str:
    return (f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
            f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  avg={m['avg_latency_ms']}ms")


if __name__ == "__main__":
    main()
