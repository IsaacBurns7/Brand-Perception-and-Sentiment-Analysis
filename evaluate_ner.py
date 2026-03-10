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
                                                    # (requires manual labels)
    python evaluate_ner.py --backend rules          # single backend
    python evaluate_ner.py --verbose                # show per-sample detail
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Evaluate NER backends")
    p.add_argument("--backend", choices=["spacy", "rules", "both"], default="both")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    corpus = EVAL_CORPUS
    print(f"\nEvaluating on {len(corpus)} articles\n" + "=" * 72)

    results = {}

    if args.backend in ("both", "spacy"):
        with stage("spaCy + Rules (combined)"):
            pipeline = NERPipeline(combine_rules=True)
            results["spaCy+Rules"] = evaluate(pipeline, corpus, args.verbose)
        print(f"  spaCy+Rules  →  {_fmt(results['spaCy+Rules'])}")

    if args.backend in ("both", "rules"):
        with stage("Rules only"):
            # Force rules-only by temporarily marking spaCy unavailable
            pipeline = NERPipeline(combine_rules=False)
            pipeline._spacy._available = False
            pipeline.model_used        = pipeline._rules.name
            results["Rules-only"] = evaluate(pipeline, corpus, args.verbose)
        print(f"  Rules-only   →  {_fmt(results['Rules-only'])}")

    print("\n" + "=" * 72)
    best = max(results, key=lambda k: results[k]["f1"])
    print(f"  Best: {best}  (F1={results[best]['f1']:.3f})\n")


def _fmt(m: dict) -> str:
    return (f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
            f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  avg={m['avg_latency_ms']}ms")


if __name__ == "__main__":
    main()
