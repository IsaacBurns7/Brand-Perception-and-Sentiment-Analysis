"""Generate all model-performance visualisations from model_performance_report.json."""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
REPORT_PATH = Path("artifacts/reports/model_performance_report.json")
OUTPUT_DIR  = Path("visuals")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Global style ───────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
DPI = 300

# ── Palette ────────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "distilbert_stage2": "#4C8BF5",
    "roberta_stage2":    "#1DB87A",
    "bertweet_stage2":   "#F5A623",
    "sklearn_baseline":  "#A0A0A0",
}
CLASS_COLORS = {
    "negative": "#E05252",
    "neutral":  "#888888",
    "positive": "#4CAF50",
}
MODEL_LABELS = {
    "distilbert_stage2": "DistilBERT S2",
    "roberta_stage2":    "RoBERTa S2",
    "bertweet_stage2":   "BERTweet S2",
    "sklearn_baseline":  "Sklearn Baseline",
}
MODEL_ORDER = ["distilbert_stage2", "roberta_stage2", "bertweet_stage2"]

SA_KEY  = "sentiment_analysis_test"
MAN_KEY = "testdata_manual_2009"

# ── Load data ──────────────────────────────────────────────────────────────────
with open(REPORT_PATH) as f:
    report = json.load(f)

models = report["models"]


def savefig(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}  ({path.stat().st_size:,} bytes)")


def annotate_bars(ax, fmt="{:.1f}", offset=0.5):
    """Add value labels on top of every bar."""
    for patch in ax.patches:
        h = patch.get_height()
        if np.isnan(h):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center", va="bottom", fontsize=7, fontweight="bold",
        )


# ══════════════════════════════════════════════════════════════════════════════
# 01 & 02 — Grouped bar charts: Macro F1 / Accuracy × both test sets
# ══════════════════════════════════════════════════════════════════════════════

def grouped_model_bar(metric_key: str, scale: float, title: str, ylabel: str,
                      ylim: tuple, fname: str) -> None:
    sa_vals  = [models[m]["evaluation"][SA_KEY][metric_key]  * scale for m in MODEL_ORDER]
    man_vals = [models[m]["evaluation"][MAN_KEY][metric_key] * scale for m in MODEL_ORDER]

    x = np.arange(len(MODEL_ORDER))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - w / 2, sa_vals,  w, label="SA Test (n=3534)",
                   color=[MODEL_COLORS[m] for m in MODEL_ORDER], edgecolor="white")
    bars2 = ax.bar(x + w / 2, man_vals, w, label="Manual 2009 (n=516)",
                   color=[MODEL_COLORS[m] for m in MODEL_ORDER], edgecolor="white",
                   alpha=0.6, hatch="//")

    annotate_bars(ax, offset=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend()
    fig.tight_layout()
    savefig(fig, fname)


grouped_model_bar(
    "macro_f1", 100,
    "Macro F1 — All Models × Both Test Sets", "Macro F1 score",
    (40, 87), "01_macro_f1_comparison.png",
)
grouped_model_bar(
    "accuracy", 100,
    "Accuracy — All Models × Both Test Sets", "Accuracy (%)",
    (40, 87), "02_accuracy_comparison.png",
)


# ══════════════════════════════════════════════════════════════════════════════
# 03 & 04 — Per-class F1: DistilBERT vs RoBERTa
# ══════════════════════════════════════════════════════════════════════════════

def per_class_f1_chart(eval_key: str, n: int, title: str, fname: str) -> None:
    classes   = ["negative", "neutral", "positive"]
    db_vals   = [models["distilbert_stage2"]["evaluation"][eval_key]["per_class_f1"][c] * 100
                 for c in classes]
    rob_vals  = [models["roberta_stage2"]["evaluation"][eval_key]["per_class_f1"][c]    * 100
                 for c in classes]

    x = np.arange(len(classes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(x - w / 2, db_vals,  w, label="DistilBERT S2",
           color=MODEL_COLORS["distilbert_stage2"], edgecolor="white")
    ax.bar(x + w / 2, rob_vals, w, label="RoBERTa S2",
           color=MODEL_COLORS["roberta_stage2"],    edgecolor="white")

    annotate_bars(ax, offset=0.2)

    ax.set_xticks(x)
    ax.set_xticklabels(["Negative", "Neutral", "Positive"])
    ax.set_ylabel("F1 score")
    ax.set_ylim(70, 90)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.legend()
    fig.tight_layout()
    savefig(fig, fname)


per_class_f1_chart(
    SA_KEY, 3534,
    "Per-Class F1 — DistilBERT vs RoBERTa (SA Test, n=3534)",
    "03_per_class_f1_sa_test.png",
)
per_class_f1_chart(
    MAN_KEY, 516,
    "Per-Class F1 — DistilBERT vs RoBERTa (Manual 2009, n=516)",
    "04_per_class_f1_manual.png",
)


# ══════════════════════════════════════════════════════════════════════════════
# 05-08 — Confusion matrices
# ══════════════════════════════════════════════════════════════════════════════

def confusion_heatmap(model_key: str, eval_key: str, cmap: str, title: str, fname: str) -> None:
    cm = np.array(models[model_key]["evaluation"][eval_key]["confusion_matrix"]["matrix"])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct   = cm / row_sums * 100

    labels = ["Negative", "Neutral", "Positive"]
    annots = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annots[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_pct, annot=annots, fmt="", cmap=cmap,
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "Row %"},
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual",    fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    savefig(fig, fname)


confusion_heatmap("distilbert_stage2", SA_KEY,  "Blues",
                  "Confusion Matrix — DistilBERT S2 (SA Test, n=3534)",
                  "05_confusion_matrix_distilbert_sa.png")
confusion_heatmap("roberta_stage2",    SA_KEY,  "Greens",
                  "Confusion Matrix — RoBERTa S2 (SA Test, n=3534)",
                  "06_confusion_matrix_roberta_sa.png")
confusion_heatmap("distilbert_stage2", MAN_KEY, "Blues",
                  "Confusion Matrix — DistilBERT S2 (Manual 2009, n=516)",
                  "07_confusion_matrix_distilbert_manual.png")
confusion_heatmap("roberta_stage2",    MAN_KEY, "Greens",
                  "Confusion Matrix — RoBERTa S2 (Manual 2009, n=516)",
                  "08_confusion_matrix_roberta_manual.png")


# ══════════════════════════════════════════════════════════════════════════════
# 09 & 10 — Precision / Recall / F1 per class
# ══════════════════════════════════════════════════════════════════════════════

def prf_chart(model_key: str, title: str, fname: str) -> None:
    classes = ["negative", "neutral", "positive"]
    detail  = models[model_key]["evaluation"][SA_KEY]["per_class_detail"]

    metrics   = ["precision", "recall", "f1"]
    hatches   = {"precision": "",  "recall": "//", "f1": "xx"}
    m_labels  = {"precision": "Precision", "recall": "Recall", "f1": "F1"}

    n_classes = len(classes)
    n_metrics = len(metrics)
    x = np.arange(n_classes)
    w = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, metric in enumerate(metrics):
        vals = [detail[c][metric] * 100 for c in classes]
        offsets = x + (i - 1) * w
        bars = ax.bar(
            offsets, vals, w,
            color=[CLASS_COLORS[c] for c in classes],
            hatch=hatches[metric],
            edgecolor="white", label=m_labels[metric],
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.4,
                f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["Negative", "Neutral", "Positive"])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(50, 97)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    # Legend: metrics (hatches)
    metric_patches = [
        mpatches.Patch(facecolor="#cccccc", hatch=hatches[m], edgecolor="black", label=m_labels[m])
        for m in metrics
    ]
    # Class colour swatches
    class_patches = [
        mpatches.Patch(facecolor=CLASS_COLORS[c], label=c.capitalize())
        for c in classes
    ]
    ax.legend(handles=metric_patches + class_patches, fontsize=8, loc="lower right",
              ncol=2, framealpha=0.8)

    fig.tight_layout()
    savefig(fig, fname)


prf_chart("distilbert_stage2",
          "Precision / Recall / F1 — DistilBERT S2 (SA Test)",
          "09_precision_recall_f1_distilbert.png")
prf_chart("roberta_stage2",
          "Precision / Recall / F1 — RoBERTa S2 (SA Test)",
          "10_precision_recall_f1_roberta.png")


# ══════════════════════════════════════════════════════════════════════════════
# 11 & 12 — All-metrics heatmaps
# ══════════════════════════════════════════════════════════════════════════════

def metrics_heatmap(eval_key: str, title: str, fname: str) -> None:
    cols    = ["Accuracy", "Macro F1", "Weighted F1", "Neg F1", "Neu F1", "Pos F1"]
    rows    = []
    data    = []
    for m in MODEL_ORDER:
        ev = models[m]["evaluation"][eval_key]
        rows.append(MODEL_LABELS[m])
        data.append([
            ev["accuracy"]   * 100,
            ev["macro_f1"]   * 100,
            ev["weighted_f1"]* 100,
            ev["per_class_f1"]["negative"] * 100,
            ev["per_class_f1"]["neutral"]  * 100,
            ev["per_class_f1"]["positive"] * 100,
        ])

    data_np = np.array(data)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        data_np, annot=True, fmt=".2f", cmap="YlGn",
        xticklabels=cols, yticklabels=rows,
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "Score (%)"},
        vmin=40, vmax=90,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    savefig(fig, fname)


metrics_heatmap(SA_KEY,  "All Metrics Heatmap — SA Test (n=3534)",      "11_all_metrics_heatmap_sa.png")
metrics_heatmap(MAN_KEY, "All Metrics Heatmap — Manual 2009 (n=516)",   "12_all_metrics_heatmap_manual.png")


# ══════════════════════════════════════════════════════════════════════════════
# 13 — Model size vs Macro F1
# ══════════════════════════════════════════════════════════════════════════════

def size_vs_f1_chart() -> None:
    plot_models = [m for m in MODEL_ORDER if models[m]["evaluation"][SA_KEY]["n_params"] is not None]

    params = [models[m]["evaluation"][SA_KEY]["n_params"]  / 1e6 for m in plot_models]
    f1s    = [models[m]["evaluation"][SA_KEY]["macro_f1"]  * 100 for m in plot_models]
    colors = [MODEL_COLORS[m]  for m in plot_models]
    names  = [MODEL_LABELS[m]  for m in plot_models]

    offsets = {
        "distilbert_stage2": (-12, 1.0),
        "roberta_stage2":    (  2, 1.0),
        "bertweet_stage2":   (  2, 1.0),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for x, y, c, name, model_key in zip(params, f1s, colors, names, plot_models):
        ax.scatter(x, y, s=180, color=c, zorder=3, edgecolors="white", linewidths=0.8)
        dx, dy = offsets.get(model_key, (2, 1))
        ax.annotate(name, (x, y), xytext=(x + dx, y + dy), fontsize=9, color=c, fontweight="bold")

    best_f1 = max(f1s)
    ax.axhline(best_f1, color="#333333", linestyle="--", linewidth=1,
               label=f"Best F1 = {best_f1:.1f}")

    ax.set_xlabel("Number of Parameters (M)", fontsize=10)
    ax.set_ylabel("Macro F1 on SA Test (%)",   fontsize=10)
    ax.set_title("Model Size vs Macro F1 (SA Test)", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "13_model_size_vs_f1.png")


size_vs_f1_chart()

# ── Summary ────────────────────────────────────────────────────────────────────
print("\nAll plots generated:")
for p in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  {p.name:55s}  {p.stat().st_size:>8,} bytes")
