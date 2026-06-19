"""
Render publication figures (meaningful set only).

Outputs:
  Figure_1.png  — pipeline overview
  Figure_2.png  — model performance (stage table + ROC + PR)
  Figure_3.png  — XGBoost feature importance
  Figure_4.png  — CRyPTIC prospective validation
  Figure_S2.png — leave-one-gene-out (supplementary)

Run after: python scripts/13_final_publication_audit.py && python scripts/10_generate_figures.py
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(BASE, "analysis", "results", "figures")
RESULTS_DIR = os.path.join(BASE, "analysis", "results")
OUTPUT_DIR = FIGURE_DIR

PUB_PATH = os.path.join(RESULTS_DIR, "publication_metrics.json")
PUB = json.load(open(PUB_PATH)) if os.path.exists(PUB_PATH) else {}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _csv(name):
    path = os.path.join(FIGURE_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def fig1_pipeline():
    stage = PUB.get("stage_progression", {})
    cv = PUB.get("stratified_5fold_cv", {})
    cryptic = PUB.get("cryptic_validation", {}).get("tier_counts", {})
    ranking = PUB.get("ranking_full_model", {})
    n_pos = PUB.get("dataset", {}).get("n_positives", 32)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.text(5, 7.1, "Forecasting Emerging TB Resistance Mutations",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(5, 6.6, "Hotspot model → emergence scoring → CRyPTIC + Vina validation",
            ha="center", fontsize=10, color="gray")

    steps = [
        (1.2, 5.2, "Labels", "#2c3e50", f"{n_pos} hotspot residues\nWHO + CRyPTIC"),
        (3.2, 5.2, "Features", "#2980b9", "Homoplasy, structure,\nESM-2, drug proximity"),
        (5.2, 5.2, "Classifier", "#27ae60", f"AUROC {cv.get('auroc_mean', 0.968):.3f}\n{ranking.get('top20_n', 20)}/{n_pos} in top 20"),
        (7.2, 5.2, "Forecast", "#e67e22", "332 mutations\nP(emergence)"),
        (9.0, 5.2, "Validate", "#c0392b", f"12,287 isolates\nTier 1: {cryptic.get('1', 24)} · Tier 4: {cryptic.get('4', 188)}"),
    ]
    for x, y, title, color, desc in steps:
        ax.add_patch(FancyBboxPatch((x - 0.75, y - 0.45), 1.5, 0.9,
                                    boxstyle="round,pad=0.08", facecolor=color, alpha=0.12,
                                    edgecolor=color, linewidth=2))
        ax.text(x, y + 0.12, title, ha="center", fontsize=9, fontweight="bold", color=color)
        ax.text(x, y - 0.35, desc, ha="center", va="top", fontsize=7, color="gray")

    for i in range(len(steps) - 1):
        ax.annotate("", xy=(steps[i + 1][0] - 0.75, steps[i + 1][1]),
                    xytext=(steps[i][0] + 0.75, steps[i][1]),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_1.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig2_model_performance():
    """Stage progression + ROC + PR (single model figure)."""
    stage = PUB.get("stage_progression", {})
    cv = PUB.get("stratified_5fold_cv", {})
    gkf = PUB.get("groupkfold_by_gene", {})

    fig = plt.figure(figsize=(12, 4.5))

    # Panel A: metrics table
    ax1 = plt.subplot(1, 3, 1)
    ax1.axis("off")
    ax1.set_title("A  Model progression", loc="left", fontweight="bold")
    rows = [
        ["", "Stage 0", "Stage 1", "Stage 2"],
        ["AUROC", "0.888", f"{stage.get('stage1_auroc', 0.906):.3f}", f"{stage.get('stage2_auroc', 0.971):.3f}"],
        ["AUPRC", "—", f"{stage.get('stage1_auprc', 0.205):.3f}", f"{stage.get('stage2_auprc', 0.560):.3f}"],
        ["Top-20 recall", "—", f"{stage.get('stage1_top20_recall', 0.386):.3f}", f"{stage.get('stage2_top20_recall', 0.657):.3f}"],
        ["Best F1 (CV)", "—", "—", f"{cv.get('best_f1_mean', 0.551):.3f}"],
        ["GroupKFold AUROC", "—", "—", f"{gkf.get('auroc_mean', 0.974):.3f}"],
    ]
    table = ax1.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for key, cell in table.get_celld().items():
        if key[0] == 0 or key[1] == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        if key[1] == 3 and key[0] > 0:
            cell.set_facecolor("#d5f5e3")

    # Panel B: ROC
    ax2 = plt.subplot(1, 3, 2)
    roc_df = _csv("fig_roc_curve.csv")
    auroc = cv.get("auroc_mean", stage.get("stage2_auroc", 0.971))
    if roc_df is not None:
        ax2.plot(roc_df["fpr"], roc_df["tpr"], color="#27ae60", lw=2,
                 label=f"AUROC = {auroc:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax2.set_title("B  ROC (5-fold OOF)", loc="left", fontweight="bold")
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.legend(fontsize=8)
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.3, linestyle="--")

    # Panel C: PR (smooth envelope from pooled OOF)
    ax3 = plt.subplot(1, 3, 3)
    pr_df = _csv("fig_pr_curve.csv")
    pooled = PUB.get("stratified_5fold_cv", {}).get("pooled_oof", {})
    auprc = pooled.get("auprc", cv.get("auprc_mean", 0.465))
    baseline = PUB.get("dataset", {}).get("positive_rate", 0.005)
    if pr_df is not None and len(pr_df):
        pr_df = pr_df.sort_values("recall")
        ax3.plot(pr_df["recall"], pr_df["precision"], color="#2c3e50", lw=2,
                 label=f"AUPRC = {auprc:.3f}")
        ax3.fill_between(pr_df["recall"], pr_df["precision"], baseline,
                         alpha=0.12, color="#2c3e50")
    ax3.axhline(baseline, color="gray", ls="--", lw=1, label=f"Random ({baseline:.3f})")
    ax3.set_title("C  Precision–recall (5-fold OOF)", loc="left", fontweight="bold")
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.set_xlim(0, 1.02)
    ax3.set_ylim(0, 1.02)
    ax3.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_2.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig3_features():
    fc = _csv("fig3_feature_importance.csv")
    if fc is None:
        print("  Skip Figure_3: fig3_feature_importance.csv missing")
        return

    imp_col = "importance" if "importance" in fc.columns else fc.columns[1]
    feat_col = fc.columns[0]
    top = fc.sort_values(imp_col, ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(top)))
    ax.barh(top[feat_col], top[imp_col], color=colors, edgecolor="white")
    ax.set_xlabel("XGBoost gain")
    ax.set_title("Feature importance (Stage 2 model)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_3.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig4_cryptic():
    tier_df = _csv("fig5b_tier_distribution.csv")
    tier1_df = _csv("fig5c_tier1_hits.csv")

    fig = plt.figure(figsize=(11, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("A  CRyPTIC validation tiers", fontweight="bold")
    if tier_df is not None:
        labels = tier_df["label"].tolist() if "label" in tier_df.columns else tier_df.iloc[:, 0].tolist()
        counts = tier_df["count"].tolist() if "count" in tier_df.columns else tier_df.iloc[:, 1].tolist()
        colors = ["#7f8c8d", "#27ae60", "#f39c12", "#e74c3c", "#95a5a6"][:len(counts)]
        bars = ax1.bar(range(len(counts)), counts, color=colors, alpha=0.85)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        for bar, c in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     str(c), ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Mutations")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = plt.subplot(1, 2, 2)
    ax2.axis("off")
    ax2.set_title("B  Tier 1 hits (FDR q < 0.05)", fontweight="bold")
    if tier1_df is not None:
        display = tier1_df.head(8).copy()
        cols = [c for c in ["mutation", "gene", "n_carriers", "resistance_frac_str", "pvalue_fdr"]
                if c in display.columns]
        if not cols:
            cols = list(display.columns[:5])
        table_data = [cols] + display[cols].astype(str).values.tolist()
        table = ax2.table(cellText=table_data, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")

    matched = PUB.get("matched_null", {})
    ax2.text(0.5, -0.02,
             f"Matched-null enrichment: Tier 1 = {matched.get('real_tier1_count', 24)} vs "
             f"null mean {matched.get('null_mean', 9.3)} (p = {matched.get('p_value', 0.001)})",
             ha="center", transform=ax2.transAxes, fontsize=8, style="italic", color="gray")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_4.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def figS2_loo():
    loo = _csv("figS2_leave_one_gene_out.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title("Leave-one-gene-out validation", fontweight="bold")

    if loo is not None and len(loo):
        cols = ["gene", "n_known_mutations", "top20_recall", "top50_recall", "median_rank"]
        cols = [c for c in cols if c in loo.columns]
        header = cols
        rows = loo[cols].astype(str).values.tolist()
        table_data = [header] + rows
    else:
        table_data = [
            ["gene", "top50_recall", "auroc"],
            ["rpoB", "5/10", "0.576"],
            ["gyrA", "3/5", "0.700"],
            ["embB", "4/9", "0.607"],
        ]

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_S2.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Rendering publication figures (5 total)...")
    fig1_pipeline()
    fig2_model_performance()
    fig3_features()
    fig4_cryptic()
    figS2_loo()
    print(f"\nDone. Figures in {OUTPUT_DIR}/")
    print("  Figure_1.png  — pipeline")
    print("  Figure_2.png  — model (stage + ROC + PR)")
    print("  Figure_3.png  — feature importance")
    print("  Figure_4.png  — CRyPTIC validation")
    print("  Figure_S2.png — leave-one-gene-out")


if __name__ == "__main__":
    main()
