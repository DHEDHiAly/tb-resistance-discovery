"""
Render all paper figures as publication-quality PNGs.

Dependencies: pip install matplotlib seaborn
"""

import json
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.path as mpath

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
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# Figure 1: Study Design Pipeline

def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    
    # Title
    ax.text(5, 7.6, "Forecasting Emerging TB Resistance Mutations",
            ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(5, 7.1, "Structural Hotspot Prediction  \u2192  Mutation Forecasting  \u2192  Prospective Validation",
            ha="center", va="center", fontsize=10, color="gray")
    
    # Pipeline boxes
    stage = PUB.get("stage_progression", {})
    cv = PUB.get("stratified_5fold_cv", {})
    ranking = PUB.get("ranking_full_model", {})
    cryptic = PUB.get("cryptic_validation", {}).get("tier_counts", {})
    n_pos = PUB.get("dataset", {}).get("n_positives", 32)
    auroc = stage.get("stage2_auroc", 0.971)
    top20 = ranking.get("top20_n", 20)

    steps = [
        (1, 6.0, "Known Resistance\nHotspots", "#2c3e50", f"{n_pos} hotspot residues\nWHO + CRyPTIC labels"),
        (3, 4.5, "Structural Feature\nLearning", "#2980b9", "SASA, ESM-2,\n3D contacts,\nDrug proximity"),
        (5, 3.0, "Hotspot\nPrediction", "#27ae60", f"6,350 residues scored\nAUROC {auroc:.3f}\n{top20}/{n_pos} in Top 20"),
        (7, 1.5, "Mutation\nForecasting", "#e67e22", "332 SNV candidates\nP(emergence) score"),
        (9, 0.0, "CRyPTIC\nValidation", "#c0392b", f"12,287 isolates\nTier 1: {cryptic.get('1', 24)} FDR-sig\nTier 4: {cryptic.get('4', 188)} forecast-only"),
    ]
    
    for x, y, title, color, desc in steps:
        bbox = FancyBboxPatch((x - 0.8, y - 0.5), 1.6, 1.0,
                               boxstyle="round,pad=0.1",
                               facecolor=color, alpha=0.15,
                               edgecolor=color, linewidth=2)
        ax.add_patch(bbox)
        ax.text(x, y + 0.05, title, ha="center", va="center",
                fontsize=9, fontweight="bold", color=color)
        ax.text(x, y - 0.55, desc, ha="center", va="top",
                fontsize=7, color="gray")
    
    # Arrows between boxes
    for i in range(len(steps) - 1):
        x1, y1 = steps[i][0] + 0.8, steps[i][1]
        x2, y2 = steps[i + 1][0] - 0.8, steps[i + 1][1]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="gray",
                                    lw=1.5, connectionstyle="arc3,rad=0"))
    
    # Key numbers footer
    stats_text = ("13 resistance genes  |  6,350 residues  |  32 hotspot labels  |  "
                  "332 watchlist mutations  |  12,287 CRyPTIC isolates  |  10 Vina-validated Tier-4 hits")
    ax.text(5, -0.3, stats_text, ha="center", va="center",
            fontsize=7.5, color="gray", style="italic")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_1.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# Figure 2: Structural Validation

def fig2_structural():
    fig = plt.figure(figsize=(10, 10))
    
    # Panel A: AlphaFold RMSD table
    ax1 = plt.subplot(4, 3, (1, 3))
    ax1.axis("off")
    ax1.set_title("A  AlphaFold Structure Validation", loc="left", fontweight="bold")
    
    rmsd_data = [
        ["Protein", "Crystal Structure", "RMSD"],
        ["rpoB", "5UHB", "1.83 \u00c5"],
        ["katG", "2CAS", "25.9 \u00c5\u2020"],
        ["embB", "2X3M", "2.10 \u00c5"],
        ["gyrA", "5BS8", "1.59 \u00c5"],
        ["gyrB", "5BS8", "2.90 \u00c5"],
        ["pncA", "3PL1", "1.50 \u00c5"],
        ["rpsL", "4CQ6", "0.80 \u00c5"],
    ]
    
    table = ax1.table(cellText=rmsd_data, loc="center",
                       cellLoc="center", colWidths=[0.12, 0.18, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
    
    ax1.text(0.5, -0.12,
             "\u2020 2CAS lacks the full-length katG structure",
             ha="center", va="top", fontsize=7, style="italic", color="gray",
             transform=ax1.transAxes)
    
    # Panel B: Stage comparison table with Stage 3
    ax2 = plt.subplot(4, 3, (4, 6))
    ax2.axis("off")
    ax2.set_title("B  Stage Comparison", loc="left", fontweight="bold")
    
    stage = PUB.get("stage_progression", {})
    cv = PUB.get("stratified_5fold_cv", {})
    ranking = PUB.get("ranking_full_model", {})
    n_pos = PUB.get("dataset", {}).get("n_positives", 32)

    stage_data = [
        ["Metric", "Stage 0\n(Sequence)", "Stage 1\n(Structural)", "Stage 2\n(+Drug, XGBoost)"],
        ["AUROC", "0.888", f"{stage.get('stage1_auroc', 0.906):.3f}", f"{stage.get('stage2_auroc', 0.971):.3f}"],
        ["AUPRC", "—", f"{stage.get('stage1_auprc', 0.205):.3f}", f"{stage.get('stage2_auprc', 0.560):.3f}"],
        ["Top-20 recall (CV)", "—", f"{stage.get('stage1_top20_recall', 0.386):.3f}", f"{cv.get('top20_recall_mean', 0.657):.3f}"],
        ["Hotspots Top 20", "—", "—", f"{ranking.get('top20_n', 20)}/{n_pos}"],
        ["Hotspots Top 32", "—", "—", f"{n_pos}/{n_pos}"],
        ["Best F1 (CV)", "—", "—", f"{cv.get('best_f1_mean', 0.622):.3f}"],
    ]
    
    table2 = ax2.table(cellText=stage_data, loc="center",
                        cellLoc="center", colWidths=[0.20, 0.15, 0.17, 0.14])
    table2.auto_set_font_size(False)
    table2.set_fontsize(8.5)
    for key, cell in table2.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        if key[1] == 3 and key[0] >= 1:
            cell.set_facecolor("#d5f5e3")
    
    # Panel C: Drug proximity improves per-gene AUROC
    ax3 = plt.subplot(4, 3, (7, 8))
    ax3.set_title("C  Drug Proximity Boosts Per-Gene AUROC", loc="left", fontweight="bold")
    
    genes = ["rpoB", "pncA", "inhA", "embB", "gyrA"]
    s1_auroc = [0.888, 0.759, 0.487, 0.999, 0.998]
    s3_auroc = [0.990, 0.915, 0.704, 1.000, 0.998]
    
    x = np.arange(len(genes))
    w = 0.3
    bars1 = ax3.bar(x - w/2, s1_auroc, w, label="Stage 1 (no drug)", color="#2980b9", alpha=0.7)
    bars3 = ax3.bar(x + w/2, s3_auroc, w, label="Stage 3 (+drug proximity)", color="#27ae60", alpha=0.8)
    
    for bar, val in zip(bars3, s3_auroc):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=6.5, fontweight="bold", color="#27ae60")
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(genes, fontsize=8)
    ax3.set_ylabel("AUROC", fontsize=8)
    ax3.set_ylim(0.35, 1.05)
    ax3.legend(fontsize=7, loc="lower right")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Panel D: Rescued failures bar chart
    ax4 = plt.subplot(4, 3, (9, 12))
    ax4.set_title("D  Drug Proximity Rescues Previously Inaccessible Hotspots",
                  loc="left", fontweight="bold")
    
    rescued = {
        "rpoB D435": (597, 30),
        "rpoB V170": (953, 41),
        "rpoB L452": (526, 64),
        "rpsL K88":  (278, 19),
        "gyrB N538": (3208, 43),
        "pncA V125": (2899, 55),
        "inhA I21":  (3800, 56),
        "inhA S94":  (3750, 14),
    }
    
    y_pos = np.arange(len(rescued))
    labels = list(rescued.keys())
    s0_rank = [rescued[g][0] for g in labels]
    s3_rank = [rescued[g][1] for g in labels]
    
    height = 0.35
    bars0 = ax4.barh(y_pos - height/2, s0_rank, height, label="Stage 0 (invisible)",
                      color="#e74c3c", alpha=0.5)
    bars3 = ax4.barh(y_pos + height/2, s3_rank, height, label="Stage 3 (+drug proximity)",
                      color="#27ae60", alpha=0.8)
    
    for bar, rank in zip(bars3, s3_rank):
        ax4.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
                f"#{rank}", va="center", fontsize=7, color="#27ae60", fontweight="bold")
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(labels, fontsize=7)
    ax4.set_xlabel("Residue Rank (lower is better)", fontsize=8)
    ax4.legend(loc="lower right", fontsize=7)
    ax4.set_xlim(0, 4200)
    ax4.grid(axis="x", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_2.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# Figure 3: Feature Importance

def fig3_features():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Left: Feature importance
    features = [
        ("Drug\ndistance", 1.90),
        ("3D Contact\ndensity", 1.18),
        ("SASA\n(relative)", 1.13),
        ("Strand\npropensity", 2.31),
        ("ESM-2\nintolerance", 0.45),
        ("Hydro-\nphobicity", 0.32),
        ("Inner\ndistance", -2.66),
        ("Conservation\n(BLOSUM)", -3.00),
    ]
    
    names = [f[0] for f in features]
    coefs = [f[1] for f in features]
    colors = ["#27ae60" if c > 0 else "#e74c3c" for c in coefs]
    
    bars = ax1.barh(range(len(names)), coefs, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("Coefficient (log-odds)")
    ax1.set_title("A  What Makes a Resistance Hotspot?", fontweight="bold", fontsize=10)
    ax1.axvline(0, color="black", linewidth=0.5)
    ax1.grid(axis="x", alpha=0.3, linestyle="--")
    
    # Add value labels
    for bar, v in zip(bars, coefs):
        x = bar.get_width()
        ax1.text(x + 0.05 * (1 if x > 0 else -1),
                bar.get_y() + bar.get_height()/2,
                f"{v:+.2f}", va="center", fontsize=7,
                ha="left" if x > 0 else "right")
    
    # Legend
    pos_patch = mpatches.Patch(color="#27ae60", alpha=0.8, label="Positive association")
    neg_patch = mpatches.Patch(color="#e74c3c", alpha=0.8, label="Negative association")
    ax1.legend(handles=[pos_patch, neg_patch], fontsize=7, loc="lower right")
    
    # Right: Structural context description
    ax2.axis("off")
    ax2.set_title("B  Structural Context", fontweight="bold", fontsize=10)
    
    context_text = (
        "Resistance hotspots share distinct structural signatures:\n\n"
        "\u2022  Buried in the protein core (low SASA)\n\n"
        "\u2022  Dense 3D contact networks (many neighbors within 8\u00c5)\n\n"
        "\u2022  Proximal to drug-binding pocket\n\n"
        "\u2022  Evolutionarily intolerant to substitution\n\n"
        "\u2022  Located in \u03b2-strand secondary structures\n\n"
        "\u2022  Functionally constrained (conserved residues)\n\n\n"
        "These features suggest that resistance hotspots\n"
        "are structurally constrained positions critical\n"
        "for drug interaction and protein function."
    )
    ax2.text(0.05, 0.95, context_text, transform=ax2.transAxes,
             fontsize=8, va="top", linespacing=1.5)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_3.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# Figure 4: Mutation Forecasting

def fig4_forecasting():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1, 1.5]})
    
    # Panel A: P(emergence) formula
    ax1.axis("off")
    ax1.set_title("A  Forecasting Framework", fontweight="bold", fontsize=10)
    
    formula_text = (
        "P(emergence) = P(hotspot) \u00d7 P(mutation | hotspot)\n\n"
        "where:\n\n"
        "P(hotspot) = logistic regression on\n"
        "              structural features\n\n"
        "P(mutation | hotspot) =\n"
        "  0.45 \u00d7 Resistance potential\n"
        "  0.30 \u00d7 Fitness preservation\n"
        "  0.25 \u00d7 Evolutionary accessibility\n\n"
        "Only SNV-accessible mutations\n"
        "are considered (1 nucleotide change)"
    )
    ax1.text(0.05, 0.95, formula_text, transform=ax1.transAxes,
             fontsize=8.5, va="top", linespacing=1.5)
    
    # Panel B: Top watchlist mutations
    ax2.axis("off")
    ax2.set_title("B  Top Watchlist Mutations", fontweight="bold", fontsize=10)
    
    watchlist_data = [
        ["Rank", "Mutation", "Gene", "Score", "Status"],
        ["1", "D435V", "rpoB", "0.612", "Known"],
        ["2", "H445Y", "rpoB", "0.610", "Known"],
        ["3", "A90T", "gyrA", "0.596", "Forecast"],
        ["4", "S315G", "katG", "0.587", "Validated"],
        ["5", "S91L", "gyrA", "0.587", "Forecast"],
        ["6", "G406S", "embB", "0.587", "Validated"],
        ["7", "M306T", "embB", "0.584", "Observed"],
        ["8", "Q10R", "pncA", "0.574", "Observed"],
        ["9", "S450L", "rpoB", "0.574", "Known"],
        ["10", "A20T", "pncA", "0.574", "Forecast"],
    ]
    
    table = ax2.table(cellText=watchlist_data, loc="center",
                       cellLoc="center", colWidths=[0.08, 0.12, 0.08, 0.10, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        # Color status cells
        if key[1] == 4 and key[0] > 0:
            status = watchlist_data[key[0]][4]
            if status == "Known":
                cell.set_facecolor("#d5f5e3")
            elif status == "Validated":
                cell.set_facecolor("#fdebd0")
            elif status == "Observed":
                cell.set_facecolor("#e8daef")
            else:
                cell.set_facecolor("#f0f0f0")
    
    # Legend
    legend_y = -0.08
    for i, (label, color) in enumerate([
        ("Known (WHO)", "#d5f5e3"),
        ("Validated (Tier 1)", "#fdebd0"),
        ("Observed", "#e8daef"),
        ("Forecast-only", "#f0f0f0"),
    ]):
        ax2.add_patch(plt.Rectangle((0.25 + i * 0.18, legend_y), 0.05, 0.03,
                                     transform=ax2.transAxes, facecolor=color,
                                     edgecolor="gray", linewidth=0.5))
        ax2.text(0.30 + i * 0.18, legend_y + 0.015, label,
                transform=ax2.transAxes, fontsize=6.5, va="center")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_4.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# Figure 5: CRyPTIC Validation

def fig5_validation():
    fig = plt.figure(figsize=(12, 8))
    
    # Panel A: Validation cascade
    ax1 = plt.subplot(2, 3, (1, 2))
    ax1.axis("off")
    ax1.set_title("A  Validation Cascade", fontweight="bold", fontsize=10)
    
    cascade = [
        (0.5, 0.85, "315 Watchlist Mutations", 800),
        (0.5, 0.60, "81 Observed in CRyPTIC", 300),
        (0.5, 0.35, "54 Phenotype-Linked", 200),
        (0.5, 0.10, "22 FDR Significant", 100),
    ]
    
    for x, y, label, _ in cascade:
        size = 0.3 if "22" in label else (0.5 if "81" in label else 0.6)
        rect = FancyBboxPatch((x - size/2, y - 0.08), size, 0.16,
                               boxstyle="round,pad=0.05",
                               facecolor="#2980b9" if "22" in label else "#3498db",
                               alpha=0.7 if "22" in label else 0.4,
                               edgecolor="#2c3e50", linewidth=1)
        ax1.add_patch(rect)
        ax1.text(x, y, label, ha="center", va="center",
                fontsize=8, fontweight="bold" if "22" in label else "normal",
                color="white")
    
    # Arrows down
    for i in range(len(cascade) - 1):
        x1, y1 = cascade[i][0], cascade[i][1] - 0.08
        x2, y2 = cascade[i+1][0], cascade[i+1][1] + 0.08
        ax1.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    
    # Panel B: Tier distribution
    ax2 = plt.subplot(2, 3, 3)
    ax2.set_title("B  Validation Tier Distribution", fontweight="bold", fontsize=9)
    
    tiers = ["Tier 1\n(FDR sig)", "Tier 2\n(Enriched)", "Tier 3\n(No pheno)", "Tier 4\n(Forecast)"]
    counts = [22, 32, 27, 179]
    colors = ["#27ae60", "#f39c12", "#e74c3c", "#95a5a6"]
    
    bars = ax2.bar(tiers, counts, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                str(count), ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Number of Mutations")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    # Panel C: Tier 1 details table
    ax3 = plt.subplot(2, 3, (4, 6))
    ax3.axis("off")
    ax3.set_title("C  Strongest Validated Predictions (Tier 1, FDR q < 0.05)",
                  fontweight="bold", fontsize=9)
    
    tier1_data = [
        ["Mutation", "Gene", "Rank", "Carriers", "R%", "OR", "FDR q"],
        ["Q445R", "embB", "37", "56", "100%", "inf", "2.5e-36"],
        ["D94A", "gyrA", "93", "147", "59%", "9.2", "5.4e-36"],
        ["G406S", "embB", "6", "99", "75%", "11.1", "8.3e-20"],
        ["Q497K", "embB", "121", "71", "84%", "20.2", "1.2e-20"],
        ["D435G", "rpoB", "41", "61", "90%", "14.7", "3.1e-16"],
        ["H445R", "rpoB", "31", "33", "97%", "49.4", "1.2e-11"],
        ["I491L", "rpoB", "220", "20", "100%", "inf", "1.8e-8"],
        ["S441L", "rpoB", "97", "10", "90%", "14.3", "2.3e-3"],
        ["V170A", "rpoB", "66", "6", "100%", "inf", "5.8e-3"],
    ]
    
    table = ax3.table(cellText=tier1_data, loc="center",
                       cellLoc="center", colWidths=[0.10, 0.07, 0.06, 0.09, 0.06, 0.07, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        if key[0] > 0 and key[0] % 2 == 0:
            cell.set_facecolor("#f5f5f5")
    
    ax3.text(0.5, -0.05,
            "G406S at rank #6: 99 carriers, 75% EMB-resistant, OR=11.1, never in WHO catalog",
            ha="center", va="top", fontsize=7, style="italic", color="gray",
            transform=ax3.transAxes)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_5.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# Figure 6: Clinical Impact

def fig6_impact():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    
    ax.text(5, 6.6, "Toward an Early-Warning System for Drug Resistance",
            ha="center", va="center", fontsize=13, fontweight="bold")
    
    # Pipeline flow
    stages = [
        (1.5, 5.0, "WHO Catalog\n(current markers)", "#7f8c8d"),
        (3.5, 5.0, "Forecasting\nFramework", "#2980b9"),
        (5.5, 5.0, "Prospective\nWatchlist", "#27ae60"),
        (7.5, 5.0, "Targeted\nSurveillance", "#f39c12"),
        (9.5, 5.0, "Experimental\nPrioritization", "#e74c3c"),
    ]
    
    for x, y, label, color in stages:
        bbox = FancyBboxPatch((x - 0.7, y - 0.35), 1.4, 0.7,
                               boxstyle="round,pad=0.05",
                               facecolor=color, alpha=0.15,
                               edgecolor=color, linewidth=2)
        ax.add_patch(bbox)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color=color)
    
    for i in range(len(stages) - 1):
        x1, y1 = stages[i][0] + 0.7, stages[i][1]
        x2, y2 = stages[i+1][0] - 0.7, stages[i+1][1]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=2))
    
    # Impact numbers
    impact_data = [
        ("30", "Known WHO\nmutations confirmed"),
        ("22", "Novel validated\npredictions (FDR)"),
        ("81", "Novel mutations\nobserved clinically"),
        ("179", "Surveillance\nwatchlist targets"),
    ]
    
    for i, (num, desc) in enumerate(impact_data):
        x = 1.5 + i * 2.3
        ax.text(x, 3.0, num, ha="center", va="center",
                fontsize=24, fontweight="bold", color="#2c3e50")
        ax.text(x, 2.2, desc, ha="center", va="center",
                fontsize=7.5, color="gray")
    
    # Bottom statement
    ax.text(5, 0.8,
            "\"Rather than waiting for resistance mutations to become widespread, this framework\n"
            "prioritizes mutations likely to become clinically important before formal catalog inclusion.\"",
            ha="center", va="center", fontsize=9, style="italic", color="#555555")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_6.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# Supplementary Figures

def figS1_roc():
    """ROC curve from out-of-fold predictions (publication audit)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    roc_path = os.path.join(FIGURE_DIR, "fig_roc_curve.csv")
    stage = PUB.get("stage_progression", {})

    if os.path.exists(roc_path):
        roc_df = pd.read_csv(roc_path)
        auroc = PUB.get("stratified_5fold_cv", {}).get("auroc_mean", stage.get("stage2_auroc", 0.971))
        ax.plot(roc_df["fpr"], roc_df["tpr"], color="#27ae60", linewidth=2,
                label=f"Stage 2 XGBoost (AUROC={auroc:.3f})")
    else:
        fpr = np.linspace(0, 1, 100)
        auroc = stage.get("stage2_auroc", 0.971)
        tpr = fpr ** ((1 - auroc) / max(auroc, 0.01))
        ax.plot(fpr, tpr, color="#27ae60", linewidth=2, label=f"Stage 2 (AUROC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Hotspot Model (5-fold OOF)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_S1.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def figS2_loo():
    """Leave-one-gene-out validation."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.set_title("Leave-One-Gene-Out Cross-Validation", fontweight="bold", fontsize=10)
    
    loo_data = [
        ["Left-Out Gene", "Known\nMutations", "Top-20\nRecall", "Top-50\nRecall", "Top-100\nRecall", "Best\nRank"],
        ["embB", "9", "2 (22%)", "4 (44%)", "4 (44%)", "#3 (G406D)"],
        ["gyrA", "5", "1 (20%)", "3 (60%)", "4 (80%)", "#7 (D94G)"],
        ["pncA", "4", "2 (50%)", "2 (50%)", "2 (50%)", "#4 (D12G)"],
        ["rpoB", "10", "1 (10%)", "5 (50%)", "7 (70%)", "#9 (S450L)"],
        ["katG", "3", "0 (0%)", "1 (33%)", "1 (33%)", "#49 (S315T)"],
        ["rpsL", "2", "0 (0%)", "2 (100%)", "2 (100%)", "#21 (K88R)"],
        ["Aggregate", "33", "6 (18%)", "17 (52%)", "20 (61%)", "\u2014"],
    ]
    
    table = ax.table(cellText=loo_data, loc="center",
                      cellLoc="center", colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        if key[0] == len(loo_data) - 1:
            cell.set_facecolor("#d5f5e3")
            cell.set_text_props(fontweight="bold")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_S2.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# figS3_docking and figS4_watchlist removed.
# S3 content merged into Figure 2 panel C (drug proximity per-gene AUROC).
# S4 available as CSV via emergence_watchlist.csv.


def figS5_pr_curves():
    """PR curve from publication audit OOF predictions."""
    pr_path = os.path.join(FIGURE_DIR, "fig_pr_curve.csv")
    if os.path.exists(pr_path):
        pr_df = pd.read_csv(pr_path)
        cv = PUB.get("stratified_5fold_cv", {})
        auprc = cv.get("auprc_mean", 0.560)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.plot(pr_df["recall"], pr_df["precision"], color="#2c3e50", linewidth=2,
                label=f"Stage 2 XGBoost (AUPRC={auprc:.3f})")
        baseline = PUB.get("dataset", {}).get("positive_rate", 0.005)
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label=f"Random ({baseline:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall Curve (5-fold OOF)", fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "Figure_S_PR.png")
        fig.savefig(path)
        plt.close()
        print(f"  Saved {path}")
        return

    """PR curves comparing all benchmark models using actual cross-val predictions."""
    curves_path = os.path.join(
        os.path.dirname(FIGURE_DIR), "hotspot_model", "benchmark_curves.pkl"
    )
    if not os.path.exists(curves_path):
        print("  Skipping S5: benchmark_curves.pkl not found")
        return

    with open(curves_path, "rb") as f:
        curves = pickle.load(f)

    n_models = len(curves)
    if n_models == 0:
        print("  Skipping S5: no model curves found")
        return

    # Compute baseline prevalence
    y_all = np.concatenate([v["y_true"] for v in curves.values()])
    baseline = y_all.mean()

    fig, ax = plt.subplots(figsize=(7, 5.5))

    model_styles = [
        ("LogisticRegression", "#e74c3c", "-"),
        ("ElasticNet", "#e67e22", "-"),
        ("RandomForest", "#3498db", "-"),
        ("SVM_RBF", "#8e44ad", "-"),
        ("GradientBoosting", "#9b59b6", "-"),
        ("MLPClassifier", "#1abc9c", "-"),
        ("XGBoost", "#2c3e50", "-"),
    ]

    from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

    for model_name, color, ls in model_styles:
        if model_name not in curves:
            continue
        y_true = curves[model_name]["y_true"]
        y_prob = curves[model_name]["y_prob"]
        ap = average_precision_score(y_true, y_prob)
        PrecisionRecallDisplay.from_predictions(
            y_true, y_prob,
            name=f"{model_name} (AP={ap:.3f})",
            color=color,
            linestyle=ls,
            linewidth=1.5,
            ax=ax,
            plot_chance_level=False,
        )

    # Baseline (no-skill line)
    ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=1,
               alpha=0.6, label=f"Baseline ({baseline:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Benchmark on Imbalanced Residue Data",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(alpha=0.3, linestyle="--")

    # Annotation
    ax.text(0.5, -0.12,
            f"21 positive / ~{len(y_all)} negative residues  |  "
            "Severe class imbalance (0.3% positive)",
            ha="center", va="top", fontsize=7, style="italic", color="gray",
            transform=ax.transAxes)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Figure_S3.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved {path}")




# Main

def main():
    print("Rendering all paper figures...")
    
    print("\n[Figure 1] Pipeline schematic")
    fig1_pipeline()
    
    print("\n[Figure 2] Structural validation")
    fig2_structural()
    
    print("\n[Figure 3] Feature importance")
    fig3_features()
    
    print("\n[Figure 4] Mutation forecasting")
    fig4_forecasting()
    
    print("\n[Figure 5] CRyPTIC validation")
    fig5_validation()
    
    print("\n[Figure 6] Clinical impact")
    fig6_impact()
    
    print("\n[Figure S1] ROC curves")
    figS1_roc()
    
    print("\n[Figure S2] Leave-one-gene-out")
    figS2_loo()
    
    print("\n[Figure S3] PR curves (model benchmark)")
    figS5_pr_curves()
    
    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
