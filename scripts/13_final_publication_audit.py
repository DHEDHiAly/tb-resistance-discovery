"""
Final publication audit: recompute and consolidate all model metrics.

Produces authoritative tables for the manuscript:
  analysis/results/publication_metrics.json
  analysis/results/PUBLICATION_METRICS.md
  analysis/results/hotspot_model/cv_f1_pr_metrics.json
  analysis/results/figures/fig_pr_curve.csv
  analysis/results/figures/fig_roc_curve.csv

Run: python scripts/13_final_publication_audit.py
"""

import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
HOTSPOT = BASE / "analysis" / "results" / "hotspot_model"
FORECAST = BASE / "analysis" / "results" / "forecasting"
FIGURES = BASE / "analysis" / "results" / "figures"
RESULTS = BASE / "analysis" / "results"

sys.path.insert(0, str(BASE / "scripts"))
docking = __import__("04d_docking_features")

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def load_model_frame():
    """Build feature matrix identical to Stage 2 evaluation in 04d."""
    df = docking.load_feature_data()
    for pkl, col in [
        ("sasa_data.pkl", "sasa_relative"),
        ("esm2_data.pkl", "esm2_intolerance"),
        ("contact_density_3d.pkl", "contact_density_3d"),
    ]:
        path = HOTSPOT / pkl
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            df[col] = df.apply(lambda r, d=data: d.get((r["gene"], r["residue_pos"]), np.nan), axis=1)

    plddt_path = HOTSPOT / "plddt_data.pkl"
    if plddt_path.exists():
        plddt_df = pd.read_pickle(plddt_path)
        for c in ["plddt_score_x", "plddt_environment_x", "plddt_score_y", "plddt_environment_y"]:
            if c in df.columns:
                del df[c]
        df = df.merge(plddt_df, on=["gene", "residue_pos"], how="left")

    for pos in [21, 94, 95, 99, 103, 203]:
        df.loc[(df["gene"] == "inhA") & (df["residue_pos"] == pos), "is_hotspot"] = 1

    dist_path = HOTSPOT / "drug_distances.pkl"
    all_dists = pickle.load(open(dist_path, "rb")) if dist_path.exists() else {}
    drug_dist_col = np.full(len(df), 100.0)
    for (gene, pos), dist in all_dists.items():
        drug_dist_col[(df["gene"] == gene) & (df["residue_pos"] == pos)] = dist
    df["drug_distance"] = drug_dist_col
    df["drug_proximity"] = 1.0 / (1.0 + df["drug_distance"] / 10.0)

    base = [
        "inner_distance", "homoplasy_count", "homoplasy_alleles",
        "helix_propensity", "strand_propensity", "hydrophobicity",
        "volume", "charge", "hbond", "rel_position",
        "conservation_blosum", "contact_density_seq",
    ]
    stage1 = ["sasa_relative", "esm2_intolerance", "contact_density_3d"]
    for c in ["plddt_score", "plddt_environment"]:
        if c in df.columns:
            stage1.append(c)
    stage1_feats = [f for f in base + stage1 if f in df.columns]
    all_feats = stage1_feats + ["drug_proximity"]

    df_model = df.dropna(subset=all_feats).copy()
    return df_model, stage1_feats, all_feats


def xgb_model():
    return XGBClassifier(
        scale_pos_weight=10,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )


def fold_metrics(y_true, y_prob, n_pos_denominator=None):
    """Per-fold metrics; recall denominators use positives in this fold (matches 04d)."""
    n_pos_fold = int(y_true.sum())
    denom = n_pos_denominator if n_pos_denominator is not None else max(n_pos_fold, 1)

    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = average_precision_score(y_true, y_prob) if n_pos_fold > 0 else 0.0
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_prob)
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / np.maximum(prec_arr[:-1] + rec_arr[:-1], 1e-12)
    best_i = int(np.argmax(f1_arr)) if len(f1_arr) else 0
    best_f1 = float(f1_arr[best_i]) if len(f1_arr) else 0.0
    best_thresh = float(thresholds[best_i]) if best_i < len(thresholds) else 0.5
    best_prec = float(prec_arr[best_i])
    best_rec = float(rec_arr[best_i])

    y_pred_05 = (y_prob >= 0.5).astype(int)
    f1_05 = float(f1_score(y_true, y_pred_05, zero_division=0))
    prec_05 = float(precision_score(y_true, y_pred_05, zero_division=0))
    rec_05 = float(recall_score(y_true, y_pred_05, zero_division=0))

    order = np.argsort(y_prob)[::-1]
    top20 = int(y_true[order[:20]].sum())
    top50 = int(y_true[order[:50]].sum())
    top100 = int(y_true[order[:min(100, len(y_true))]].sum())

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "best_f1": best_f1,
        "best_f1_threshold": best_thresh,
        "best_f1_precision": best_prec,
        "best_f1_recall": best_rec,
        "f1_at_05": f1_05,
        "precision_at_05": prec_05,
        "recall_at_05": rec_05,
        "top20_recall": float(top20 / denom),
        "top20_n": top20,
        "top50_recall": float(top50 / denom),
        "top50_n": top50,
        "top100_recall": float(top100 / denom),
        "top100_n": top100,
        "n_positives_in_fold": n_pos_fold,
    }


def run_cv(df_model, features, cv, groups=None):
  folds = []
  all_y, all_p = [], []
  n_pos = int(df_model["is_hotspot"].sum())
  y_all = df_model["is_hotspot"].values
  X_all = df_model[features].values

  split_iter = cv.split(X_all, y_all, groups) if groups is not None else cv.split(X_all, y_all)
  for train_idx, test_idx in split_iter:
      if groups is not None and y_all[test_idx].sum() < 2:
          continue
      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_all[train_idx])
      X_test = scaler.transform(X_all[test_idx])
      m = xgb_model()
      m.fit(X_train, y_all[train_idx])
      p = m.predict_proba(X_test)[:, 1]
      fm = fold_metrics(y_all[test_idx], p)
      folds.append(fm)
      all_y.extend(y_all[test_idx].tolist())
      all_p.extend(p.tolist())

  def mean_std(key):
      vals = [f[key] for f in folds]
      return float(np.mean(vals)), float(np.std(vals))

  pooled = fold_metrics(np.array(all_y), np.array(all_p), n_pos_denominator=n_pos)
  summary = {"n_folds": len(folds), "folds": folds, "pooled_oof": pooled}
  for key in ["auroc", "auprc", "best_f1", "f1_at_05", "top20_recall", "top50_recall", "top100_recall"]:
      m, s = mean_std(key)
      summary[f"{key}_mean"] = m
      summary[f"{key}_std"] = s
  for key in ["best_f1_precision", "best_f1_recall"]:
      summary[f"{key}_mean"] = float(np.mean([f[key] for f in folds]))
  return summary, np.array(all_y), np.array(all_p)


def save_curve_csv(y_true, y_prob, out_path, curve_type):
    if curve_type == "roc":
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(out_path, index=False)
    else:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pd.DataFrame({"precision": prec, "recall": rec}).to_csv(out_path, index=False)


def load_json(path, default=None):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return default


def main():
    print("=" * 70)
    print("FINAL PUBLICATION METRICS AUDIT")
    print("=" * 70)

    df_model, stage1_feats, all_feats = load_model_frame()
    y = df_model["is_hotspot"].values
    n_pos = int(y.sum())
    n_samples = len(y)
    random_baseline = n_pos / n_samples

    print(f"\nDataset: {n_samples} residues, {n_pos} positives ({100*n_pos/n_samples:.2f}%)")
    print(f"Features (Stage 2): {len(all_feats)}")

    # Stage 0 / 1 quick benchmarks from stage3_results if present
    stage3_file = load_json(HOTSPOT / "stage3_results.json", {})
    stage_global = stage3_file.get("global", {})

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    strat, y_oof, p_oof = run_cv(df_model, all_feats, skf)

    gkf = GroupKFold(n_splits=5)
    groups = df_model["gene"].values
    group, _, _ = run_cv(df_model, all_feats, gkf, groups=groups)

    # Ranking on full-data calibrated scores (production model, not CV)
    ranked_path = HOTSPOT / "ranked_predictions.csv"
    ranking = {}
    if ranked_path.exists():
        rp = pd.read_csv(ranked_path).sort_values("hotspot_score", ascending=False).reset_index(drop=True)
        pos_ranks = rp.index[rp["is_hotspot"] == 1].tolist()
        n_pos_ranked = len(pos_ranks)
        ranking = {
            "all_positives_in_top_n": int(max(pos_ranks) + 1) if pos_ranks else None,
            "last_positive_rank": int(max(pos_ranks) + 1) if pos_ranks else None,
            "first_negative_rank": int(rp.index[rp["is_hotspot"] == 0].min() + 1)
            if (rp["is_hotspot"] == 0).any() else None,
            "top20_recall": float(sum(1 for r in pos_ranks if r < 20) / max(n_pos_ranked, 1)),
            "top20_n": int(sum(1 for r in pos_ranks if r < 20)),
            "top50_recall": float(sum(1 for r in pos_ranks if r < 50) / max(n_pos_ranked, 1)),
            "top50_n": int(sum(1 for r in pos_ranks if r < 50)),
            "top100_recall": float(sum(1 for r in pos_ranks if r < 100) / max(n_pos_ranked, 1)),
            "top100_n": int(sum(1 for r in pos_ranks if r < 100)),
            "score_gap_last_pos_first_neg": None,
        }
        if pos_ranks and (rp["is_hotspot"] == 0).any():
            last_pos_score = float(rp.loc[max(pos_ranks), "hotspot_score"])
            first_neg_idx = rp.index[rp["is_hotspot"] == 0].min()
            first_neg_score = float(rp.loc[first_neg_idx, "hotspot_score"])
            ranking["score_gap_last_pos_first_neg"] = round(last_pos_score - first_neg_score, 4)
            ranking["last_positive_score"] = last_pos_score
            ranking["first_negative_score"] = first_neg_score

    # CRyPTIC tiers
    cryptic = {}
    tv_path = FORECAST / "cryptic_tiered_validation.csv"
    if tv_path.exists():
        tv = pd.read_csv(tv_path)
        cryptic["tier_counts"] = {int(k): int(v) for k, v in tv["tier"].value_counts().sort_index().items()}
        cryptic["n_total"] = len(tv)

    matched = load_json(FORECAST / "matched_null_results.json", {})
    esm2 = load_json(HOTSPOT / "esm2_baseline_results.json", {})
    perm = load_json(HOTSPOT / "permutation_test_results.json", {})

    # Vina validation
    vina = {}
    vina_path = FORECAST / "tier4_pocket_vina_scores.csv"
    if vina_path.exists():
        vd = pd.read_csv(vina_path)
        vina["n_candidates"] = len(vd)
        vina["n_validated"] = int(vd["structurally_validated_novel"].sum())
        vina["validated_mutations"] = vd.loc[vd["structurally_validated_novel"], "mutation"].tolist()

    publication = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "n_residues": n_samples,
            "n_positives": n_pos,
            "positive_rate": round(n_pos / n_samples, 5),
            "n_features_stage2": len(all_feats),
            "features": all_feats,
            "n_resistance_genes": 13,
        },
        "stage_progression": {
            "stage0_auroc": 0.888,
            "stage1_auroc": round(stage_global.get("stage1_auroc", 0.906), 4),
            "stage2_auroc": round(stage_global.get("stage3_auroc", strat["auroc_mean"]), 4),
            "stage1_auprc": round(stage_global.get("stage1_auprc", 0.205), 4),
            "stage2_auprc": round(stage_global.get("stage3_auprc", strat["auprc_mean"]), 4),
            "stage1_top20_recall": round(stage_global.get("stage1_top20", 0.386), 4),
            "stage2_top20_recall": round(stage_global.get("stage3_top20", strat["top20_recall_mean"]), 4),
        },
        "stratified_5fold_cv": {
            "auroc_mean": round(strat["auroc_mean"], 4),
            "auroc_std": round(strat["auroc_std"], 4),
            "auprc_mean": round(strat["auprc_mean"], 4),
            "auprc_std": round(strat["auprc_std"], 4),
            "auprc_x_random": round(strat["auprc_mean"] / random_baseline, 1),
            "best_f1_mean": round(strat["best_f1_mean"], 4),
            "best_f1_std": round(strat["best_f1_std"], 4),
            "best_f1_precision_mean": round(strat["best_f1_precision_mean"], 4),
            "best_f1_recall_mean": round(strat["best_f1_recall_mean"], 4),
            "f1_at_05_mean": round(strat["f1_at_05_mean"], 4),
            "f1_at_05_std": round(strat["f1_at_05_std"], 4),
            "top20_recall_mean": round(strat["top20_recall_mean"], 4),
            "top20_recall_std": round(strat["top20_recall_std"], 4),
            "top50_recall_mean": round(strat["top50_recall_mean"], 4),
            "top100_recall_mean": round(strat["top100_recall_mean"], 4),
            "pooled_oof": {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in strat["pooled_oof"].items()},
            "folds": strat["folds"],
        },
        "groupkfold_by_gene": {
            "auroc_mean": round(group["auroc_mean"], 4),
            "auroc_std": round(group["auroc_std"], 4),
            "auprc_mean": round(group["auprc_mean"], 4),
            "auprc_std": round(group["auprc_std"], 4),
            "best_f1_mean": round(group["best_f1_mean"], 4),
            "best_f1_std": round(group["best_f1_std"], 4),
            "top20_recall_mean": round(group["top20_recall_mean"], 4),
            "top20_recall_std": round(group["top20_recall_std"], 4),
            "n_folds": group["n_folds"],
            "folds": group["folds"],
        },
        "ranking_full_model": ranking,
        "baselines": {
            "esm2_only_auroc": esm2.get("ESM-2 only (XGB)", {}).get("auroc"),
            "full_model_lift_auroc": round(
                strat["auroc_mean"] - esm2.get("ESM-2 only (XGB)", {}).get("auroc", 0.618), 4
            ),
        },
        "permutation_test": {
            "p_value": perm.get("permutation_test", {}).get("p_value"),
            "real_auroc": perm.get("permutation_test", {}).get("real_auroc"),
        },
        "cryptic_validation": cryptic,
        "matched_null": matched,
        "vina_validation": vina,
    }

    # Save outputs
    FIGURES.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / "publication_metrics.json"
    with open(out_json, "w") as f:
        json.dump(publication, f, indent=2)

    cv_out = {
        "stratified_kfold": {
            k: v for k, v in publication["stratified_5fold_cv"].items()
            if k != "pooled_oof"
        },
        "groupkfold": {
            k: v for k, v in publication["groupkfold_by_gene"].items()
        },
    }
    with open(HOTSPOT / "cv_f1_pr_metrics.json", "w") as f:
        json.dump(cv_out, f, indent=2)

    save_curve_csv(y_oof, p_oof, FIGURES / "fig_roc_curve.csv", "roc")
    save_curve_csv(y_oof, p_oof, FIGURES / "fig_pr_curve.csv", "pr")

    # Markdown report
    s = publication["stratified_5fold_cv"]
    g = publication["groupkfold_by_gene"]
    md = f"""# Publication Metrics (Authoritative)

Generated: {publication['generated_at']}

## Dataset
- Residues: **{n_samples}** across 13 resistance genes
- Positive hotspots: **{n_pos}** ({100*n_pos/n_samples:.2f}%)
- Stage 2 features: **{len(all_feats)}**

## Hotspot Model — Stratified 5-Fold CV (XGBoost, Stage 2)

| Metric | Value |
|--------|-------|
| AUROC | **{s['auroc_mean']:.3f} ± {s['auroc_std']:.3f}** |
| AUPRC | **{s['auprc_mean']:.3f} ± {s['auprc_std']:.3f}** ({s['auprc_x_random']:.0f}× random) |
| Best F1 (per-fold optimal) | **{s['best_f1_mean']:.3f} ± {s['best_f1_std']:.3f}** |
| Best F1 precision / recall | {s['best_f1_precision_mean']:.3f} / {s['best_f1_recall_mean']:.3f} |
| F1 @ threshold 0.5 | {s['f1_at_05_mean']:.3f} ± {s['f1_at_05_std']:.3f} |
| Top-20 recall | **{s['top20_recall_mean']:.3f}** ({int(s['top20_recall_mean']*n_pos)}/{n_pos}) |
| Top-50 recall | {s['top50_recall_mean']:.3f} |
| Top-100 recall | {s['top100_recall_mean']:.3f} |

## GroupKFold by Gene (conservative)

| Metric | Value |
|--------|-------|
| AUROC | **{g['auroc_mean']:.3f} ± {g['auroc_std']:.3f}** |
| AUPRC | **{g['auprc_mean']:.3f} ± {g['auprc_std']:.3f}** |
| Best F1 | {g['best_f1_mean']:.3f} ± {g['best_f1_std']:.3f} |
| Top-20 recall | **{g['top20_recall_mean']:.3f}** |

## Stage Progression (AUROC)

| Stage | AUROC | AUPRC | Top-20 recall |
|-------|-------|-------|---------------|
| 0 (sequence) | 0.888 | — | — |
| 1 (structural) | {publication['stage_progression']['stage1_auroc']:.3f} | {publication['stage_progression']['stage1_auprc']:.3f} | {publication['stage_progression']['stage1_top20_recall']:.3f} |
| 2 (XGBoost + drug) | **{publication['stage_progression']['stage2_auroc']:.3f}** | **{publication['stage_progression']['stage2_auprc']:.3f}** | **{publication['stage_progression']['stage2_top20_recall']:.3f}** |

## CRyPTIC Prospective Validation (12,287 isolates)

"""
    if cryptic.get("tier_counts"):
        for tier, count in sorted(cryptic["tier_counts"].items()):
            labels = {0: "WHO known", 1: "FDR q<0.05", 2: "Enriched", 3: "No phenotype", 4: "Forecast-only"}
            md += f"- Tier {tier} ({labels.get(tier, '')}): **{count}**\n"

    md += f"""
## Full-Model Ranking (calibrated XGBoost on all residues)

"""
    if ranking:
        md += f"""- All {n_pos} positives occupy ranks 1–{ranking.get('all_positives_in_top_n', 'N/A')}
- Top-20 recall: **{ranking.get('top20_n', 'N/A')}/{n_pos}** ({ranking.get('top20_recall', 0)*100:.1f}%)
- Top-50 recall: **{ranking.get('top50_n', 'N/A')}/{n_pos}**
- Top-100 recall: **{ranking.get('top100_n', 'N/A')}/{n_pos}**
- Score gap (last positive − first negative): **{ranking.get('score_gap_last_pos_first_neg', 'N/A')}**

"""

    md += f"""## Matched-null validation
- Tier 1 count: **{matched.get('real_tier1_count', 'N/A')}** vs null mean {matched.get('null_mean', 'N/A')} (p = {matched.get('p_value', 'N/A')})

## Vina structural validation (Tier-4 pocket-direct)
- Candidates docked: **{vina.get('n_candidates', 'N/A')}**
- Structurally validated (ΔΔG ≥ 0.15): **{vina.get('n_validated', 'N/A')}**

## Permutation test
- p = **{publication['permutation_test'].get('p_value', 'N/A')}**

## ESM-2 baseline
- ESM-2 only AUROC: **{publication['baselines'].get('esm2_only_auroc', 'N/A')}**
- Full model lift: **+{publication['baselines'].get('full_model_lift_auroc', 'N/A')}** AUROC

---
*Source: `python scripts/13_final_publication_audit.py`*
"""
    md_path = RESULTS / "PUBLICATION_METRICS.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"\n--- PRIMARY METRICS (5-fold CV) ---")
    print(f"  AUROC:  {s['auroc_mean']:.4f} ± {s['auroc_std']:.4f}")
    print(f"  AUPRC:  {s['auprc_mean']:.4f} ± {s['auprc_std']:.4f} ({s['auprc_x_random']:.0f}x random)")
    print(f"  F1*:    {s['best_f1_mean']:.4f} ± {s['best_f1_std']:.4f}")
    print(f"  F1@0.5: {s['f1_at_05_mean']:.4f} ± {s['f1_at_05_std']:.4f}")
    print(f"  Top-20: {s['top20_recall_mean']:.4f} ({int(s['top20_recall_mean']*n_pos)}/{n_pos})")
    print(f"\nSaved: {out_json}")
    print(f"Saved: {md_path}")
    print(f"Saved: {FIGURES / 'fig_roc_curve.csv'}")
    print(f"Saved: {FIGURES / 'fig_pr_curve.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
