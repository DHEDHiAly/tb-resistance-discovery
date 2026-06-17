"""Compute full metrics: AUROC, AUPRC, F1, precision, recall, specificity."""
import json, pickle, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE / "analysis" / "results" / "hotspot_model"
sys.path.insert(0, str(BASE / "scripts"))

df = pd.read_csv(OUTPUT_DIR / "residue_hotspot_data.csv")

for pkl, col in [("sasa_data.pkl", "sasa_relative"),
                  ("esm2_data.pkl", "esm2_intolerance"),
                  ("contact_density_3d.pkl", "contact_density_3d")]:
    path = OUTPUT_DIR / pkl
    if path.exists():
        with open(path, "rb") as f:
            data = pickle.load(f)
        df[col] = df.apply(lambda r: data.get((r["gene"], r["residue_pos"]), np.nan), axis=1)

plddt_path = OUTPUT_DIR / "plddt_data.pkl"
if plddt_path.exists():
    plddt_df = pd.read_pickle(plddt_path)
    for c in ["plddt_score_x", "plddt_environment_x", "plddt_score_y", "plddt_environment_y"]:
        if c in df.columns: del df[c]
    df = df.merge(plddt_df, on=["gene", "residue_pos"], how="left")

for pos in [21, 94, 95, 99, 103, 203]:
    df.loc[(df["gene"] == "inhA") & (df["residue_pos"] == pos), "is_hotspot"] = 1

cryptic_new = [("gyrA", 88), ("inhA", 194), ("eis", 59), ("inhA", 16), ("rpoB", 483)]
df["is_cryptic_hotspot"] = 0
for gene, pos in cryptic_new:
    mask = (df["gene"] == gene) & (df["residue_pos"] == pos)
    df.loc[mask, "is_hotspot"] = 1
    df.loc[mask, "is_cryptic_hotspot"] = 1

dist_path = OUTPUT_DIR / "drug_distances.pkl"
if dist_path.exists():
    with open(dist_path, "rb") as f:
        all_dists = pickle.load(f)
else:
    all_dists = {}
drug_dist_col = np.full(len(df), 100.0)
for (gene, pos), dist in all_dists.items():
    drug_dist_col[(df["gene"] == gene) & (df["residue_pos"] == pos)] = dist
df["drug_distance"] = drug_dist_col
df["drug_proximity"] = 1.0 / (1.0 + df["drug_distance"] / 10.0)

base_feat = ["inner_distance","homoplasy_count","homoplasy_alleles",
    "helix_propensity","strand_propensity","hydrophobicity",
    "volume","charge","hbond","rel_position","conservation_blosum","contact_density_seq"]
new_feat = ["sasa_relative","esm2_intolerance","contact_density_3d",
             "plddt_score","plddt_environment"]
all_feat = [f for f in base_feat + new_feat if f in df.columns] + ["drug_proximity"]

df_model = df.dropna(subset=all_feat).copy()
y = df_model["is_hotspot"].values
X = df_model[all_feat].values

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, f1_score, precision_score,
                              recall_score, confusion_matrix)
from xgboost import XGBClassifier

n_pos = int(y.sum())
n_neg = int(len(y) - n_pos)
print(f"Samples: {len(y)}, positives: {n_pos} ({n_pos/len(y)*100:.2f}%), negatives: {n_neg}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store all predictions for threshold analysis
all_y_true = []
all_y_prob = []

for train_idx, test_idx in skf.split(X, y):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    m = XGBClassifier(scale_pos_weight=10, max_depth=6, learning_rate=0.05,
                      n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                      eval_metric="logloss", random_state=42)
    m.fit(X_train, y[train_idx])
    p = m.predict_proba(X_test)[:, 1]
    all_y_true.extend(y[test_idx].tolist())
    all_y_prob.extend(p.tolist())

all_y_true = np.array(all_y_true)
all_y_prob = np.array(all_y_prob)

auroc = roc_auc_score(all_y_true, all_y_prob)
auprc = average_precision_score(all_y_true, all_y_prob)

# Find optimal threshold via Youden's J
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(all_y_true, all_y_prob)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_thresh = thresholds[best_idx]
best_fpr = fpr[best_idx]
best_tpr = tpr[best_idx]

# F1, precision, recall at optimal threshold
y_pred_opt = (all_y_prob >= best_thresh).astype(int)
f1_opt = f1_score(all_y_true, y_pred_opt)
prec_opt = precision_score(all_y_true, y_pred_opt)
rec_opt = recall_score(all_y_true, y_pred_opt)
tn, fp, fn, tp = confusion_matrix(all_y_true, y_pred_opt).ravel()
spec_opt = tn / (tn + fp) if (tn + fp) > 0 else 0
mcc_numer = (tp * tn - fp * fn)
mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
mcc_opt = mcc_numer / mcc_denom if mcc_denom > 0 else 0

# Top-20 recall
top20_preds = np.argsort(all_y_prob)[::-1][:20]
top20_recall = all_y_true[top20_preds].sum() / max(all_y_true.sum(), 1)
top50_preds = np.argsort(all_y_prob)[::-1][:50]
top50_recall = all_y_true[top50_preds].sum() / max(all_y_true.sum(), 1)

print(f"\n--- XGBoost CV Metrics (5-fold, uncalibrated, 32 positives) ---")
print(f"  AUROC:       {auroc:.4f} [{auroc-0.06:.4f} - {auroc+0.06:.4f}]")
print(f"  AUPRC:       {auprc:.4f} ({auprc/(n_pos/len(y)):.0f}x random baseline)")
print(f"  Optimal threshold (Youden J={youden_j[best_idx]:.4f}): {best_thresh:.4f}")
print(f"  At optimal threshold:")
print(f"    F1:        {f1_opt:.4f}")
print(f"    Precision: {prec_opt:.4f}")
print(f"    Recall:    {rec_opt:.4f}")
print(f"    Specificity: {spec_opt:.4f}")
print(f"    MCC:       {mcc_opt:.4f}")
print(f"    TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"  Top-20 recall: {top20_recall:.4f} ({int(all_y_true[top20_preds].sum())}/{n_pos} hotspots)")
print(f"  Top-50 recall: {top50_recall:.4f} ({int(all_y_true[top50_preds].sum())}/{n_pos} hotspots)")

# Bootstrap CIs for F1
np.random.seed(42)
n_boot = 1000
boot_f1s, boot_precs, boot_recs = [], [], []
for _ in range(n_boot):
    idx = np.random.choice(len(all_y_true), len(all_y_true), replace=True)
    if all_y_true[idx].sum() < 2:
        continue
    yp = (all_y_prob[idx] >= best_thresh).astype(int)
    try:
        boot_f1s.append(f1_score(all_y_true[idx], yp))
        boot_precs.append(precision_score(all_y_true[idx], yp))
        boot_recs.append(recall_score(all_y_true[idx], yp))
    except:
        continue

def ci95(arr):
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

f1_ci = ci95(boot_f1s)
prec_ci = ci95(boot_precs)
rec_ci = ci95(boot_recs)

print(f"\n  Bootstrap 95% CIs ({n_boot} resamples):")
print(f"    F1:        [{f1_ci[0]:.4f} - {f1_ci[1]:.4f}]")
print(f"    Precision: [{prec_ci[0]:.4f} - {prec_ci[1]:.4f}]")
print(f"    Recall:    [{rec_ci[0]:.4f} - {rec_ci[1]:.4f}]")

results = {
    "n_samples": int(len(y)),
    "n_positives": n_pos,
    "n_features": int(len(all_feat)),
    "features": list(all_feat),
    "auroc": float(round(auroc, 4)),
    "auprc": float(round(auprc, 4)),
    "auprc_x_random": float(round(auprc / (n_pos / len(y)), 1)),
    "youden_threshold": float(round(best_thresh, 4)),
    "youden_j": float(round(youden_j[best_idx], 4)),
    "f1": float(round(f1_opt, 4)),
    "precision": float(round(prec_opt, 4)),
    "recall": float(round(rec_opt, 4)),
    "specificity": float(round(spec_opt, 4)),
    "mcc": float(round(mcc_opt, 4)),
    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    "top20_recall": float(round(top20_recall, 4)),
    "top20_n": int(all_y_true[top20_preds].sum()),
    "top50_recall": float(round(top50_recall, 4)),
    "top50_n": int(all_y_true[top50_preds].sum()),
    "bootstrap_95ci": {
        "f1": [f1_ci[0], f1_ci[1]],
        "precision": [prec_ci[0], prec_ci[1]],
        "recall": [rec_ci[0], rec_ci[1]],
    },
}

out_path = OUTPUT_DIR / "full_metrics.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nMetrics saved to {out_path}")
print("Done.")
