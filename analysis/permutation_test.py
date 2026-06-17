"""Permutation test + bootstrap CIs for XGBoost hotspot model.
Reduced permutations (200) with fewer trees (100) for null distribution.
"""
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE / "analysis" / "results" / "hotspot_model"
sys.path.insert(0, str(BASE / "scripts"))
docking = __import__("04d_docking_features")

# ---- Load data (same as model sees) ----
df = docking.load_feature_data()
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

# mutation_sensitivity
codon_map = {"A":["GCT","GCC","GCA","GCG"],"C":["TGT","TGC"],"D":["GAT","GAC"],
    "E":["GAA","GAG"],"F":["TTT","TTC"],"G":["GGT","GGC","GGA","GGG"],
    "H":["CAT","CAC"],"I":["ATT","ATC","ATA"],"K":["AAA","AAG"],
    "L":["TTA","TTG","CTT","CTC","CTA","CTG"],"M":["ATG"],"N":["AAT","AAC"],
    "P":["CCT","CCC","CCA","CCG"],"Q":["CAA","CAG"],
    "R":["CGT","CGC","CGA","CGG","AGA","AGG"],
    "S":["TCT","TCC","TCA","TCG","AGT","AGC"],"T":["ACT","ACC","ACA","ACG"],
    "V":["GTT","GTC","GTA","GTG"],"W":["TGG"],"Y":["TAT","TAC"]}
all_aas = set("ACDEFGHIKLMNPQRSTVWY")
def compute_mut_sens(wt):
    codons = codon_map.get(wt, ["NNN"])
    n = 0
    for c in codons:
        for i in range(3):
            for nt in "ATCG":
                if nt == c[i]: continue
                mc = c[:i] + nt + c[i+1:]
                for a in all_aas:
                    if a != wt and any(mc == cc for cc in codon_map.get(a, [])):
                        n += 1
    return min(9, n) / 9.0
df["mutation_sensitivity"] = df["wt_aa"].apply(compute_mut_sens)

for pos in [21, 94, 95, 99, 103, 203]:
    df.loc[(df["gene"] == "inhA") & (df["residue_pos"] == pos), "is_hotspot"] = 1

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
             "plddt_score","plddt_environment","mutation_sensitivity"]
all_feat = [f for f in base_feat + new_feat if f in df.columns] + ["drug_proximity"]

df_model = df.dropna(subset=all_feat).copy()
y = df_model["is_hotspot"].values
X = df_model[all_feat].values

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from xgboost import XGBClassifier

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_and_eval(X_tr, y_tr, X_te, y_te, n_estimators=300):
    m = XGBClassifier(scale_pos_weight=10, max_depth=6, learning_rate=0.05,
                      n_estimators=n_estimators, subsample=0.8, colsample_bytree=0.8,
                      eval_metric="logloss", random_state=42)
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, p), average_precision_score(y_te, p)

# ---- Real CV (full model, 300 trees) ----
print("Real 5-fold CV...")
real_aucs, real_aps = [], []
all_preds, all_trues = np.zeros(len(df_model)), np.zeros(len(df_model))
for tr, te in skf.split(X, y):
    s = StandardScaler()
    X_tr = s.fit_transform(X[tr])
    X_te = s.transform(X[te])
    auc, ap = train_and_eval(X_tr, y[tr], X_te, y[te], n_estimators=300)
    real_aucs.append(auc); real_aps.append(ap)

    m = XGBClassifier(scale_pos_weight=10, max_depth=6, learning_rate=0.05,
                      n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                      eval_metric="logloss", random_state=42)
    m.fit(X_tr, y[tr])
    all_preds[te] = m.predict_proba(X_te)[:, 1]
    all_trues[te] = y[te]

real_auc = float(np.mean(real_aucs))
real_ap = float(np.mean(real_aps))
print(f"  AUROC: {real_auc:.4f} +/- {np.std(real_aucs):.4f}")
print(f"  AUPRC: {real_ap:.4f} +/- {np.std(real_aps):.4f}")

# ---- Bootstrap CIs on pooled predictions ----
print("\nBootstrap 95% CIs (1000 resamples)...")
np.random.seed(42)
boot_aucs, boot_aps = [], []
for _ in range(1000):
    idx = np.random.choice(len(all_preds), len(all_preds), replace=True)
    if y[idx].sum() < 2: continue
    boot_aucs.append(roc_auc_score(all_trues[idx], all_preds[idx]))
    boot_aps.append(average_precision_score(all_trues[idx], all_preds[idx]))

def ci95(arr):
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

auc_ci = ci95(boot_aucs)
ap_ci = ci95(boot_aps)
print(f"  AUROC: {real_auc:.4f} [{auc_ci[0]:.4f} - {auc_ci[1]:.4f}]")
print(f"  AUPRC: {real_ap:.4f} [{ap_ci[0]:.4f} - {ap_ci[1]:.4f}]")

# ---- Permutation test (200 shuffles, 100 trees for speed) ----
print("\nPermutation test (200 shuffles, 100 trees)...")
np.random.seed(42)
perm_aucs = []
for i in range(200):
    y_shuf = np.random.permutation(y)
    fa = []
    for tr, te in skf.split(X, y_shuf):
        s = StandardScaler()
        X_tr = s.fit_transform(X[tr])
        X_te = s.transform(X[te])
        auc, _ = train_and_eval(X_tr, y_shuf[tr], X_te, y_shuf[te], n_estimators=100)
        fa.append(auc)
    perm_aucs.append(float(np.mean(fa)))
    if (i+1) % 50 == 0:
        print(f"  {i+1}/200 done, max perm AUROC: {max(perm_aucs):.4f}")

perm_aucs = np.array(perm_aucs)
n_exceed = int((perm_aucs >= real_auc).sum())
p_value = float((n_exceed + 1) / (len(perm_aucs) + 1))
print(f"  Real AUROC: {real_auc:.4f}")
print(f"  Perm mean +/- std: {perm_aucs.mean():.4f} +/- {perm_aucs.std():.4f}")
print(f"  Perm max: {perm_aucs.max():.4f}")
print(f"  p-value: {p_value:.4f} ({n_exceed}/{len(perm_aucs)} exceeded)")
print(f"  Significant at p<0.001: {p_value < 0.001}")

# ---- Mutation sensitivity analysis ----
print("\nMutation sensitivity analysis...")
feat_no_ms = [f for f in all_feat if f != "mutation_sensitivity"]
auc_ms, auc_no_ms = [], []
for tr, te in skf.split(X, y):
    s = StandardScaler()
    auc1, _ = train_and_eval(s.fit_transform(df_model[all_feat].values[tr]), y[tr],
                              s.transform(df_model[all_feat].values[te]), y[te])
    auc2, _ = train_and_eval(s.fit_transform(df_model[feat_no_ms].values[tr]), y[tr],
                              s.transform(df_model[feat_no_ms].values[te]), y[te])
    auc_ms.append(auc1); auc_no_ms.append(auc2)

print(f"  With mutation_sensitivity:    {np.mean(auc_ms):.4f}")
print(f"  Without mutation_sensitivity: {np.mean(auc_no_ms):.4f}")

min_s, max_s = float(df_model["mutation_sensitivity"].min()), float(df_model["mutation_sensitivity"].max())
mean_s = float(df_model["mutation_sensitivity"].mean())
print(f"  mutation_sensitivity range: {min_s:.3f} - {max_s:.3f}, mean: {mean_s:.3f}")
print(f"  Feature has {df_model['mutation_sensitivity'].nunique():.0f} unique values across {len(df_model)} residues")

# ---- Results ----
results = {
    "n_samples": int(len(df_model)),
    "n_positives": int(y.sum()),
    "n_features": int(len(all_feat)),
    "features": list(all_feat),
    "real_cv": {
        "auroc_mean": real_auc,
        "auroc_std": float(np.std(real_aucs)),
        "auprc_mean": real_ap,
        "auprc_std": float(np.std(real_aps)),
        "per_fold_aurocs": [float(a) for a in real_aucs],
    },
    "bootstrap_95ci": {
        "auroc": [auc_ci[0], auc_ci[1]],
        "auprc": [ap_ci[0], ap_ci[1]],
    },
    "permutation_test": {
        "real_auroc": real_auc,
        "perm_mean": float(perm_aucs.mean()),
        "perm_max": float(perm_aucs.max()),
        "perm_std": float(perm_aucs.std()),
        "n_exceeded": n_exceed,
        "p_value": p_value,
    },
    "mutation_sensitivity": {
        "min": min_s, "max": max_s, "mean": mean_s,
        "unique_values": int(df_model["mutation_sensitivity"].nunique()),
        "with_feature_auroc": float(np.mean(auc_ms)),
        "without_feature_auroc": float(np.mean(auc_no_ms)),
        "delta": float(np.mean(auc_ms) - np.mean(auc_no_ms)),
    },
}

with open(OUTPUT_DIR / "permutation_test_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved!")
print("Done.")
