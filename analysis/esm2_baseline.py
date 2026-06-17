"""ESM-2 baseline comparison: sequence-only vs. full structural model.
Runs both through identical 5-fold CV and measures AUROC/AUPRC lift.
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

# ---- Load data ----
df = docking.load_feature_data()
for pkl, col in [("sasa_data.pkl","sasa_relative"),("esm2_data.pkl","esm2_intolerance"),
                  ("contact_density_3d.pkl","contact_density_3d")]:
    path = OUTPUT_DIR / pkl
    if path.exists():
        with open(path,"rb") as f:
            data=pickle.load(f)
        df[col]=df.apply(lambda r:data.get((r["gene"],r["residue_pos"]),np.nan),axis=1)

plddt_path=OUTPUT_DIR/"plddt_data.pkl"
if plddt_path.exists():
    plddt_df=pd.read_pickle(plddt_path)
    for c in ["plddt_score_x","plddt_environment_x","plddt_score_y","plddt_environment_y"]:
        if c in df.columns: del df[c]
    df=df.merge(plddt_df,on=["gene","residue_pos"],how="left")

for pos in [21,94,95,99,103,203]:
    df.loc[(df["gene"]=="inhA")&(df["residue_pos"]==pos),"is_hotspot"]=1

dist_path=OUTPUT_DIR/"drug_distances.pkl"
if dist_path.exists():
    with open(dist_path,"rb") as f: all_dists=pickle.load(f)
else: all_dists={}
drug_dc=np.full(len(df),100.0)
for (gene,pos),dist in all_dists.items():
    drug_dc[(df["gene"]==gene)&(df["residue_pos"]==pos)]=dist
df["drug_distance"]=drug_dc
df["drug_proximity"]=1.0/(1.0+df["drug_distance"]/10.0)

base_feat=["inner_distance","homoplasy_count","homoplasy_alleles",
    "helix_propensity","strand_propensity","hydrophobicity",
    "volume","charge","hbond","rel_position","conservation_blosum","contact_density_seq"]
struct_feat=["sasa_relative","contact_density_3d","plddt_score","plddt_environment"]
full_feat=[f for f in base_feat+struct_feat if f in df.columns]+["drug_proximity"]

df_model=df.dropna(subset=full_feat).copy()
y=df_model["is_hotspot"].values
X_full=df_model[full_feat].values
X_esm2=df_model[["esm2_intolerance"]].values
X_stage0=df_model[[f for f in base_feat if f in df.columns]].values
X_stage1=df_model[[f for f in base_feat+struct_feat if f in df.columns]].values

print(f"Samples: {len(df_model)}, positives: {y.sum()} ({y.mean()*100:.2f}%)")
print(f"Baselines:")
print(f"  ESM-2 only      : 1 feature")
print(f"  Stage 0 (seq)   : {X_stage0.shape[1]} features (sequence)")
print(f"  Stage 1 (struct): {X_stage1.shape[1]} features (seq+struct)")
print(f"  Full (Stage 3)  : {X_full.shape[1]} features (+drug_proximity)")
print(f"  ESM-2 NaN count : {pd.isna(df_model['esm2_intolerance']).sum()}")
print(f"  ESM-2 range     : {df_model['esm2_intolerance'].min():.4f} - {df_model['esm2_intolerance'].max():.4f}")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

def cv_auc_ap(X, y, model_fn):
    aucs, aps = [], []
    for tr, te in skf.split(X, y):
        s=StandardScaler()
        m=model_fn()
        if isinstance(m, XGBClassifier):
            m.fit(s.fit_transform(X[tr]),y[tr])
            p=m.predict_proba(s.transform(X[te]))[:,1]
        else:
            m.fit(s.fit_transform(X[tr]),y[tr])
            p=m.predict_proba(s.transform(X[te]))[:,1]
        aucs.append(roc_auc_score(y[te],p))
        aps.append(average_precision_score(y[te],p))
    return np.mean(aucs), np.mean(aps), np.std(aucs), np.std(aps)

def xgb():
    return XGBClassifier(scale_pos_weight=10,max_depth=6,learning_rate=0.05,
                         n_estimators=300,subsample=0.8,colsample_bytree=0.8,
                         eval_metric="logloss",random_state=42)
def lr():
    return LogisticRegression(C=1.0,class_weight="balanced",max_iter=1000,random_state=42)

print("\n--- 5-fold CV comparison ---")
results={}
for name, X_data, model_fn in [
    ("ESM-2 only (XGB)", X_esm2, xgb),
    ("ESM-2 only (LR)", X_esm2, lr),
    ("Stage 0 (XGB)", X_stage0, xgb),
    ("Stage 1 (XGB)", X_stage1, xgb),
    ("Full Stage 3 (XGB)", X_full, xgb),
]:
    auc, ap, auc_std, ap_std = cv_auc_ap(X_data, y, model_fn)
    results[name] = {"auroc": round(float(auc),4), "auprc": round(float(ap),4),
                     "auroc_std": round(float(auc_std),4), "auprc_std": round(float(ap_std),4)}
    print(f"  {name:<30s} AUROC={auc:.4f}+-{auc_std:.4f}  AUPRC={ap:.4f}+-{ap_std:.4f}")

# --- Summary ---
print("\n--- Lift over ESM-2 baseline ---")
esm2_xgb_auc = results.get("ESM-2 only (XGB)",{}).get("auroc",0)
full_auc = results.get("Full Stage 3 (XGB)",{}).get("auroc",0)
print(f"  ESM-2 only AUROC:        {esm2_xgb_auc:.4f}")
print(f"  Full model AUROC:        {full_auc:.4f}")
print(f"  Absolute lift:           {full_auc - esm2_xgb_auc:+.4f}")
print(f"  Relative lift:           {((full_auc - esm2_xgb_auc)/max(esm2_xgb_auc,0.001))*100:+.1f}%")

with open(OUTPUT_DIR/"esm2_baseline_results.json","w") as f:
    json.dump(results,f,indent=2)
print(f"\nResults saved.")
