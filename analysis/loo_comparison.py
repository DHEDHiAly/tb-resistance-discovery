"""Compare LR vs simpler XGBoost (depth=3) for LOO generalization."""
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE / "analysis" / "results" / "forecasting"
HOTSPOT_DIR = BASE / "analysis" / "results" / "hotspot_model"
sys.path.insert(0, str(BASE / "scripts"))
docking = __import__("04d_docking_features")

# ---- Load data ----
df = docking.load_feature_data()
for pkl, col in [("sasa_data.pkl","sasa_relative"),("esm2_data.pkl","esm2_intolerance"),
                  ("contact_density_3d.pkl","contact_density_3d")]:
    path = HOTSPOT_DIR / pkl
    if path.exists():
        with open(path,"rb") as f:
            data=pickle.load(f)
        df[col]=df.apply(lambda r:data.get((r["gene"],r["residue_pos"]),np.nan),axis=1)

plddt_path=HOTSPOT_DIR/"plddt_data.pkl"
if plddt_path.exists():
    plddt_df=pd.read_pickle(plddt_path)
    for c in ["plddt_score_x","plddt_environment_x","plddt_score_y","plddt_environment_y"]:
        if c in df.columns: del df[c]
    df=df.merge(plddt_df,on=["gene","residue_pos"],how="left")

for pos in [21,94,95,99,103,203]:
    df.loc[(df["gene"]=="inhA")&(df["residue_pos"]==pos),"is_hotspot"]=1

dist_path=HOTSPOT_DIR/"drug_distances.pkl"
if dist_path.exists():
    with open(dist_path,"rb") as f: all_dists=pickle.load(f)
else: all_dists={}
drug_dc=np.full(len(df),100.0)
for (gene,pos),dist in all_dists.items():
    drug_dc[(df["gene"]==gene)&(df["residue_pos"]==pos)]=dist
df["drug_distance"]=drug_dc
df["drug_proximity"]=1.0/(1.0+df["drug_distance"]/10.0)

feat=["inner_distance","homoplasy_count","homoplasy_alleles",
    "helix_propensity","strand_propensity","hydrophobicity",
    "volume","charge","hbond","rel_position","conservation_blosum","contact_density_seq",
    "sasa_relative","contact_density_3d","plddt_score","plddt_environment",
    "drug_proximity"]
feat=[f for f in feat if f in df.columns]

df_model=df.dropna(subset=feat).copy()
print(f"Model samples: {len(df_model)}, positives: {df_model['is_hotspot'].sum()}")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# LOO by gene
genes_with_hotspots = [g for g in sorted(df_model["gene"].unique())
                        if df_model[df_model["gene"]==g]["is_hotspot"].sum() >= 2]

print(f"\nLOO comparison ({len(genes_with_hotspots)} genes):")
print(f"{'Gene':<8} {'LR AUROC':<10} {'XGB_d3 AUROC':<14} {'LR Top20':<10} {'XGB_d3 Top20':<12} {'n_pos':<6}")
print("-"*60)

loo_results = {}
for held_out in genes_with_hotspots:
    train=df_model[df_model["gene"]!=held_out].copy()
    test=df_model[df_model["gene"]==held_out].copy()
    y_train,y_test=train["is_hotspot"].values,test["is_hotspot"].values
    X_train,X_test=train[feat].values,test[feat].values

    s=StandardScaler()
    Xtr_s=s.fit_transform(X_train)
    Xte_s=s.transform(X_test)

    lr=LogisticRegression(C=10.0,class_weight="balanced",max_iter=1000,random_state=42)
    lr.fit(Xtr_s,y_train)
    lr_p=lr.predict_proba(Xte_s)[:,1]

    xgb=XGBClassifier(scale_pos_weight=10,max_depth=3,learning_rate=0.05,
                      n_estimators=100,subsample=0.8,colsample_bytree=0.8,
                      eval_metric="logloss",random_state=42)
    xgb.fit(Xtr_s,y_train)
    xgb_p=xgb.predict_proba(Xte_s)[:,1]

    lr_auc=roc_auc_score(y_test,lr_p)
    xgb_auc=roc_auc_score(y_test,xgb_p)

    n_top20_lr=y_test[np.argsort(lr_p)[::-1][:20]].sum()
    n_top20_xgb=y_test[np.argsort(xgb_p)[::-1][:20]].sum()

    loo_results[held_out]={
        "lr_auroc":round(float(lr_auc),4),"xgb_depth3_auroc":round(float(xgb_auc),4),
        "lr_top20_hits":int(n_top20_lr),"xgb_depth3_top20_hits":int(n_top20_xgb),
        "n_positives":int(y_test.sum()),
    }
    print(f"{held_out:<8} {lr_auc:.4f}    {xgb_auc:.4f}        {n_top20_lr:<10} {n_top20_xgb:<12} {y_test.sum():<6}")

# Summary
print("\nSummary:")
lr_aucs=[v["lr_auroc"] for v in loo_results.values()]
xgb_aucs=[v["xgb_depth3_auroc"] for v in loo_results.values()]
print(f"  Mean LR AUROC:    {np.mean(lr_aucs):.4f}")
print(f"  Mean XGB(d=3) AUROC: {np.mean(xgb_aucs):.4f}")
print(f"  LR wins:  {(np.array(lr_aucs)>np.array(xgb_aucs)).sum()} / {len(lr_aucs)}")
print(f"  XGB wins: {(np.array(xgb_aucs)>np.array(lr_aucs)).sum()} / {len(xgb_aucs)}")

with open(OUTPUT_DIR/"loo_comparison_results.json","w") as f:
    json.dump(loo_results,f,indent=2)
print("\nSaved.")
