"""Export out-of-fold CV predictions for precision-recall curve plotting."""
from __future__ import annotations

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


def load_model_frame() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(OUTPUT_DIR / "residue_hotspot_data_with_docking.csv")

    for pkl, col in [
        ("sasa_data.pkl", "sasa_relative"),
        ("esm2_data.pkl", "esm2_intolerance"),
        ("contact_density_3d.pkl", "contact_density_3d"),
    ]:
        path = OUTPUT_DIR / pkl
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            df[col] = df.apply(
                lambda r, d=data: d.get((r["gene"], r["residue_pos"]), np.nan), axis=1
            )

    for pos in [21, 94, 95, 99, 103, 203]:
        df.loc[(df["gene"] == "inhA") & (df["residue_pos"] == pos), "is_hotspot"] = 1

    cryptic_new = [
        ("gyrA", 88), ("inhA", 194), ("eis", 59), ("inhA", 16), ("rpoB", 483),
    ]
    for gene, pos in cryptic_new:
        mask = (df["gene"] == gene) & (df["residue_pos"] == pos)
        df.loc[mask, "is_hotspot"] = 1

    dist_path = OUTPUT_DIR / "drug_distances.pkl"
    if dist_path.exists():
        with open(dist_path, "rb") as f:
            all_dists = pickle.load(f)
    else:
        all_dists = {}
    drug_dist = np.full(len(df), 100.0)
    for (gene, pos), dist in all_dists.items():
        drug_dist[(df["gene"] == gene) & (df["residue_pos"] == pos)] = dist
    df["drug_distance"] = drug_dist
    df["drug_proximity"] = 1.0 / (1.0 + df["drug_distance"] / 10.0)

    base_feat = [
        "inner_distance", "homoplasy_count", "homoplasy_alleles",
        "helix_propensity", "strand_propensity", "hydrophobicity",
        "volume", "charge", "hbond", "rel_position", "conservation_blosum",
        "contact_density_seq",
    ]
    new_feat = ["sasa_relative", "esm2_intolerance", "contact_density_3d"]
    features = [f for f in base_feat + new_feat if f in df.columns] + ["drug_proximity"]

    df_model = df.dropna(subset=features).copy()
    return df_model, features


def export_oof_predictions() -> Path:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score, roc_auc_score
    from xgboost import XGBClassifier

    df_model, features = load_model_frame()
    y = df_model["is_hotspot"].values.astype(int)
    X = df_model[features].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = XGBClassifier(
            scale_pos_weight=10,
            max_depth=6,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y[train_idx])
        oof[test_idx] = model.predict_proba(X_test)[:, 1]

    out_df = df_model[["gene", "residue_pos", "is_hotspot"]].copy()
    out_df["y_true"] = y
    out_df["y_score"] = oof

    csv_path = OUTPUT_DIR / "pr_curve_oof.csv"
    out_df.to_csv(csv_path, index=False)

    summary = {
        "n_samples": int(len(y)),
        "n_positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, oof)),
        "auprc": float(average_precision_score(y, oof)),
        "auprc_x_random": float(average_precision_score(y, oof) / y.mean()),
    }
    json_path = OUTPUT_DIR / "pr_curve_oof_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved {csv_path} ({len(out_df)} rows, AUPRC={summary['auprc']:.4f})")
    print(f"Saved {json_path}")
    return csv_path


if __name__ == "__main__":
    export_oof_predictions()
