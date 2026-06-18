"""Check why mmpL5 was dropped from the model."""
import pandas as pd
import numpy as np

df = pd.read_csv("analysis/results/hotspot_model/residue_hotspot_data.csv")
m = df[df["gene"] == "mmpL5"]
print(f"mmpL5 rows: {len(m)}")
print(f"mmpL5 is_hotspot: {m['is_hotspot'].sum()} positives")
print()

# Check for NaN in critical features
critical = ["inner_distance", "homoplasy_count", "plddt_score", "sasa_relative",
            "drug_proximity", "volume", "charge", "conservation_blosum"]
print("NaN counts for critical features:")
for c in critical:
    if c in m.columns:
        n = m[c].isna().sum()
        if n > 0:
            print(f"  {c}: {n}/{len(m)} NaN")
print()

# Check if there's an AlphaFold PDB for mmpL5
import os
pdb_dir = "data/pdb/alphafold"
pdb_files = [f for f in os.listdir(pdb_dir) if "mmpL5" in f or "mmpL5" in f.lower()]
print(f"mmpL5 AlphaFold PDBs: {pdb_files}")

# Check what features are actually in the ranked predictions
rp = pd.read_csv("analysis/results/hotspot_model/ranked_predictions.csv")
print(f"\nmmpL5 in ranked predictions: {(rp['gene']=='mmpL5').sum()} rows")
