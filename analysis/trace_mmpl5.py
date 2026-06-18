"""Trace mmpL5 through pipeline stages to find where it was dropped."""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

# Check final output
rp = pd.read_csv(BASE / "analysis/results/hotspot_model/ranked_predictions.csv")
print(f"Final ranked_predictions.csv: {len(rp)} rows, {rp['gene'].nunique()} genes")
print(f"Genes: {sorted(rp['gene'].unique())}")
print(f"mmpL5 present: {(rp['gene']=='mmpL5').sum()} rows")

# Check Stage 1 feature data
st1 = pd.read_csv(BASE / "analysis/results/hotspot_model/residue_hotspot_data.csv")
print(f"\nresidue_hotspot_data.csv: {len(st1)} rows, {st1['gene'].nunique()} genes")
print(f"Genes: {sorted(st1['gene'].unique())}")
print(f"mmpL5 present: {(st1['gene']=='mmpL5').sum()} rows")

if (st1['gene']=='mmpL5').sum() > 0:
    m = st1[st1['gene']=='mmpL5']
    print(f"\nmmpL5 columns with any NaN:")
    for c in m.columns:
        if m[c].isna().sum() > 0:
            print(f"  {c}: {m[c].isna().sum()}/{len(m)} NaN ({m[c].isna().mean()*100:.0f}%)")

# If not in stage1, check earlier: 04b base data
try:
    hb = __import__("04b_hotspot_model")
    base = hb.load_feature_data()
    print(f"\n04b base data: {len(base)} rows, {base['gene'].nunique()} genes")
    print(f"mmpL5 present: {(base['gene']=='mmpL5').sum()} rows")
except:
    print("\nCould not load 04b data")
