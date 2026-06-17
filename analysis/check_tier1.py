"""Check Tier 1 CRyPTIC mutations for positive expansion."""
import pandas as pd

tv = pd.read_csv("analysis/results/forecasting/cryptic_tiered_validation.csv")
t1 = tv[tv["tier"] == 1]
print(f"Tier 1: {len(t1)}")
for _, r in t1.iterrows():
    print(f"  {r['gene']} {r['mutation']} — rank={r['rank']}, carriers={r['n_carriers']}, R%={r['resistance_frac']*100:.0f}%")

known = pd.read_csv("analysis/results/hotspot_model/ranked_predictions.csv")
hot = set()
for _, r in known[known["is_hotspot"] == 1].iterrows():
    hot.add((r["gene"], int(r["residue_pos"])))

print("\nOverlap with training positives:")
for _, r in t1.iterrows():
    pos = int(r["mutation"][1:-1])
    in_training = (r["gene"], pos) in hot
    print(f"  {r['gene']} {r['mutation']} — residue {pos} — in training: {in_training}")
