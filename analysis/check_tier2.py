"""Check how many new positive residues Tier 1-2 would add."""
import pandas as pd

tv = pd.read_csv("analysis/results/forecasting/cryptic_tiered_validation.csv")
known = pd.read_csv("analysis/results/hotspot_model/ranked_predictions.csv")
hot_residues = set()
for _, r in known[known["is_hotspot"] == 1].iterrows():
    hot_residues.add((r["gene"], int(r["residue_pos"])))

print(f"Current training positives: {len(hot_residues)} residues")

for tier_label in [1, 2, 3]:
    t = tv[tv["tier"] == tier_label]
    new_residues = set()
    for _, r in t.iterrows():
        pos = int(r["mutation"][1:-1])
        if (r["gene"], pos) not in hot_residues:
            new_residues.add((r["gene"], pos, r["mutation"]))
    print(f"\nTier {tier_label} ({len(t)} mutations):")
    print(f"  Mutations at new residues: {len(new_residues)}")
    for g, p, m in sorted(new_residues):
        print(f"    {g} {m} (residue {p})")

# Total if we include all Tier 1-3 new residues
all_new = set()
for tier_label in [1, 2, 3]:
    t = tv[tv["tier"] == tier_label]
    for _, r in t.iterrows():
        pos = int(r["mutation"][1:-1])
        if (r["gene"], pos) not in hot_residues:
            all_new.add((r["gene"], pos))
print(f"\nTotal new residues from Tier 1-3: {len(all_new)}")
print(f"Total if added: {len(hot_residues)} + {len(all_new)} = {len(hot_residues) + len(all_new)}")
for g, p in sorted(all_new):
    print(f"  {g} residue {p}")
