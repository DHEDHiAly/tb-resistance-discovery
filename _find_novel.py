"""
Find truly novel forecast-only mutations from the validation data.
Target: Tier 4, rank 20-100, 0 carriers, outside standard RRDR/QRDR.
"""
import pandas as pd

val = pd.read_csv('analysis/results/forecasting/cryptic_tiered_validation.csv')

# Filter: Tier 4 (forecast-only), rank > 20
mask = (val['tier'] == 4) & (val['rank'] > 20) & (val['n_carriers'] <= 1)
candidates = val[mask].sort_values('rank')

print(f"Forecast-only candidates (Tier 4, rank>20, carriers<=1): {len(candidates)}")
print()
print(f"{'Rank':<6} {'Gene':<8} {'Mutation':<18} {'Carriers':<10} {'Phenotyped':<12} {'Resist%':<10} {'Score':<8}")
print("-" * 75)

best_candidates = []
for _, r in candidates.iterrows():
    resist_frac = f"{r['resistance_frac']:.0%}" if pd.notna(r['resistance_frac']) else "N/A"
    print(f"{r['rank']:<6} {r['gene']:<8} {r['mutation']:<18} {r['n_carriers']:<10} {r['n_phenotyped']:<12} {resist_frac:<10} {r['emergence_score']:<8.3f}")
    best_candidates.append(r)

# Focus on candidates with 0 carriers (truly unseen)
zero = candidates[candidates['n_carriers'] == 0]
print(f"\nTruly unseen (0 carriers): {len(zero)}")
print(f"\nTop 15 truly unseen candidates:")
print(f"{'Rank':<6} {'Gene':<8} {'Mutation':<18} {'Score':<8}")
for _, r in zero.head(15).iterrows():
    print(f"{r['rank']:<6} {r['gene']:<8} {r['mutation']:<18} {r['emergence_score']:<8.3f}")

# Gene distribution
print("\n\nBy gene for Tier 4, rank>20:")
print(val[(val['tier']==4) & (val['rank']>20)]['gene'].value_counts().to_string())

# Outside standard resistance regions
# rpoB RRDR: ~426-452 (the 81bp region)
# gyrA QRDR: ~74-114
# For completely novel candidates, focus on non-rpoB-gyrA
other_genes = val[(val['tier']==4) & (val['rank']>20) & (val['n_carriers']<=1) & ~val['gene'].isin(['rpoB','gyrA','katG'])]
print(f"\n\nNon-rpoB/gyrA/katG candidates (Tier 4, rank>20): {len(other_genes)}")
print(f"{'Rank':<6} {'Gene':<8} {'Mutation':<18} {'Carriers':<10} {'Score':<8}")
for _, r in other_genes.head(20).iterrows():
    print(f"{r['rank']:<6} {r['gene']:<8} {r['mutation']:<18} {r['n_carriers']:<10} {r['emergence_score']:<8.3f}")
