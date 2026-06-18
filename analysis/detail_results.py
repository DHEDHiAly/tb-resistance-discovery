"""Detailed mutation forecasting validation and CRyPTIC breakdown."""
import pandas as pd, json, os
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent

w = pd.read_csv(BASE / "analysis/results/forecasting/emergence_watchlist.csv")
cv = pd.read_csv(BASE / "analysis/results/forecasting/cryptic_validation_results.csv")
tv = pd.read_csv(BASE / "analysis/results/forecasting/cryptic_tiered_validation.csv")

print("=== KNOWN RESISTANCE MUTATIONS IN WATCHLIST ===")
known = w[w["is_known_resistance"]==1]
print(f"Total known resistance mutations in watchlist: {len(known)}")
for _,r in known.iterrows():
    print(f"  #{int(r['overall_rank']):<4d} {r['gene']:<6s} {r['mutation']:<10s} score={r['emergence_score']:.4f}  hotspot={r['hotspot_score']:.4f}")

print("\n=== NOVEL (NON-KNOWN) TOP 20 WATCHLIST ===")
novel = w[w["is_known_resistance"]!=1].head(20)
for i,(_,r) in enumerate(novel.iterrows()):
    print(f"  #{i+1:<3d} {r['gene']:<6s} {r['mutation']:<10s} score={r['emergence_score']:.4f}  hotspot={r['hotspot_score']:.4f}  "
          f"blosum={r['blosum62']}  charge_change={r['charge_change']}  transition={r['is_transition']}")

print("\n=== CRYPTIC VALIDATION CATEGORIES ===")
print(f"  Category A (Known WHO, observed): {len(cv[cv['category']=='A'])}")
print(f"  Category B (Novel, observed): {len(cv[cv['category']=='B'])}")
print(f"  Category C (Forecast-only): {len(cv[cv['category']=='C'])}")

print("\n=== NOVEL OBSERVED WITH PHENOTYPE DATA ===")
b = cv[cv['category']=='B']
with_pheno = b[b['n_phenotyped'] > 0]
print(f"  Novel with phenotype data: {len(with_pheno)}")
enriched = with_pheno[with_pheno['resistance_frac'] > 0.5]
print(f"  Novel enriched R>50%: {len(enriched)}")
for _,r in enriched.iterrows():
    print(f"    {r['gene']:<6s} {r['mutation']:<10s} carriers={r['n_carriers']:<4d}  R%={r['resistance_frac']:.0%}  OR={r['odds_ratio']:.1f}")

print("\n=== WATCHLIST STATISTICS ===")
print(f"  Total watchlist: {len(w)}")
print(f"  Unique genes: {w['gene'].nunique()}")
print(f"  Known resistance mutations: {w['is_known_resistance'].sum()}")
print(f"  Known hotspot residues: {w['is_known_hotspot'].sum()}")
print(f"  Score range: [{w['emergence_score'].min():.4f}, {w['emergence_score'].max():.4f}]")
print(f"  Score mean +/- std: {w['emergence_score'].mean():.4f} +/- {w['emergence_score'].std():.4f}")
