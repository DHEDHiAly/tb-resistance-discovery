"""Get enriched novel mutations with correct column names."""
import pandas as pd
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent

cv = pd.read_csv(BASE / "analysis/results/forecasting/cryptic_validation_results.csv")
b = cv[cv['category']=='B']
with_pheno = b[b['n_phenotyped'] > 0]
enriched = with_pheno[with_pheno['resistance_frac'] > 0.5]
print(f"Novel with phenotype: {len(with_pheno)}, enriched R>50%: {len(enriched)}")
print(f"\nEnriched novel mutations:")
for _,r in enriched.iterrows():
    or_val = r.get('odds_ratio', r.get('OR', 'N/A'))
    print(f"  {r['gene']:<6s} {r['mutation']:<10s} carriers={r['n_carriers']:<4d}  R%={r['resistance_frac']:.0%}  OR={or_val}")

# Check for traits enrichment phenotype
es = enriched[enriched['n_carriers'] > 5]
print(f"\nEnriched with >5 carriers: {len(es)}")
for _,r in es.iterrows():
    print(f"  {r['gene']:<6s} {r['mutation']:<10s} carriers={r['n_carriers']:<4d}  R%={r['resistance_frac']:.0%}")

# CRyPTIC FDR summary from tiered validation
tv = pd.read_csv(BASE / "analysis/results/forecasting/cryptic_tiered_validation.csv")
t1 = tv[tv['tier']==1]
print(f"\nFDR summary: {len(t1)} Tier 1 mutations")
pvals = t1['pvalue_fdr'].values
print(f"  FDR p range: [{pvals.min():.2e}, {pvals.max():.2e}]")
print(f"  FDR p < 1e-15: {(pvals < 1e-15).sum()}")
print(f"  FDR p < 1e-10: {(pvals < 1e-10).sum()}")
print(f"  FDR p < 1e-5:  {(pvals < 1e-5).sum()}")
print(f"  FDR p < 0.01:  {(pvals < 0.01).sum()}")
print(f"  FDR p < 0.05:  {(pvals < 0.05).sum()}")
