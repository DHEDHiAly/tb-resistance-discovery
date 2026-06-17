import pandas as pd, numpy as np

wl = pd.read_csv('analysis/results/forecasting/emergence_watchlist.csv')
cv = pd.read_csv('analysis/results/forecasting/cryptic_validation_results.csv')
fdr = pd.read_csv('analysis/results/forecasting/cryptic_fdr_analysis.csv')

m = wl.merge(cv, on=['gene','mutation'], how='left', suffixes=('','_cv'))
fdr_sig = set(zip(fdr.loc[fdr['significant_fdr']==True, 'gene'], fdr.loc[fdr['significant_fdr']==True, 'mutation']))

def classify(r):
    name = f"{r['gene']}_{r['mutation']}"
    cat = r.get('category','C')
    is_fdr = (r['gene'], r['mutation']) in fdr_sig
    ncar = r.get('n_carriers',0) or 0
    nphen = r.get('n_phenotyped',0) or 0
    rfrac = r.get('resistance_frac', -1)
    
    if cat == 'A':
        return None
    if cat == 'B':
        if is_fdr:
            return ('validated Tier 1', f"{name}: {ncar} carriers, {rfrac:.0%} R, FDR q<0.05")
        if name == 'pncA_Q10R':
            return ('uncertain/caveat', f"{name}: {ncar} carriers, NO PZA phenotype data (blind spot)")
        if rfrac is not None and rfrac >= 0 and rfrac < 0.3:
            return ('uncertain/caveat', f"{name}: {ncar} carriers, {rfrac:.0%} R — likely polymorphism")
        if ncar < 5:
            return ('uncertain/caveat', f"{name}: {ncar} carriers — low power")
        return ('uncertain/caveat', f"{name}: {ncar} carriers, {rfrac:.0%} R — not FDR-sig")
    if cat == 'C':
        return ('forecast-only', name)
    return None

seen = set()
rows = []
for _, r in m.iterrows():
    mut = f"{r['gene']}_{r['mutation']}"
    if mut in seen:
        continue
    seen.add(mut)
    result = classify(r)
    if result is not None:
        tier, detail = result
        rows.append({
            'rank': int(r['overall_rank']),
            'mutation': mut,
            'score': r['emergence_score'],
            'category': tier,
            'detail': detail
        })

df = pd.DataFrame(rows)
tier_order = {'validated Tier 1': 0, 'uncertain/caveat': 1, 'forecast-only': 2}
df['_sort'] = df['category'].map(tier_order)
df = df.sort_values(['_sort','rank']).drop('_sort', axis=1)

prev = None
for _, r in df.iterrows():
    if r['category'] != prev:
        print()
        print(f"## {r['category']}")
        print()
        prev = r['category']
    print(f"  {r['rank']:>4}  {r['mutation']:<18}  {r['score']:.4f}  {r['detail']}")

df.to_csv('analysis/results/forecasting/tiered_mutation_map.csv', index=False)
print()
print(f"Saved: analysis/results/forecasting/tiered_mutation_map.csv ({len(df)} rows)")
