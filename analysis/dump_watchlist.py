import pandas as pd, numpy as np

wl = pd.read_csv('analysis/results/forecasting/emergence_watchlist.csv')
cv = pd.read_csv('analysis/results/forecasting/cryptic_validation_results.csv')
fdr = pd.read_csv('analysis/results/forecasting/cryptic_fdr_analysis.csv')

m = wl.merge(cv, on=['gene','mutation'], how='left', suffixes=('','_cv'))
for c in ['rank_cv','emergence_score_cv']:
    if c in m.columns: del m[c]

fdr_sig = set(zip(fdr['gene'], fdr['mutation']))
m['fdr_significant'] = m.apply(lambda r: (r['gene'],r['mutation']) in fdr_sig, axis=1)

m['tier'] = 4
m.loc[m['category']=='A','tier'] = 0
m.loc[m['category']=='B','tier'] = 1
m.loc[m['category']=='C','tier'] = 4
m.loc[(m['tier']==1)&(m['fdr_significant']==False),'tier'] = 2
m.loc[(m['tier']==1)&(m['n_phenotyped'].isna()|(m['n_phenotyped']==0)),'tier'] = 3

out = m[['overall_rank','gene','mutation','emergence_score',
         'n_carriers','n_phenotyped','resistant','susceptible','resistance_frac',
         'tier','category','fdr_significant','is_known_who']].copy()
out['resistance_frac'] = out['resistance_frac'].fillna(-1)

# Tier labels
tier_labels = {0:'WHO known',1:'FDR-sig novel',2:'Observed enriched',3:'Observed no pheno',4:'Forecast only'}
status_map = {0:'WHO KNOWN',1:'FDR-sig',2:'Enriched',3:'No pheno',4:'Forecast'}

rows_out = []
for _, r in out.iterrows():
    frac = f'{r["resistance_frac"]:.0%}' if r['resistance_frac']>=0 else 'N/A'
    ncar = int(r['n_carriers']) if pd.notna(r['n_carriers']) else 0
    nphen = int(r['n_phenotyped']) if pd.notna(r['n_phenotyped']) else 0
    st = status_map.get(r['tier'], '?')
    rows_out.append({
        'rank': int(r['overall_rank']),
        'mutation': f"{r['gene']}_{r['mutation']}",
        'score': r['emergence_score'],
        'carriers': ncar, 'phenotyped': nphen,
        'r_pct': frac, 'tier': r['tier'],
        'status': st, 'fdr': r['fdr_significant'],
    })

# Print ALL 330
print('=== ALL 330 WATCHLIST MUTATIONS ===')
print(f"{'Rank':>5} {'Mutation':<18} {'Score':<8} {'Car':<5} {'Ph':<4} {'R%':<5} {'Tier':<5} Status")
print('-'*85)
for r in rows_out:
    print(f"{r['rank']:>5} {r['mutation']:<18} {r['score']:<8.4f} {r['carriers']:<5} {r['phenotyped']:<4} {r['r_pct']:<5} {r['tier']:<5} {r['status']}")

# Strongest by tier
print('\n\n=== STRONGEST TO PRESENT ===')
for t in [1,2,3,0,4]:
    sub = [r for r in rows_out if r['tier']==t][:20]
    if not sub: continue
    print(f'\n--- Tier {t}: {tier_labels[t]} ---')
    print(f"{'Rank':>5} {'Mutation':<18} {'Score':<8} {'Car':<5} {'R%':<5} Note")
    for r in sub:
        notes = {'WHO known':'WHO catalog','FDR-sig novel':'FDR q<0.05',
                 'Observed enriched':'R>50% (low power)','Observed no pheno':'no phenotype data',
                 'Forecast only':'not yet observed'}
        print(f"{r['rank']:>5} {r['mutation']:<18} {r['score']:<8.4f} {r['carriers']:<5} {r['r_pct']:<5} {notes[tier_labels[r['tier']]]}")

# Uncertain
print('\n\n=== UNCERTAIN / MIXED SIGNAL ===')
uncert2 = [r for r in rows_out if r['tier']==2 and r['phenotyped']>=5]
if uncert2:
    print(f'\n--- Tier 2 (enriched R>50% but NOT FDR-sig, n_phenotyped>=5) ---')
    for r in uncert2:
        print(f"  Rank {r['rank']:>4} {r['mutation']:<18} Score={r['score']:.4f} Car={r['carriers']} R%={r['r_pct']}")

uncert3 = [r for r in rows_out if r['tier']==3 and r['carriers']>=10]
if uncert3:
    print(f'\n--- Tier 3 (observed, no phenotype, carriers>=10) ---')
    for r in uncert3:
        print(f"  Rank {r['rank']:>4} {r['mutation']:<18} Score={r['score']:.4f} Car={r['carriers']} [no phenotype data]")

# Summary
n_tiers = {t:len([r for r in rows_out if r['tier']==t]) for t in range(5)}
print(f"\n\nSummary: {len(rows_out)} total | "
      f"Tier0(WHO)={n_tiers[0]} | Tier1(FDR)={n_tiers[1]} | "
      f"Tier2(Enriched)={n_tiers[2]} | Tier3(NoPheno)={n_tiers[3]} | "
      f"Tier4(Forecast)={n_tiers[4]}")
