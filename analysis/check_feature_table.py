import pandas as pd
df = pd.read_csv('analysis/results/hotspot_model/residue_hotspot_data.csv')
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print()
h = df['homoplasy_count']
print("homoplasy_count:", h.min(), "-", h.max(), "sum:", h.sum(), "nonzero:", (h>0).sum())
if 'n_genomes' in df.columns:
    print("n_genomes:", df['n_genomes'].iloc[0])
print("is_hotspot:", df['is_hotspot'].sum())
print()
for g in ['pncA','katG','rpoB','embB','rpsL']:
    gd = df[df['gene']==g]
    n = (gd['homoplasy_count']>0).sum()
    print("{}: {} residues, max={}".format(g, n, gd['homoplasy_count'].max()))
