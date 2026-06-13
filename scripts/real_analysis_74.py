import gzip, csv, re, os, time
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

t0 = time.time()
os.makedirs('analysis/results/figures', exist_ok=True)

# ── 1. Load phenotypes ──
pheno = pd.read_csv('analysis/results/phenotype_100.csv')
print(f"P: {len(pheno)} R={int(pheno['is_resistant'].sum())} S={len(pheno)-int(pheno['is_resistant'].sum())}")

p_dict = dict(zip(pheno['sample'], pheno['is_resistant']))
samples = list(p_dict.keys())
n_res = sum(p_dict.values())
n_sus = len(p_dict) - n_res

# ── 2. Parse VCF header ──
with gzip.open('data/demo/drprg_sparse.vcf.gz', 'rt') as f:
    for line in f:
        if line.startswith('#CHROM'):
            vcf_samples = line.strip().split('\t')[9:]
            break

s_idx = {s: i for i, s in enumerate(vcf_samples) if s in p_dict}
filtered = [s for s in vcf_samples if s in p_dict]
print(f"Samples: {len(filtered)}/{len(p_dict)} matched")

# ── 3. Parse GFF ──
genes = []
with open('reference/H37Rv.gff') as f:
    for line in f:
        if line.startswith('#') or line.strip() == '':
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9 or parts[2] != 'gene':
            continue
        attr = dict(re.findall(r'([\w-]+)=([^;\n]+)', parts[8]))
        if 'locus_tag' in attr:
            genes.append({
                'start': int(parts[3]), 'end': int(parts[4]),
                'gene_id': attr['locus_tag'],
                'name': attr.get('gene', attr.get('Name', attr['locus_tag']))
            })

KNOWN = {'Rv0005','Rv0006','Rv0667','Rv0668','Rv1908c','Rv2043c','Rv3795','Rv3794',
         'Rv1473','Rv0682','Rv1694','Rv2416c','Rv1258c','Rv0678','Rv2680'}

def find_gene(pos):
    for g in genes:
        if g['start'] <= pos <= g['end']:
            return g
    return None

# ── 4. Parse VCF once ──
all_results = []
mutation_matrix = []  # for PCA later

print(f"Parsing VCF...")
with gzip.open('data/demo/drprg_sparse.vcf.gz', 'rt') as f:
    for i, line in enumerate(f):
        if line.startswith('#'):
            continue
        cols = line.strip().split('\t')
        pos = int(cols[1])
        vid = f"{cols[0]}:{pos}{cols[3]}>{cols[4].split(',')[0]}"
        
        r1 = s1 = 0
        gt_row = []
        for s in filtered:
            gt = cols[9 + s_idx[s]].split(':')[0]
            val = 1 if gt in ('1/1', '1|1', '0/1', '0|1') else 0
            gt_row.append(val)
            if val:
                if p_dict[s]:
                    r1 += 1
                else:
                    s1 += 1
        
        if r1 == 0 and s1 == 0:
            continue
        
        r0, s0 = n_res - r1, n_sus - s1
        or_val, pv = fisher_exact([[r1, r0], [s1, s0]])
        
        g = find_gene(pos)
        gene_annot = f"{g['name']} ({g['gene_id']})" if g else 'intergenic'
        is_known = any(gene_annot.find(k) >= 0 for k in list(KNOWN) + ['rpoB','katG','embB','pncA','gyrA','rrs'])
        
        all_results.append({'vid': vid, 'pos': pos, 'r1': r1, 'r0': r0, 's1': s1, 's0': s0,
                            'pv': pv, 'or': or_val, 'gene': gene_annot, 'known': is_known})
        mutation_matrix.append((vid, gt_row))
        
        if (i+1) % 5000 == 0:
            print(f"  {i+1} variants processed ({time.time()-t0:.1f}s)")

print(f"Done: {len(all_results)} segregating variants ({time.time()-t0:.1f}s)")

# ── 5. Multiple testing correction ──
pvals = np.array([r['pv'] for r in all_results])
reject, pcorr, _, _ = multipletests(pvals, method='fdr_bh')

df = pd.DataFrame(all_results)
df['p_corrected'] = pcorr
df['significant'] = reject
df.to_csv('analysis/results/association_results_74.csv', index=False)

sig_df = df[df['significant']]
known_df = df[df['known']]

print(f"\n{'='*60}")
print(f"RESULTS: {len(df)} variants tested, {len(sig_df)} FDR-significant")
print(f"{'='*60}")
print(f"Top 15 associations:")
for _, r in df.nsmallest(15, 'pv').iterrows():
    print(f"  {r['vid']:30s} p={r['pv']:.2e} OR={r['or']:.1f} {'***' if r['significant'] else ''} {r['gene']}")

print(f"\nKnown resistance gene variants ({len(known_df)}):")
for _, r in known_df.sort_values('pv').iterrows():
    print(f"  {r['vid']:30s} R={r['r1']}/{r['r1']+r['r0']} S={r['s1']}/{r['s1']+r['s0']} OR={r['or']:.1f} p={r['pv']:.2e} {r['gene']}")

# ── 6. Manhattan plot ──
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 4))
xs = np.arange(len(df))
ax.scatter(xs, -np.log10(np.maximum(df['pv'], 1e-300)), s=4, c='#3498db', alpha=0.4)
sig_idx = df['significant'].values
ax.scatter(xs[sig_idx], -np.log10(np.maximum(df.loc[sig_idx, 'pv'], 1e-300)), s=6, c='#e74c3c', alpha=0.8)
ax.axhline(-np.log10(0.05/len(df)), color='gray', ls='--', alpha=0.4, label='Bonferroni')
ax.set_title(f'TB Resistance Association: 74 genomes ({len(sig_df)} FDR-significant at q<0.05)')
ax.set_xlabel('Variant index'); ax.set_ylabel('−log₁₀(p)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('analysis/results/figures/manhattan_74.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nManhattan saved")

# ── 7. Known resistance heatmap ──
if len(known_df) > 0:
    known_vids = known_df['vid'].tolist()
    samp_order = sorted(filtered, key=lambda s: p_dict[s])
    samp_to_idx = {s: i for i, s in enumerate(filtered)}  # index in filtered (mutation matrix order)
    
    hm = np.zeros((len(known_vids), len(samp_order)), dtype=int)
    vid_idx = {v: i for i, v in enumerate(known_vids)}
    
    for vid, row in mutation_matrix:
        if vid in vid_idx:
            for sj, s in enumerate(samp_order):
                hm[vid_idx[vid], sj] = row[samp_to_idx[s]]
    
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(14, max(3, len(known_vids)*0.25)))
    cmap = sns.color_palette(['#f0f0f0', '#c0392b'])
    labs = [f"{s.split('.')[2][:10]}" for s in samp_order]
    sns.heatmap(hm, cmap=cmap, cbar_kws={'label': 'Variant present'},
                xticklabels=labs, yticklabels=[v[:30] for v in known_vids], ax=ax)
    ax.set_title(f'Known Resistance Gene Variants ({len(known_vids)} loci)')
    plt.tight_layout()
    plt.savefig('analysis/results/figures/known_resistance_heatmap_74.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved")

# ── 8. Effect size histogram ──
fig, ax = plt.subplots(figsize=(8, 4))
ors = df['or'].replace([np.inf, -np.inf], np.nan).dropna().clip(0, 100)
log_ors = np.log2(ors[ors > 0])
if len(log_ors) > 0:
    ax.hist(log_ors, bins=100, color='#3498db', alpha=0.7)
    ax.axvline(0, color='gray', ls='--')
    ax.set_xlabel('log₂(Odds Ratio)'); ax.set_ylabel('Count')
    ax.set_title('Effect size distribution across all variants')
plt.tight_layout()
plt.savefig('analysis/results/figures/effect_sizes_74.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Effect size plot saved")

print(f"\nDone. Total: {time.time()-t0:.1f}s")
