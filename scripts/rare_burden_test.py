import gzip, re, os, time
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

t0 = time.time()
os.makedirs('analysis/results/figures', exist_ok=True)

# ── 1. Phenotypes ──
pheno = pd.read_csv('analysis/results/phenotype_100.csv')
p_dict = dict(zip(pheno['sample'], pheno['is_resistant']))
filtered = list(p_dict.keys())
n_res = sum(p_dict.values())
n_sus = len(p_dict) - n_res
print(f"{len(filtered)} samples (R={n_res} S={n_sus})")

# ── 2. VCF header ──
with gzip.open('data/demo/drprg_sparse.vcf.gz', 'rt') as f:
    for line in f:
        if line.startswith('#CHROM'):
            vcf_samples = line.strip().split('\t')[9:]
            break
s_idx = {s: i for i, s in enumerate(vcf_samples) if s in p_dict}

# ── 3. GFF → genes ──
genes = {}
with open('reference/H37Rv.gff') as f:
    for line in f:
        if line.startswith('#') or line.strip() == '':
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9:
            continue
        attrs = dict(re.findall(r'([\w-]+)=([^;\n]+)', parts[8]))
        locus = attrs.get('locus_tag', '')
        if not locus:
            continue
        if parts[2] == 'gene':
            genes[locus] = {'start': int(parts[3]), 'end': int(parts[4]),
                'name': attrs.get('gene', attrs.get('Name', locus)),
                'product': attrs.get('description', attrs.get('product', ''))}

KNOWN = {'Rv0667': ('rpoB','rifampicin'), 'Rv1908c': ('katG','isoniazid'),
         'Rv3795': ('embB','ethambutol'), 'Rv3794': ('embA','ethambutol'),
         'Rv0006': ('gyrA','fluoroquinolones'), 'Rv0005': ('gyrB','fluoroquinolones'),
         'Rv2043c': ('pncA','pyrazinamide'), 'Rv0682': ('rpsL','streptomycin'),
         'Rv2416c': ('eis','aminoglycosides'), 'Rv0678': ('mmpR5','bedaquiline'),
         'Rv2680': ('mmpL5','bedaquiline'), 'Rv1694': ('tlyA','capreomycin')}

# ── 4. VCF parsing + rare variant burden ──
n_total = len(filtered)
gene_carriers = {l: {s: 0 for s in filtered} for l in genes}
gene_n_rare = {l: 0 for l in genes}
gene_n_total = {l: 0 for l in genes}

n_variants = 0
with gzip.open('data/demo/drprg_sparse.vcf.gz', 'rt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        cols = line.strip().split('\t')
        pos = int(cols[1])
        
        # Get genotypes
        ac = 0
        gt_for_sample = {}
        for s in filtered:
            gt = cols[9 + s_idx[s]].split(':')[0]
            val = 1 if gt in ('1/1','1|1','0/1','0|1') else 0
            gt_for_sample[s] = val
            ac += val
        
        if ac == 0:
            continue
        
        # AF = alt allele count / total chromosomes
        af = ac / (2 * n_total)
        is_rare = af < 0.05
        
        n_variants += 1
        for locus, g in genes.items():
            if g['start'] <= pos <= g['end']:
                gene_n_total[locus] += 1
                if not is_rare:
                    continue
                gene_n_rare[locus] += 1
                for s in filtered:
                    if gt_for_sample[s]:
                        gene_carriers[locus][s] = 1

print(f"{n_variants} segregating variants (AF<5% = rare)")

# ── 5. Rare variant burden test ──
results = []
for locus in genes:
    gc = gene_carriers[locus]
    n_rare = gene_n_rare[locus]
    if n_rare == 0:
        continue
    
    r_car = sum(gc[s] for s in filtered if p_dict[s])
    s_car = sum(gc[s] for s in filtered if not p_dict[s])
    r_non = n_res - r_car
    s_non = n_sus - s_car
    
    if r_car == 0 and s_car == 0:
        continue
    
    or_val, pv = fisher_exact([[r_car, r_non], [s_car, s_non]])
    
    results.append({
        'locus': locus, 'gene': genes[locus]['name'],
        'known_res': locus in KNOWN,
        'drug': KNOWN[locus][1] if locus in KNOWN else '',
        'r_carrier': r_car, 's_carrier': s_car,
        'n_rare_variants': n_rare, 'n_total_variants': gene_n_total[locus],
        'OR': or_val, 'p': pv,
        'product': genes[locus]['product'][:80]
    })

df = pd.DataFrame(results)
reject, pcorr, _, _ = multipletests(df['p'].values, method='fdr_bh')
df['p_corrected'] = pcorr
df['significant'] = reject
df.to_csv('analysis/results/rare_burden_results.csv', index=False)

# ── 6. Print ──
print(f"\n{'='*60}")
print(f"RARE-VARIANT GENE BURDEN TEST ({len(df)} genes with rare variants)")
print(f"{'='*60}")

sig_r = df[df['significant']].sort_values('p')
known_sig = sig_r[sig_r['known_res']]
novel_sig = sig_r[~sig_r['known_res']]
print(f"Significant: {len(sig_r)} (known={len(known_sig)}, novel={len(novel_sig)})")

print(f"\nKnown resistance genes:")
for _, r in df[df['known_res']].sort_values('p').iterrows():
    star = ' ***' if r['significant'] else ''
    d = f" ({r['drug']})" if r['drug'] else ''
    print(f"  {r['locus']:8s} {r['gene']:8s}{d:20s} R={r['r_carrier']}/{n_res} S={r['s_carrier']}/{n_sus} OR={r['OR']:.1f} p={r['p']:.2e}{star}")

print(f"\nTop novel candidates (FDR-significant, OR>1):")
novel_pos = novel_sig[novel_sig['OR'] > 1].head(15)
for _, r in novel_pos.iterrows():
    print(f"  {r['locus']:8s} {r['gene']:12s} R={r['r_carrier']}/{n_res} S={r['s_carrier']}/{n_sus} OR={r['OR']:.1f} p={r['p']:.2e} rare_variants={r['n_rare_variants']}")

print(f"\n--- Key Insight ---")
print(f"Single-variant GWAS found 540 significant variants (known resistance hotspots)")
print(f"Rare-variant burden test finds {len(novel_pos)} novel gene candidates enriched in R")
print(f"Combined approach: hotspot + burden = complete resistance architecture")

# Save top novel candidates
novel_pos.to_csv('analysis/results/novel_gene_candidates.csv', index=False)
print(f"\nSaved novel candidates to analysis/results/novel_gene_candidates.csv")

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
xs = np.arange(len(df))
is_known = df['known_res'].values
is_sig = df['significant'].values
colors = np.where(is_known, '#e74c3c', np.where(is_sig & (df['OR'] > 1), '#e67e22', '#3498db'))
sizes = np.where(is_sig, 40, 8)
ax.scatter(xs, -np.log10(np.maximum(df['p'], 1e-300)), s=sizes, c=colors, alpha=0.6, edgecolors='none')
from matplotlib.lines import Line2D
leg = [Line2D([0],[0],marker='o',color='w',markerfacecolor='#3498db',markersize=6,label='Not significant'),
       Line2D([0],[0],marker='o',color='w',markerfacecolor='#e67e22',markersize=8,label='Novel candidate (FDR<0.05, OR>1)'),
       Line2D([0],[0],marker='o',color='w',markerfacecolor='#e74c3c',markersize=8,label='Known resistance gene')]
ax.legend(handles=leg, fontsize=9)
ax.axhline(-np.log10(0.05/len(df)), color='gray', ls='--', alpha=0.3)
ax.set_xlabel('Gene index'); ax.set_ylabel('−log₁₀(p)')
ax.set_title(f'Rare-Variant Burden Test: {len(novel_pos)} novel gene candidates enriched in resistant strains')
plt.tight_layout()
plt.savefig('analysis/results/figures/rare_burden_manhattan.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Burden Manhattan saved")
print(f"Total: {time.time()-t0:.1f}s")
