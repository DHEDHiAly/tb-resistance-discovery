import gzip, re, os, time
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency
from statsmodels.stats.multitest import multipletests

t0 = time.time()
os.makedirs('analysis/results/figures', exist_ok=True)

# ── 1. Phenotypes ──
pheno = pd.read_csv('analysis/results/phenotype_100.csv')
p_dict = dict(zip(pheno['sample'], pheno['is_resistant']))
filtered = list(p_dict.keys())
n_res = sum(p_dict.values())
n_sus = len(p_dict) - n_res
print(f"Samples: {len(filtered)} (R={n_res} S={n_sus})")

# ── 2. VCF header ──
with gzip.open('data/demo/drprg_sparse.vcf.gz', 'rt') as f:
    for line in f:
        if line.startswith('#CHROM'):
            vcf_samples = line.strip().split('\t')[9:]
            break
s_idx = {s: i for i, s in enumerate(vcf_samples) if s in p_dict}

# ── 3. Parse GFF → build gene list and CDS positions ──
genes = {}  # locus -> {start, end, name, product, strand}
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
            if locus not in genes:
                genes[locus] = {
                    'start': int(parts[3]), 'end': int(parts[4]), 'strand': parts[6],
                    'name': attrs.get('gene', attrs.get('Name', locus)),
                    'product': attrs.get('description', ''), 'biotype': attrs.get('gene_biotype', ''),
                    'cds': []
                }
        elif parts[2] == 'CDS' and locus in genes:
            genes[locus]['cds'].append({
                'start': int(parts[3]), 'end': int(parts[4]),
                'product': attrs.get('product', ''),
                'protein_id': attrs.get('protein_id', '')
            })

print(f"Genes: {len(genes)}")

# Known resistance genes (WHO catalog + literature)
KNOWN_RES = {
    'Rv0667': ('rpoB', 'rifampicin'), 'Rv1908c': ('katG', 'isoniazid'),
    'Rv3795': ('embB', 'ethambutol'), 'Rv3794': ('embA', 'ethambutol'),
    'Rv0006': ('gyrA', 'fluoroquinolones'), 'Rv0005': ('gyrB', 'fluoroquinolones'),
    'Rv2043c': ('pncA', 'pyrazinamide'), 'Rv0682': ('rpsL', 'streptomycin'),
    'Rv2416c': ('eis', 'aminoglycosides'), 'Rv1258c': ('tap', 'aminoglycosides'),
    'Rv0678': ('mmpR5', 'bedaquiline'), 'Rv2680': ('mmpL5', 'bedaquiline'),
    'Rv1694': ('tlyA', 'capreomycin'), 'Rv1473': ('Rv1473', 'aminoglycosides'),
}

KNOWN_LOCUS = set(KNOWN_RES.keys())

# ── 4. Parse VCF and compute per-gene mutation counts ──
# Track: for each gene and each sample, does the sample have ANY variant in that gene?
gene_carrier = {l: {s: 0 for s in filtered} for l in genes}
gene_n_variants = {l: 0 for l in genes}
gene_known_count = {l: 0 for l in genes}

n_variants_processed = 0

with gzip.open('data/demo/drprg_sparse.vcf.gz', 'rt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        cols = line.strip().split('\t')
        pos = int(cols[1])
        ref, alt = cols[3], cols[4].split(',')[0]
        
        # Get genotypes for each sample
        gt_for_sample = {}
        for s in filtered:
            gt = cols[9 + s_idx[s]].split(':')[0]
            gt_for_sample[s] = 1 if gt in ('1/1', '1|1', '0/1', '0|1') else 0
        
        if sum(gt_for_sample.values()) == 0:
            continue
        
        # Assign to gene(s)
        for locus, g in genes.items():
            if g['start'] <= pos <= g['end']:
                gene_n_variants[locus] += 1
                if locus in KNOWN_LOCUS:
                    gene_known_count[locus] += 1
                for s in filtered:
                    if gt_for_sample[s]:
                        gene_carrier[locus][s] = 1
        
        n_variants_processed += 1
        if n_variants_processed % 5000 == 0:
            print(f"  {n_variants_processed} variants ({time.time()-t0:.1f}s)")

print(f"Processed: {n_variants_processed} segregating variants")

# ── 5. Gene-level burden test ──
burden_results = []
for locus, g in sorted(genes.items()):
    gc = gene_carrier[locus]
    n_r_carrier = sum(gc[s] for s in filtered if p_dict[s])
    n_s_carrier = sum(gc[s] for s in filtered if not p_dict[s])
    n_r_non = n_res - n_r_carrier
    n_s_non = n_sus - n_s_carrier
    
    if n_r_carrier == 0 and n_s_carrier == 0:
        continue
    
    table = [[n_r_carrier, n_r_non], [n_s_carrier, n_s_non]]
    odds_ratio, pval = fisher_exact(table)
    
    burden_results.append({
        'locus': locus, 'gene': g['name'], 'product': g['product'][:80],
        'known_res': locus in KNOWN_LOCUS,
        'drug': KNOWN_RES[locus][1] if locus in KNOWN_RES else '',
        'n_r_carrier': n_r_carrier, 'n_s_carrier': n_s_carrier,
        'n_variants': gene_n_variants[locus],
        'n_known_variants': gene_known_count[locus],
        'burden_OR': odds_ratio, 'burden_p': pval,
    })

burden = pd.DataFrame(burden_results)

# BH correction
if len(burden) > 0:
    reject, p_corr, _, _ = multipletests(burden['burden_p'].values, method='fdr_bh')
    burden['burden_p_corrected'] = p_corr
    burden['burden_significant'] = reject

burden.to_csv('analysis/results/gene_burden_results.csv', index=False)

# ── 6. Print results ──
print(f"\n{'='*70}")
print(f"GENE-LEVEL BURDEN TEST: {len(burden)} genes tested")
print(f"  Significant (FDR<0.05): {burden['burden_significant'].sum()}")
print(f"  Known resistance genes tested: {len(KNOWN_RES)}")
print(f"{'='*70}")

sig_burden = burden[burden['burden_significant']].sort_values('burden_p')
print(f"\nTop 20 genes by mutation burden enrichment:")
for _, r in sig_burden.head(20).iterrows():
    flag = " **KNOWN**" if r['known_res'] else ""
    print(f"  {r['locus']:10s} {r['gene']:15s} R_carrier={r['n_r_carrier']:2d}/{n_res} S_carrier={r['n_s_carrier']:2d}/{n_sus} OR={r['burden_OR']:.2f} p={r['burden_p']:.2e}{flag}")

# Known resistance genes specifically
print(f"\nKnown resistance genes - burden test:")
for _, r in burden[burden['known_res']].sort_values('burden_p').iterrows():
    star = " ***" if r['burden_significant'] else ""
    drug_name = r['drug'] if r['drug'] else ''
print(f"  {r['locus']:10s} {r['gene']:15s} ({drug_name:15s}) R_carrier={r['n_r_carrier']}/{n_res} S_carrier={r['n_s_carrier']}/{n_sus} OR={r['burden_OR']:.2f} p={r['burden_p']:.2e}{star}")

# ── 7. Novel discovery: genes not known as resistance but significant ──
novel_sig = burden[(~burden['known_res']) & (burden['burden_significant'])]
print(f"\n--- NOVEL resistance gene candidates ({len(novel_sig)} genes) ---")
for _, r in novel_sig.sort_values('burden_p').head(15).iterrows():
    print(f"  {r['locus']:10s} {r['gene']:20s} OR={r['burden_OR']:.2f} p={r['burden_p']:.2e} variants={r['n_variants']} product={r['product'][:60]}")

# ── 8. Known vs novel signal comparison ──
print(f"\n--- Known vs Novel Signal ---")
known_burden = burden[burden['known_res']]
novel_burden = burden[~burden['known_res']]
print(f"  Known resistance genes: median OR={known_burden['burden_OR'].dropna().median():.2f} (n={len(known_burden)})")
print(f"  Novel genes:            median OR={novel_burden['burden_OR'].dropna().median():.2f} (n={len(novel_burden)})")
print(f"  Known genes significant: {known_burden['burden_significant'].sum()}/{len(known_burden)}")
print(f"  Novel genes significant:  {novel_burden['burden_significant'].sum()}/{len(novel_burden)}")

# ── 10. Generate figures ──
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 10a. Gene burden Manhattan
fig, ax = plt.subplots(figsize=(12, 5))
xs = np.arange(len(burden))
is_known = burden['known_res'].values
is_sig = burden['burden_significant'].values
colors = np.where(is_known, '#e74c3c', '#3498db')
sizes = np.where(is_sig, 30, 10)
ax.scatter(xs, -np.log10(np.maximum(burden['burden_p'], 1e-300)), s=sizes, c=colors, alpha=0.6, edgecolors='none')
ax.axhline(-np.log10(0.05/len(burden)), color='gray', ls='--', alpha=0.3, label='Bonferroni')
ax.set_xlabel('Gene index'); ax.set_ylabel('−log₁₀(p)')
ax.set_title(f'Gene-Level Mutation Burden: {is_sig.sum()} genes significant (FDR q<0.05)')
from matplotlib.lines import Line2D
leg = [Line2D([0],[0],marker='o',color='w',markerfacecolor='#3498db',markersize=6,label='Other gene'),
       Line2D([0],[0],marker='o',color='w',markerfacecolor='#e74c3c',markersize=6,label='Known resistance gene'),
       Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',markersize=8,label='Significant (FDR<0.05)')]
ax.legend(handles=leg, fontsize=8)
plt.tight_layout()
plt.savefig('analysis/results/figures/gene_burden_manhattan.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nGene burden Manhattan saved")

# 10b. Known vs Novel OR comparison
fig, ax = plt.subplots(figsize=(8, 4))
known_or = np.log2(known_burden['burden_OR'].replace([0, np.inf, -np.inf], np.nan).dropna())
novel_or = np.log2(novel_burden['burden_OR'].replace([0, np.inf, -np.inf], np.nan).dropna())
ax.hist(known_or, bins=30, alpha=0.6, label=f'Known resistance genes (n={len(known_or)})', color='#e74c3c')
ax.hist(novel_or, bins=50, alpha=0.4, label=f'Other genes (n={len(novel_or)})', color='#3498db')
ax.axvline(0, color='gray', ls='--')
ax.set_xlabel('log₂(Odds Ratio)'); ax.set_ylabel('Count')
ax.set_title('Gene-level burden effect sizes: Known vs Novel genes')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('analysis/results/figures/gene_burden_known_vs_novel.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Known vs Novel plot saved")

# 10c. Top novel genes bar chart
top_novel = burden[(~burden['known_res']) & (burden['burden_significant'])].nsmallest(15, 'burden_p')
if len(top_novel) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    ypos = range(len(top_novel))
    ax.barh(ypos, -np.log10(top_novel['burden_p'].values), color='#3498db', alpha=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r['gene']} ({r['locus']})" for _, r in top_novel.iterrows()], fontsize=8)
    ax.axvline(-np.log10(0.05/len(burden)), color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('−log₁₀(p)')
    ax.set_title('Top 15 Novel Resistance Gene Candidates')
    plt.tight_layout()
    plt.savefig('analysis/results/figures/novel_gene_candidates.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Novel gene candidates plot saved")

print(f"\nTotal: {time.time()-t0:.1f}s")
print(f"Results saved to analysis/results/gene_burden_results.csv")
print(f"Novel resistance gene candidates: {len(novel_sig)}")
