import gzip, csv, re, os, time
import numpy as np
import pandas as pd

t0 = time.time()

# ── 1. Load VCF sample names and parse ──
with gzip.open('data/demo/drprg_sparse.vcf.gz', 'rt') as f:
    for line in f:
        if line.startswith('#CHROM'):
            vcf_samples = line.strip().split('\t')[9:]
            break

def vcf_to_uniqueid(name):
    m = re.match(r'site\.(\d+)\.iso\.(\d+)\.subject\.(.+?)\.lab_id\.(.+?)\.seq_reps\.(.+)', name)
    if m:
        return f"site.{m.group(1)}.subj.{m.group(3)}.lab.{m.group(4)}.iso.{m.group(2)}"
    return None

vcf_to_uid = {}
for s in vcf_samples:
    uid = vcf_to_uniqueid(s)
    if uid:
        vcf_to_uid[s] = uid

print(f"Mapped {len(vcf_to_uid)}/{len(vcf_samples)} VCF samples to UNIQUEIDs")

# ── 2. Load DST_MEASUREMENTS ──
print("Loading DST_MEASUREMENTS...")
mic_data = {}  # {uniqueid: {drug: {'mic': float, 'phenotype': str}}}
with gzip.open('/tmp/DST_MEASUREMENTS.csv.gz', 'rt') as f:
    reader = csv.DictReader(f)
    for row in reader:
        uid = row['UNIQUEID']
        drug = row['DRUG']
        mic_str = row['METHOD_MIC'].strip()
        pheno = row['PHENOTYPE'].strip()
        
        # Parse MIC to numeric
        m = re.match(r'([<>=]+)\s*([\d.]+)', mic_str)
        if m:
            mic_val = float(m.group(2))
        else:
            try:
                mic_val = float(mic_str)
            except ValueError:
                continue
        
        if uid not in mic_data:
            mic_data[uid] = {}
        mic_data[uid][drug] = {'mic': mic_val, 'phenotype': pheno}

print(f"  {len(mic_data)} samples with MIC data")
drugs_seen = set()
for uid, drugs in mic_data.items():
    drugs_seen.update(drugs.keys())
print(f"  {len(drugs_seen)} drugs measured")

# ── 3. Map VCF samples to MIC ──
# Load phenotype file for the 74 samples
pheno = pd.read_csv('analysis/results/phenotype_100.csv')
p_dict = dict(zip(pheno['sample'], pheno['is_resistant']))
filtered = list(p_dict.keys())

# Track MIC per VCF sample
vcf_mic = {}  # {vcf_sample: {drug: mic_val}}
vcf_pheno = {}  # {vcf_sample: {drug: binary_phenotype}}

for vcf_s in filtered:
    uid = vcf_to_uid.get(vcf_s)
    if uid and uid in mic_data:
        vcf_mic[vcf_s] = mic_data[uid]
        vcf_pheno[vcf_s] = {d: info['phenotype'] for d, info in mic_data[uid].items()}

print(f"\n{len(vcf_mic)} samples with matched MIC data")

# ── 4. Create per-drug MIC matrix ──
# Focus on key TB drugs
KEY_DRUGS = ['RIF', 'INH', 'EMB', 'PZA', 'MXF', 'STM', 'AMI', 'KAN', 'CAP', 'LZD', 'BDQ', 'CFZ', 'DLM', 'LEV', 'OFX']

mic_matrix = pd.DataFrame(index=filtered, columns=KEY_DRUGS, dtype=float)
pheno_matrix = pd.DataFrame(index=filtered, columns=KEY_DRUGS, dtype=str)

for s in filtered:
    drugs = vcf_mic.get(s, {})
    for d in KEY_DRUGS:
        if d in drugs:
            mic_matrix.loc[s, d] = drugs[d]['mic']
            pheno_matrix.loc[s, d] = drugs[d]['phenotype']

# Log2 transform MIC
log_mic = np.log2(mic_matrix)
log_mic.columns = [f"{c}_log2MIC" for c in log_mic.columns]

# ── 5. Print summary ──
print(f"\n{'='*60}")
print(f"MIC data coverage ({len(filtered)} samples)")
print(f"{'='*60}")
for d in KEY_DRUGS:
    n = mic_matrix[d].notna().sum()
    if n > 0:
        vals = mic_matrix[d].dropna()
        r_frac = (pheno_matrix[d] == 'R').sum()
        print(f"  {d:4s}: {n:3d}/{len(filtered)} samples  MIC range={vals.min():.2f}-{vals.max():.2f}  {int(r_frac)} resistant")

# ── 6. Save ──
log_mic.to_csv('analysis/results/mic_data_log2.csv')
pheno_matrix.to_csv('analysis/results/mic_phenotypes.csv')
print(f"\nSaved MIC data")

# ── 7. MIC-colored PCA ──
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib as mpl

# Build mutation matrix for PCA
s_idx = {s: i for i, s in enumerate(vcf_samples) if s in filtered}
pca_data = pd.DataFrame(0, index=filtered, columns=range(100))  # temp

with gzip.open('data/demo/drprg_sparse.vcf.gz', 'rt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        cols = line.strip().split('\t')
        # Just use the first 100 variants for speed
        pass

# Build proper matrix from first 5000 variants
rows = []
with gzip.open('data/demo/drprg_sparse.vcf.gz', 'rt') as f:
    for i, line in enumerate(f):
        if line.startswith('#') or i > 5000:
            continue
        cols = line.strip().split('\t')
        vid = f"{cols[0]}:{cols[1]}"
        for si, s in enumerate(filtered):
            gt = cols[9 + s_idx[s]].split(':')[0]
            rows.append({'sample': s, 'variant': vid, 'gt': 1 if gt in ('1/1','1|1','0/1','0|1') else 0})

pca_mat = pd.pivot_table(pd.DataFrame(rows), index='variant', columns='sample', values='gt', fill_value=0).T
pca = PCA(n_components=2)
pcs = pca.fit_transform(pca_mat.values)

for drug in ['RIF', 'INH', 'EMB']:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Binary phenotype
    for title, ax, use_continuous in [('Binary R/S', axes[0], False), ('Continuous log₂(MIC)', axes[1], True)]:
        if use_continuous:
            vals = log_mic[f'{drug}_log2MIC'].values
            valid = ~np.isnan(vals)
            sc = ax.scatter(pcs[valid, 0], pcs[valid, 1], c=vals[valid], 
                          cmap='viridis', s=50, alpha=0.8, edgecolors='k', linewidth=0.3)
            plt.colorbar(sc, ax=ax, label=f'log₂({drug} MIC)')
        else:
            r_idx = [i for i, s in enumerate(filtered) if pheno_matrix.loc[s, drug] == 'R']
            s_idx_plot = [i for i, s in enumerate(filtered) if pheno_matrix.loc[s, drug] == 'S']
            u_idx = [i for i, s in enumerate(filtered) if pheno_matrix.loc[s, drug] not in ('R', 'S')]
            ax.scatter(pcs[s_idx_plot, 0], pcs[s_idx_plot, 1], c='#2ecc71', s=50, alpha=0.7, label='Susceptible')
            ax.scatter(pcs[r_idx, 0], pcs[r_idx, 1], c='#e74c3c', s=50, alpha=0.7, label='Resistant')
            ax.legend(fontsize=9)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title(f'{drug}: {title}')
    
    plt.tight_layout()
    plt.savefig(f'analysis/results/figures/MIC_PCA_{drug}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"MIC PCA for {drug} saved")

# ── 8. MIC histogram per drug ──
fig, axes = plt.subplots(3, 5, figsize=(16, 10))
axes = axes.flatten()
for i, drug in enumerate(KEY_DRUGS):
    ax = axes[i]
    vals = mic_matrix[drug].dropna()
    if len(vals) > 0:
        ax.hist(np.log2(vals), bins=20, color='#3498db', alpha=0.7)
        ax.axvline(np.log2(1.0), color='red', ls='--', alpha=0.5, label='CLSI breakpoint')
        ax.set_title(f'{drug} (n={len(vals)})', fontsize=9)
        ax.set_xlabel('log₂(MIC)')
        ax.tick_params(labelsize=7)
    else:
        ax.set_title(f'{drug} (no data)')
    ax.set_ylabel('Count')
plt.suptitle('MIC Distributions per Drug (log₂ scale)', fontsize=14)
plt.tight_layout()
plt.savefig('analysis/results/figures/MIC_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"MIC distributions saved")

# ── 9. Continuous association test (RIF as example) ──
# For each variant, test if carrier status correlates with log2(MIC)
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

drug = 'RIF'
mic_col = f'{drug}_log2MIC'
rif_mic = log_mic[mic_col].dropna()
rif_samples = rif_mic.index.tolist()

rif_tests = []
with gzip.open('data/demo/drprg_sparse.vcf.gz', 'rt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        cols = line.strip().split('\t')
        pos = int(cols[1])
        vid = f"{cols[0]}:{pos}{cols[3]}>{cols[4].split(',')[0]}"
        
        carriers = []
        noncarriers = []
        for s in rif_samples:
            gt = cols[9 + s_idx[s]].split(':')[0]
            val = 1 if gt in ('1/1','1|1','0/1','0|1') else 0
            mic = rif_mic[s]
            if val:
                carriers.append(mic)
            else:
                noncarriers.append(mic)
        
        if len(carriers) < 2 or len(noncarriers) < 2:
            continue
        
        stat, pval = mannwhitneyu(carriers, noncarriers, alternative='two-sided')
        rif_tests.append({
            'variant': vid, 'pos': pos,
            'n_carrier': len(carriers), 'n_non': len(noncarriers),
            'carrier_mean_MIC': np.mean(carriers),
            'noncarrier_mean_MIC': np.mean(noncarriers),
            'p_value': pval
        })

rif_df = pd.DataFrame(rif_tests)
reject, pcorr, _, _ = multipletests(rif_df['p_value'].values, method='fdr_bh')
rif_df['p_corrected'] = pcorr
rif_df['significant'] = reject
rif_df.to_csv('analysis/results/rif_mic_association.csv', index=False)

print(f"\nRIF MIC association: {len(rif_df)} variants tested, {reject.sum()} FDR-significant")
top_rif = rif_df.nsmallest(5, 'p_value')
for _, r in top_rif.iterrows():
    print(f"  {r['variant']:30s} RIF_MIC: carrier={r['carrier_mean_MIC']:.2f} non={r['noncarrier_mean_MIC']:.2f} p={r['p_value']:.2e}")

print(f"\nTotal: {time.time()-t0:.1f}s")
