"""
Merge existing VCF-based homoplasy with new assembly-based counts.
Updates residue_hotspot_data.csv with combined homoplasy estimates.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load existing feature table
feat_path = PROJECT_DIR / "analysis" / "results" / "hotspot_model" / "residue_hotspot_data.csv"
df = pd.read_csv(feat_path)
print(f"Existing features: {len(df)} residues, {df['homoplasy_count'].sum()} total homoplasy count (from {df['homoplasy_count'].max()} max)")

# Load assembly-based homoplasy
asm_path = PROJECT_DIR / "analysis" / "results" / "homoplasy" / "homoplasy_from_assemblies.csv"
asm = pd.read_csv(asm_path)
print(f"\nAssembly-based homoplasy: {len(asm)} residues ({asm['n_genomes'].iloc[0]} genomes)")
print(f"  Total mutated residues: {(asm['homoplasy_count'] > 0).sum()}")
print(f"  Total homoplasy count: {asm['homoplasy_count'].sum()}")

# Add n_genomes column to existing table (initialize to VCF sample count)
old_vcf_count = 117  # original VCF sample count
if 'n_genomes' not in df.columns:
    df['n_genomes'] = old_vcf_count

# For each residue in the existing table, check if we have new assembly data
pipeline_genes = set(df['gene'].unique())
assembly_n = asm['n_genomes'].iloc[0]  # 60 genomes
matched = 0
new_mutations = 0

for _, row in asm.iterrows():
    gene = row['gene']
    pos = row['residue_pos']
    asm_count = row['homoplasy_count']
    asm_alleles = row['homoplasy_alleles']
    
    mask = (df['gene'] == gene) & (df['residue_pos'] == pos)
    if mask.any():
        matched += 1
        # Add assembly counts to existing VCF-based homoplasy
        if asm_count > 0:
            df.loc[mask, 'homoplasy_count'] = df.loc[mask, 'homoplasy_count'] + asm_count
            df.loc[mask, 'homoplasy_alleles'] = df.loc[mask, 'homoplasy_alleles'] + asm_alleles
            new_mutations += 1
        # Update n_genomes: VCF samples + assembly genomes
        df.loc[mask, 'n_genomes'] = old_vcf_count + assembly_n
    else:
        # This residue didn't exist in VCF data (new gene coverage from assemblies)
        # Only add if the gene is in our pipeline
        if gene in pipeline_genes and row['residue_pos'] > 0:
            new_row = {col: 0 for col in df.columns}
            new_row['gene'] = gene
            new_row['residue_pos'] = pos
            new_row['homoplasy_count'] = asm_count if asm_count > 0 else 0
            new_row['homoplasy_alleles'] = asm_alleles if asm_count > 0 else 0
            new_row['n_genomes'] = old_vcf_count + assembly_n
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

print(f"\nMatched residues: {matched}")
print(f"Residues with new assembly mutations added: {new_mutations}")
print(f"New residues added (assembly-only coverage): {len(df) - len(pd.read_csv(feat_path))}")
print(f"\nUpdated homoplasy_count range: {df['homoplasy_count'].min()} - {df['homoplasy_count'].max()}")
print(f"Updated non-zero residues: {(df['homoplasy_count'] > 0).sum()}")
print(f"Updated total homoplasy count: {df['homoplasy_count'].sum()}")

# Save updated feature table
out_path = PROJECT_DIR / "analysis" / "results" / "hotspot_model" / "residue_hotspot_data_updated.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved updated features to {out_path}")

# Summary per gene
print("\nPer-gene homoplasy summary (updated):")
for gene in pipeline_genes:
    gdf = df[df['gene'] == gene]
    n_mut = (gdf['homoplasy_count'] > 0).sum()
    total = gdf['homoplasy_count'].sum()
    max_c = gdf['homoplasy_count'].max()
    print(f"  {gene:8s}: {n_mut:4d} mutated residues, total count={total:4d}, max={max_c}")
