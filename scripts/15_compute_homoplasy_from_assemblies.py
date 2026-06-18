"""
Compute homoplasy counts from assembled TB genomes.
Aligns each genome assembly to H37Rv reference for 12 resistance genes,
counts non-synonymous mutations per residue, and outputs homoplacy features.
"""

import os, sys, gzip, csv, json
from pathlib import Path
from collections import defaultdict

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq

PROJECT_DIR = Path(__file__).resolve().parent.parent
REFERENCE_DIR = PROJECT_DIR / "reference"
GENOME_DIR = PROJECT_DIR / "data" / "genomes"
OUTPUT_DIR = PROJECT_DIR / "analysis" / "results" / "homoplasy"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 12 resistance genes with their coordinates
# From the pipeline's gene definitions
# 13 protein-coding resistance genes from the pipeline (rrs is rRNA, excluded)
RESISTANCE_GENES = {
    'rpoB':  (759807, 763325, '+'),
    'katG':  (2153889, 2156111, '-'),
    'embB':  (4246514, 4249810, '+'),
    'gyrA':  (7302, 9818, '+'),
    'gyrB':  (5240, 7267, '+'),
    'pncA':  (2288681, 2289241, '-'),
    'rpsL':  (781560, 781934, '+'),
    'eis':   (2714124, 2715332, '-'),
    'tap':   (1170039, 1170980, '-'),
    'mmpR5': (1446425, 1448896, '-'),
    'mmpL5': (775586, 778480, '-'),
    'tlyA':  (1917940, 1918746, '+'),
    'inhA':  (1674202, 1675011, '+'),
}

REVCOMP_GENES = {g for g, c in RESISTANCE_GENES.items() if c[2] == '-'}

REF_FASTA = REFERENCE_DIR / "H37Rv.fasta"
if not REF_FASTA.exists():
    REF_FASTA = REFERENCE_DIR / "H37Rv.fna"


def load_reference():
    ref = SeqIO.to_dict(SeqIO.parse(str(REF_FASTA), "fasta"))
    chrom_id = list(ref.keys())[0]
    chrom = str(ref[chrom_id].seq)
    gene_seqs = {}
    for gene, (start, end, strand) in RESISTANCE_GENES.items():
        seq = chrom[start-1:end]
        if strand == '-':
            seq = str(Seq(seq).reverse_complement())
        gene_seqs[gene] = seq
    return chrom, gene_seqs


def extract_assembly_genes(assembly_path, ref_gene_seqs):
    """Extract resistance gene sequences from an assembly by 
    finding the exact match for each gene (TB is highly conserved)"""
    try:
        record = SeqIO.read(str(assembly_path), "fasta")
    except:
        # Multi-contig assembly
        records = list(SeqIO.parse(str(assembly_path), "fasta"))
        record = records[0]  # Use first/largest contig
        for r in records:
            if len(r) > len(record):
                record = r
    
    assembly_seq = str(record.seq)
    results = {}
    
    for gene, ref_seq in ref_gene_seqs.items():
        ref_subseq = ref_seq[:100]  # Use first 100bp as anchor
        pos = assembly_seq.find(ref_subseq)
        if pos == -1:
            results[gene] = None
            continue
        
        # Extract the gene at the same position
        gene_seq = assembly_seq[pos:pos+len(ref_seq)]
        if len(gene_seq) < len(ref_seq):
            results[gene] = None
            continue
        
        results[gene] = gene_seq
    
    return results


def compute_homoplasy(genome_dir, ref_gene_seqs):
    """Compute homoplasy counts: For each residue in each gene,
    count how many genomes have a non-synonymous mutation"""
    
    genome_files = sorted(genome_dir.glob("*.fasta"))
    print(f"Found {len(genome_files)} genome assemblies")
    
    # For each gene, for each position: count mutations per residue
    mut_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # mut_counts[gene][position][alt_aa] = count
    total_genomes = 0
    
    for gf in genome_files:
        try:
            gene_seqs = extract_assembly_genes(gf, ref_gene_seqs)
        except Exception as e:
            print(f"  Error processing {gf.name}: {e}")
            continue
        
        valid = sum(1 for v in gene_seqs.values() if v is not None)
        if valid < len(RESISTANCE_GENES) * 0.5:  # Need at least 50% of genes
            print(f"  Skipping {gf.name}: only {valid}/{len(RESISTANCE_GENES)} genes found")
            continue
        
        total_genomes += 1
        
        for gene, ref_seq in ref_gene_seqs.items():
            asm_seq = gene_seqs.get(gene)
            if asm_seq is None:
                continue
            
            # Translate both to protein
            # Handle length differences
            min_len = min(len(ref_seq), len(asm_seq))
            ref_seq_t = ref_seq[:min_len]
            asm_seq_t = asm_seq[:min_len]
            
            ref_prot = str(Seq(ref_seq_t).translate())
            asm_prot = str(Seq(asm_seq_t).translate())
            
            for i, (raa, aaa) in enumerate(zip(ref_prot, asm_prot)):
                if raa != aaa and raa != '*' and aaa != '*':
                    pos = i + 1
                    mut_counts[gene][pos][aaa] += 1
        
        if total_genomes % 10 == 0:
            print(f"  Processed {total_genomes} genomes...")
    
    print(f"\nTotal genomes processed: {total_genomes}")
    return mut_counts, total_genomes


def build_homoplasy_features(mut_counts, total_genomes):
    """Build homoplasy features matching pipeline format"""
    features = []
    
    for gene, positions in mut_counts.items():
        gene_start = RESISTANCE_GENES[gene][0]
        gene_end = RESISTANCE_GENES[gene][1]
        gene_len = (gene_end - gene_start + 1) // 3
        
        for pos in range(1, gene_len + 1):
            pos_data = positions.get(pos, {})
            n_alts = len(pos_data)
            total_alt_count = sum(pos_data.values())
            
            features.append({
                'gene': gene,
                'residue_pos': pos,
                'homoplasy_count': total_alt_count,
                'homoplasy_alleles': n_alts,
                'n_genomes': total_genomes,
                'alt_alleles': ','.join(f"{aa}:{c}" for aa, c in sorted(pos_data.items())),
            })
    
    return features


def merge_with_existing(new_features, existing_vcf_count=117):
    """Merge new assembly-based homoplasy with existing VCF-based counts"""
    # For now, we'll create a combined count
    # In the future, this should merge with the VCF-based homoplasy
    print(f"\nMerging with existing VCF homoplasy ({existing_vcf_count} samples)...")
    
    # Load existing homoplasy data if available
    existing_path = OUTPUT_DIR / "homoplasy_from_assemblies.csv"
    
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['gene','residue_pos','homoplasy_count','homoplasy_alleles','n_genomes','alt_alleles'])
    for feat in new_features:
        w.writerow([feat['gene'], feat['residue_pos'], feat['homoplasy_count'], feat['homoplasy_alleles'], feat['n_genomes'], feat['alt_alleles']])
    return buf.getvalue()


if __name__ == "__main__":
    print("=" * 60)
    print("Homoplasy Computation from Assembled Genomes")
    print("=" * 60)
    
    chrom, ref_gene_seqs = load_reference()
    print(f"Reference loaded: {len(ref_gene_seqs)} genes")
    for g, s in ref_gene_seqs.items():
        print(f"  {g}: {len(s)} bp")
    
    mut_counts, total_genomes = compute_homoplasy(GENOME_DIR, ref_gene_seqs)
    
    features = build_homoplasy_features(mut_counts, total_genomes)
    print(f"\nGenerated features for {len(features)} residues")
    
    output_csv = merge_with_existing(features)
    
    out_path = OUTPUT_DIR / "homoplasy_from_assemblies.csv"
    with open(out_path, "w") as f:
        f.write(output_csv)
    print(f"\nSaved to {out_path}")
    
    # Print summary statistics
    gene_counts = defaultdict(int)
    for feat in features:
        if feat['homoplasy_count'] > 0:
            gene_counts[feat['gene']] += 1
    
    print("\nResidues with at least 1 mutation across all genomes:")
    for gene, count in sorted(gene_counts.items()):
        print(f"  {gene}: {count} residues with mutations")
    
    total_mutated = sum(1 for f in features if f['homoplasy_count'] > 0)
    print(f"\n  Total: {total_mutated}/{len(features)} residues")
    
    # Top mutations by count
    sorted_feats = sorted(features, key=lambda x: -x['homoplasy_count'])
    print("\nTop 30 most mutated residues:")
    for f in sorted_feats[:30]:
        if f['homoplasy_count'] > 0:
            print(f"  {f['gene']} {f['residue_pos']}: {f['homoplasy_count']} carriers, {f['homoplasy_alleles']} alleles [{f['alt_alleles'][:50]}]")
