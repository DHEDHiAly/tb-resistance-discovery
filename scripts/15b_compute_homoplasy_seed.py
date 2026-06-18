"""
Seed-based homoplasy using k-mer hashing for speed.
Build a set of all seeds (20bp) from each assembly, then look up gene starts.
"""

import os, sys, csv, io
from pathlib import Path
from collections import defaultdict

from Bio import SeqIO
from Bio.Seq import Seq

PROJECT_DIR = Path(__file__).resolve().parent.parent
GENOME_DIR = PROJECT_DIR / "data" / "genomes"
OUTPUT_DIR = PROJECT_DIR / "analysis" / "results" / "homoplasy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REF_FASTA = PROJECT_DIR / "reference" / "H37Rv.fasta"
if not REF_FASTA.exists():
    REF_FASTA = PROJECT_DIR / "reference" / "H37Rv.fna"

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

SEED_LEN = 20  # short seed for better matching across strains


def extract_gene_seq(chrom, start, end, strand):
    seq = chrom[start-1:end]
    if strand == '-':
        seq = str(Seq(seq).reverse_complement())
    return seq


def build_kmer_map(seq, k=SEED_LEN):
    """Build position map: for each k-mer, store its first occurrence."""
    kmap = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer not in kmap:
            kmap[kmer] = i
    return kmap


def find_exact(seq, query):
    """Simple find"""
    return seq.find(query)


def main():
    # Load reference and extract gene sequences
    ref = SeqIO.to_dict(SeqIO.parse(str(REF_FASTA), "fasta"))
    chrom_id = list(ref.keys())[0]
    chrom = str(ref[chrom_id].seq)
    
    ref_gene_seqs = {}
    for gene, (start, end, strand) in RESISTANCE_GENES.items():
        seq = extract_gene_seq(chrom, start, end, strand)
        ref_gene_seqs[gene] = seq
    
    print(f"Reference loaded: {len(ref_gene_seqs)} genes")
    for g, s in ref_gene_seqs.items():
        print(f"  {g}: {len(s)} bp")
    
    # Process each assembly
    genome_files = sorted(GENOME_DIR.glob("*.fasta"))
    print(f"\nFound {len(genome_files)} genome assemblies")
    
    mut_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    n_processed = 0
    n_tried = 0
    
    for gf in genome_files:
        try:
            records = list(SeqIO.parse(str(gf), "fasta"))
        except:
            continue
        
        assembly_seq = ''.join(str(r.seq) for r in records).upper()
        n_tried += 1
        
        # Build k-mer map for this assembly (one pass)
        kmap = build_kmer_map(assembly_seq, SEED_LEN)
        
        genes_found = 0
        for gene, ref_seq in ref_gene_seqs.items():
            seed = ref_seq[:SEED_LEN]
            pos = kmap.get(seed)
            
            if pos is None:
                # Try reverse complement of seed
                rev_seed = str(Seq(seed).reverse_complement())
                pos = kmap.get(rev_seed)
            
            if pos is None:
                continue
            
            # Extract the gene region
            gene_end = pos + len(ref_seq)
            if gene_end > len(assembly_seq):
                continue
            
            gene_seq = assembly_seq[pos:gene_end]
            genes_found += 1
            
            # Translate and compare
            ref_prot = str(Seq(ref_seq).translate())
            asm_prot = str(Seq(gene_seq).translate())
            
            min_len = min(len(ref_prot), len(asm_prot))
            for i in range(min_len):
                if ref_prot[i] != asm_prot[i] and ref_prot[i] != '*' and asm_prot[i] != '*':
                    pos_res = i + 1
                    mut_counts[gene][pos_res][asm_prot[i]] += 1
        
        if genes_found >= len(RESISTANCE_GENES) * 0.5:
            n_processed += 1
        
        if n_processed > 0 and n_processed % 10 == 0:
            print(f"  Processed {n_processed}/{n_tried} genomes...")
    
    print(f"\nTotal genomes processed: {n_processed} of {n_tried} tried")
    
    # Build output features
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
                'n_genomes': n_processed,
                'alt_alleles': ','.join(f"{aa}:{c}" for aa, c in sorted(pos_data.items())),
            })
    
    # Save
    out_path = OUTPUT_DIR / "homoplasy_from_assemblies.csv"
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['gene','residue_pos','homoplasy_count','homoplasy_alleles','n_genomes','alt_alleles'])
    for feat in features:
        w.writerow([feat['gene'], feat['residue_pos'], feat['homoplasy_count'],
                    feat['homoplasy_alleles'], feat['n_genomes'], feat['alt_alleles']])
    with open(out_path, "w") as f:
        f.write(buf.getvalue())
    print(f"Saved to {out_path}")
    
    # Summary
    gene_summary = defaultdict(int)
    for feat in features:
        if feat['homoplasy_count'] > 0:
            gene_summary[feat['gene']] += 1
    
    total_mutated = sum(1 for f in features if f['homoplasy_count'] > 0)
    print(f"Generated features for {len(features)} residues")
    print(f"Residues with at least 1 mutation: {total_mutated}")
    for g, c in sorted(gene_summary.items()):
        print(f"  {g}: {c} residues")
    
    sorted_feats = sorted(features, key=lambda x: -x['homoplasy_count'])
    print("\nTop 30 mutations:")
    for f in sorted_feats[:30]:
        if f['homoplasy_count'] > 0:
            print(f"  {f['gene']} {f['residue_pos']}: {f['homoplasy_count']} carriers [{f['alt_alleles'][:70]}]")


if __name__ == "__main__":
    main()
