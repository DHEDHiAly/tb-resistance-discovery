"""
Robust seed-based homoplasy using 30bp seeds + best-match verification.
Finds ALL seed occurrences, extends to verify, picks the best match.
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

SEED_LEN = 30
VERIFY_LEN = 120  # verify first 120bp of extracted gene


def extract_gene_seq(chrom, start, end, strand):
    seq = chrom[start-1:end]
    if strand == '-':
        seq = str(Seq(seq).reverse_complement())
    return seq


def find_gene(assembly_seq, ref_gene_seq, gene_name):
    """Find the best position for a gene in assembly. Returns (pos, score) or None."""
    seed = ref_gene_seq[:SEED_LEN]
    verify_target = ref_gene_seq[:VERIFY_LEN]
    
    candidates = []
    
    # Search forward strand
    start = 0
    while True:
        pos = assembly_seq.find(seed, start)
        if pos == -1:
            break
        candidates.append(('fwd', pos))
        start = pos + 1
    
    # Search revcomp strand
    rev_seed = str(Seq(seed).reverse_complement())
    start = 0
    while True:
        pos = assembly_seq.find(rev_seed, start)
        if pos == -1:
            break
        candidates.append(('rev', pos))
        start = pos + 1
    
    if not candidates:
        return None
    
    best = None
    best_score = -1
    
    for strand, pos in candidates:
        end_pos = pos + VERIFY_LEN
        if end_pos > len(assembly_seq):
            continue
        
        extracted = assembly_seq[pos:end_pos]
        matches = sum(1 for a, b in zip(extracted, verify_target) if a == b)
        
        if matches > best_score:
            best_score = matches
            best = (strand, pos)
    
    # Require at least VERIFY_LEN-5 matches for verification
    if best_score >= VERIFY_LEN - 5:
        return best
    
    return None


def main():
    # Load reference and extract genes
    ref = SeqIO.to_dict(SeqIO.parse(str(REF_FASTA), "fasta"))
    chrom_id = list(ref.keys())[0]
    chrom = str(ref[chrom_id].seq)
    
    ref_gene_seqs = {}
    for gene, (start, end, strand) in RESISTANCE_GENES.items():
        seq = extract_gene_seq(chrom, start, end, strand)
        ref_gene_seqs[gene] = seq
    
    print(f"Reference loaded: {len(ref_gene_seqs)} genes")
    
    genome_files = sorted(GENOME_DIR.glob("*.fasta"))
    print(f"Found {len(genome_files)} genome assemblies")
    
    mut_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    total_genes_found = 0
    total_possible = 0
    
    for gf in genome_files:
        try:
            records = list(SeqIO.parse(str(gf), "fasta"))
        except:
            continue
        
        assembly_seq = ''.join(str(r.seq) for r in records).upper()
        
        genes_found = 0
        for gene, ref_seq in ref_gene_seqs.items():
            total_possible += 1
            result = find_gene(assembly_seq, ref_seq, gene)
            if result is None:
                continue
            
            strand, pos = result
            extracted = assembly_seq[pos:pos+len(ref_seq)]
            genes_found += 1
            total_genes_found += 1
            
            # Translate and compare
            ref_prot = str(Seq(ref_seq).translate())
            asm_prot = str(Seq(extracted).translate())
            
            min_len = min(len(ref_prot), len(asm_prot))
            for i in range(min_len):
                if ref_prot[i] != asm_prot[i] and ref_prot[i] != '*' and asm_prot[i] != '*':
                    pos_res = i + 1
                    mut_counts[gene][pos_res][asm_prot[i]] += 1
    
    n_processed = total_possible // len(ref_gene_seqs)
    print(f"Total genomes processed: {n_processed}")
    print(f"Genes found: {total_genes_found}/{total_possible} ({100*total_genes_found/total_possible:.1f}%)")
    
    # Build output
    features = []
    for gene, positions in mut_counts.items():
        gene_len = len(ref_gene_seqs[gene]) // 3
        for pos in range(1, gene_len + 1):
            pos_data = positions.get(pos, {})
            features.append({
                'gene': gene,
                'residue_pos': pos,
                'homoplasy_count': sum(pos_data.values()),
                'homoplasy_alleles': len(pos_data),
                'n_genomes': n_processed,
                'alt_alleles': ','.join(f"{aa}:{c}" for aa, c in sorted(pos_data.items())),
            })
    
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
    
    gene_summary = defaultdict(int)
    for feat in features:
        if feat['homoplasy_count'] > 0:
            gene_summary[feat['gene']] += 1
    
    total_mutated = sum(1 for f in features if f['homoplasy_count'] > 0)
    print(f"Total residues: {len(features)}, mutated: {total_mutated}")
    for g, c in sorted(gene_summary.items()):
        print(f"  {g}: {c}")
    
    sorted_feats = sorted(features, key=lambda x: -x['homoplasy_count'])
    print("\nTop 30:")
    for f in sorted_feats[:30]:
        if f['homoplasy_count'] > 0:
            print(f"  {f['gene']} {f['residue_pos']}: {f['homoplasy_count']} carriers [{f['alt_alleles'][:70]}]")


if __name__ == "__main__":
    main()
