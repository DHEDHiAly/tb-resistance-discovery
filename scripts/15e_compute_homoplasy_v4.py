"""
Robust homoplasy v4: k-mer voting approach.
Splits each reference gene into 20bp seeds (stride 5bp), finds exact matches
in assembly via hash table, clusters votes to find true gene start.
Robust to sequence variation at the gene start.
"""

import os, sys, csv, io, time
from pathlib import Path
from collections import defaultdict, Counter

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

KMER_LEN = 20
KMER_STRIDE = 5


def extract_gene_seq(chrom, start, end, strand):
    seq = chrom[start-1:end]
    if strand == '-':
        seq = str(Seq(seq).reverse_complement())
    return seq


def build_kmer_map(seq, k=KMER_LEN):
    """Build map of kmer positions (first occurrence only, memory efficient)."""
    kmap = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer not in kmap:
            kmap[kmer] = i
    return kmap


def find_gene_via_voting(assembly_seq, kmap, ref_seq):
    """Find gene position using k-mer voting (first-occurrence map)."""
    gene_len = len(ref_seq)
    n_kmers = (gene_len - KMER_LEN) // KMER_STRIDE + 1
    votes = Counter()
    
    for ki in range(n_kmers):
        kstart = ki * KMER_STRIDE
        kmer = ref_seq[kstart:kstart+KMER_LEN]
        rev_kmer = str(Seq(kmer).reverse_complement())
        
        pos = kmap.get(kmer)
        if pos is not None:
            votes[('fwd', pos - kstart)] += 1
        
        pos = kmap.get(rev_kmer)
        if pos is not None:
            gene_start_vote = pos + kstart + KMER_LEN - gene_len
            votes[('rev', gene_start_vote)] += 1
    
    if not votes:
        return None, 0.0
    
    (strand_type, start_vote), vote_count = votes.most_common(1)[0]
    
    def try_extract(start_pos, is_revcomp):
        if start_pos < 0 or start_pos + gene_len > len(assembly_seq):
            return None, 0.0
        raw = assembly_seq[start_pos:start_pos+gene_len]
        if is_revcomp:
            extracted = str(Seq(raw).reverse_complement())
        else:
            extracted = raw
        score = sum(1 for a, b in zip(extracted, ref_seq) if a == b) / gene_len
        return extracted, score
    
    is_rev = (strand_type == 'rev')
    extracted, score = try_extract(start_vote, is_rev)
    
    # If score is low, try nearby offsets (gene start may vary by a few codons)
    if score < 0.70:
        best_extracted = extracted
        best_score = score
        for offset in range(-90, 91):
            if offset == 0:
                continue
            ext, s = try_extract(start_vote + offset, is_rev)
            if ext is not None and s > best_score:
                best_score = s
                best_extracted = ext
        extracted = best_extracted
        score = best_score
    
    return extracted, score


def main():
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
    
    t0 = time.time()
    for gi, gf in enumerate(genome_files):
        try:
            records = list(SeqIO.parse(str(gf), "fasta"))
        except:
            continue
        
        assembly_seq = ''.join(str(r.seq) for r in records).upper()
        
        # Build k-mer map ONCE per assembly
        kmap = build_kmer_map(assembly_seq, KMER_LEN)
        
        genes_found = 0
        for gene, ref_seq in ref_gene_seqs.items():
            total_possible += 1
            
            # Search for gene using k-mer voting
            extracted, score = find_gene_via_voting(assembly_seq, kmap, ref_seq)
            
            if extracted is None or score < 0.95:
                continue
            
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
        
        del kmap, assembly_seq
        elapsed = time.time() - t0
        rate = (gi + 1) / elapsed if elapsed > 0 else 0
        eta = (len(genome_files) - gi - 1) / rate if rate > 0 else 0
        if (gi + 1) % 20 == 0 or (gi + 1) == len(genome_files):
            if eta < 3600:
                print(f"  {gi+1}/{len(genome_files)} ({100*(gi+1)/len(genome_files):.0f}%) - {genes_found}/13 genes, {rate:.1f}/s ETA {int(eta/60)}min")
            else:
                print(f"  {gi+1}/{len(genome_files)} ({100*(gi+1)/len(genome_files):.0f}%) - {genes_found}/13 genes, {rate:.1f}/s ETA {eta/3600:.1f}h")
    
    n_processed = len(genome_files)
    print(f"\nTotal genomes processed: {n_processed}")
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
