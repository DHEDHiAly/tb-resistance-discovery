"""Debug k-mer voting for katG"""
from pathlib import Path
from collections import Counter
from Bio import SeqIO, Seq

PROJECT_DIR = Path(__file__).resolve().parent.parent
GENOME_DIR = PROJECT_DIR / "data" / "genomes"
REF_FASTA = PROJECT_DIR / "reference" / "H37Rv.fasta"
if not REF_FASTA.exists():
    REF_FASTA = PROJECT_DIR / "reference" / "H37Rv.fna"

RESISTANCE_GENES = {
    'katG': (2153889, 2156111, '-'),
    'embB': (4246514, 4249810, '+'),
}

KMER_LEN = 20
KMER_STRIDE = 5

def extract_gene_seq(chrom, start, end, strand):
    seq = chrom[start-1:end]
    if strand == '-':
        seq = str(Seq.Seq(seq).reverse_complement())
    return seq

def build_kmer_map(seq, k=KMER_LEN):
    kmap = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer not in kmap:
            kmap[kmer] = i
    return kmap

ref = SeqIO.to_dict(SeqIO.parse(str(REF_FASTA), "fasta"))
chrom_id = list(ref.keys())[0]
chrom = str(ref[chrom_id].seq)

for gf in sorted(GENOME_DIR.glob("*.fasta"))[:3]:
    records = list(SeqIO.parse(str(gf), "fasta"))
    assembly_seq = ''.join(str(r.seq) for r in records).upper()
    kmap = build_kmer_map(assembly_seq, KMER_LEN)
    
    print(f"\n=== {gf.name} ===")
    
    for gene, (start, end, strand) in RESISTANCE_GENES.items():
        ref_seq = extract_gene_seq(chrom, start, end, strand)
        gene_len = len(ref_seq)
        n_kmers = (gene_len - KMER_LEN) // KMER_STRIDE + 1
        
        votes = Counter()
        for ki in range(min(n_kmers, 100)):
            kstart = ki * KMER_STRIDE
            kmer = ref_seq[kstart:kstart+KMER_LEN]
            rev_kmer = str(Seq.Seq(kmer).reverse_complement())
            
            pos = kmap.get(kmer)
            if pos is not None:
                votes[('fwd', pos - kstart)] += 1
            
            pos = kmap.get(rev_kmer)
            if pos is not None:
                votes[('rev', pos - kstart)] += 1
        
        print(f"\n  {gene} ({strand}): {len(votes)} unique vote positions from {n_kmers} kmers")
        
        for (st, pos), cnt in votes.most_common(5):
            print(f"    {st} pos={pos}: {cnt} votes ({100*cnt/n_kmers:.0f}%)")
        
        # Compare with expected position
        if strand == '+':
            expected_pos = start
            expected_type = 'fwd'
        else:
            expected_pos = start  # forward strand position of gene start
            expected_type = 'rev'
        
        print(f"    Expected: {expected_type} pos={expected_pos}")
        
        # Check if expected position has votes
        if strand == '+':
            exp_votes = votes.get(('fwd', expected_pos), 0)
        else:
            exp_votes = votes.get(('rev', expected_pos), 0)
        print(f"    Votes at expected position: {exp_votes}")
