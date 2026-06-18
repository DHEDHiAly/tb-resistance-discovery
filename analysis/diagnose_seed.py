"""Diagnose why genes are missed in assemblies"""
from pathlib import Path
from Bio import SeqIO, Seq

PROJECT_DIR = Path(__file__).resolve().parent.parent
GENOME_DIR = PROJECT_DIR / "data" / "genomes"
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

def extract_gene_seq(chrom, start, end, strand):
    seq = chrom[start-1:end]
    if strand == '-':
        seq = str(Seq.Seq(seq).reverse_complement())
    return seq

ref = SeqIO.to_dict(SeqIO.parse(str(REF_FASTA), "fasta"))
chrom_id = list(ref.keys())[0]
chrom = str(ref[chrom_id].seq)

ref_gene_seqs = {}
for gene, (start, end, strand) in RESISTANCE_GENES.items():
    seq = extract_gene_seq(chrom, start, end, strand)
    ref_gene_seqs[gene] = seq

genome_files = sorted(GENOME_DIR.glob("*.fasta"))

# For each gene, check how many assemblies have the seed vs pass verification
for gene, ref_seq in ref_gene_seqs.items():
    seed = ref_seq[:30]
    verify = ref_seq[:120]
    
    n_seed_match = 0
    n_verify_pass = 0
    
    for gf in genome_files:
        try:
            records = list(SeqIO.parse(str(gf), "fasta"))
        except:
            continue
        assembly_seq = ''.join(str(r.seq) for r in records).upper()
        
        # Check forward
        pos = assembly_seq.find(seed)
        if pos != -1:
            n_seed_match += 1
            ext = assembly_seq[pos:pos+120]
            matches = sum(1 for a,b in zip(ext, verify) if a == b)
            if matches >= 115:  # 120-5 threshold
                n_verify_pass += 1
            continue
        
        # Check revcomp
        rev_seed = str(Seq.Seq(seed).reverse_complement())
        pos = assembly_seq.find(rev_seed)
        if pos != -1:
            n_seed_match += 1
            ext = assembly_seq[pos:pos+120]
            matches = sum(1 for a,b in zip(ext, verify) if a == b)
            if matches >= 115:
                n_verify_pass += 1
    
    print(f"{gene:8s}: seed matches {n_seed_match:2d}/60, passes verify {n_verify_pass:2d}/60")
