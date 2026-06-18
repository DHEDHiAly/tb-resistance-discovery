"""Debug '-' strand gene verification"""
from pathlib import Path
from Bio import SeqIO, Seq

PROJECT_DIR = Path(__file__).resolve().parent.parent
GENOME_DIR = PROJECT_DIR / "data" / "genomes"
REF_FASTA = PROJECT_DIR / "reference" / "H37Rv.fasta"
if not REF_FASTA.exists():
    REF_FASTA = PROJECT_DIR / "reference" / "H37Rv.fna"

RESISTANCE_GENES = {
    'katG':  (2153889, 2156111, '-'),
    'pncA':  (2288681, 2289241, '-'),
    'eis':   (2714124, 2715332, '-'),
    'tap':   (1170039, 1170980, '-'),
    'mmpR5': (1446425, 1448896, '-'),
    'mmpL5': (775586, 778480, '-'),
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
    print(f"{gene}: first 30bp of coding seq = {seq[:30]}")
    print(f"         revcomp of that     = {str(Seq.Seq(seq[:30]).reverse_complement())}")

genome_files = sorted(GENOME_DIR.glob("*.fasta"))

for gf in list(genome_files)[:5]:
    records = list(SeqIO.parse(str(gf), "fasta"))
    assembly_seq = ''.join(str(r.seq) for r in records).upper()
    print(f"\n=== {gf.name} ===")
    
    for gene, (start, end, strand) in RESISTANCE_GENES.items():
        ref_seq = ref_gene_seqs[gene]
        seed = ref_seq[:30]  # coding sequence start
        rev_seed = str(Seq.Seq(seed).reverse_complement())  # what's on forward strand
        
        # Search for rev_seed in assembly
        pos = assembly_seq.find(rev_seed)
        if pos == -1:
            print(f"  {gene}: rev_seed NOT FOUND")
            continue
        
        # Extract forward strand at that position and revcomp to get coding sequence
        extracted_fwd = assembly_seq[pos:pos+len(ref_seq)]
        extracted_coding = str(Seq.Seq(extracted_fwd).reverse_complement())
        
        # Compare coding sequences
        matches = sum(1 for a, b in zip(extracted_coding, ref_seq) if a == b)
        score = matches / len(ref_seq)
        
        # Also try different offsets (-30 to +30)
        best_off = 0
        best_score = score
        for offset in range(-30, 31):
            p = pos + offset
            if p < 0 or p + len(ref_seq) > len(assembly_seq):
                continue
            ef = assembly_seq[p:p+len(ref_seq)]
            ec = str(Seq.Seq(ef).reverse_complement())
            m = sum(1 for a, b in zip(ec, ref_seq) if a == b)
            s = m / len(ref_seq)
            if s > best_score:
                best_score = s
                best_off = offset
        
        print(f"  {gene}: rev_seed pos={pos}, offset0 score={score:.4f}, best_off={best_off} score={best_score:.4f}")
        
        # Show why it fails: first 10 codons
        if score < 0.95:
            ref_prot = str(Seq.Seq(ref_seq).translate())
            asm_prot = str(Seq.Seq(extracted_coding).translate())
            n_match = sum(1 for a,b in zip(ref_prot, asm_prot) if a==b)
            print(f"    protein match: {n_match}/{min(len(ref_prot), len(asm_prot))}")
            print(f"    ref  first 30aa: {ref_prot[:30]}")
            print(f"    asm  first 30aa: {asm_prot[:30]}")
