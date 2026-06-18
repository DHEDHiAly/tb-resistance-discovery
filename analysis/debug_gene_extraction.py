"""Debug: verify gene extraction for rpoB and key mutations"""
from pathlib import Path
from Bio import SeqIO, Seq
from collections import defaultdict

PROJECT_DIR = Path(__file__).resolve().parent.parent
GENOME_DIR = PROJECT_DIR / "data" / "genomes"
REF_FASTA = PROJECT_DIR / "reference" / "H37Rv.fasta"
if not REF_FASTA.exists():
    REF_FASTA = PROJECT_DIR / "reference" / "H37Rv.fna"

RESISTANCE_GENES = {
    'rpoB':  (759807, 763325, '+'),
    'embB':  (4246514, 4249810, '+'),
    'katG':  (2153889, 2156111, '-'),
}

def extract_gene_seq(chrom, start, end, strand):
    seq = chrom[start-1:end]
    if strand == '-':
        seq = str(Seq.Seq(seq).reverse_complement())
    return seq

ref = SeqIO.to_dict(SeqIO.parse(str(REF_FASTA), "fasta"))
chrom_id = list(ref.keys())[0]
chrom = str(ref[chrom_id].seq)

# Extract reference gene seqs
ref_gene_seqs = {}
for gene, (start, end, strand) in RESISTANCE_GENES.items():
    seq = extract_gene_seq(chrom, start, end, strand)
    ref_gene_seqs[gene] = seq
    print(f"{gene}: {len(seq)}bp, first 50bp: {seq[:50]}")
    print(f"  Translation 1-50: {Seq.Seq(seq).translate()[:30]}")
    print(f"  Translation (frame +1): {Seq.Seq(seq[1:]).translate()[:30]}")
    print(f"  Translation (frame +2): {Seq.Seq(seq[2:]).translate()[:30]}")

print("\n--- Checking assemblies ---")
genome_files = sorted(GENOME_DIR.glob("*.fasta"))

for gf in list(genome_files)[:5]:
    records = list(SeqIO.parse(str(gf), "fasta"))
    assembly_seq = ''.join(str(r.seq) for r in records).upper()
    
    print(f"\n=== {gf.name} ===")
    
    for gene, ref_seq in ref_gene_seqs.items():
        seed = ref_seq[:20]
        rev_seed = str(Seq.Seq(seed).reverse_complement())
        
        # Find all occurrences of seed
        pos = assembly_seq.find(seed)
        rev_pos = assembly_seq.find(rev_seed)
        
        if pos != -1:
            extracted = assembly_seq[pos:pos+len(ref_seq)]
            prot = str(Seq.Seq(extracted).translate())
            ref_prot = str(Seq.Seq(ref_seq).translate())
            # Count matching positions
            matches = sum(1 for a,b in zip(prot, ref_prot) if a==b)
            print(f"  {gene} (forward strand, pos {pos}): {matches}/{min(len(prot), len(ref_prot))} matching residues")
            print(f"    First 20nt: {extracted[:20]}")
            print(f"    First 10aa: {prot[:10]} (ref: {ref_prot[:10]})")
        elif rev_pos != -1:
            extracted = assembly_seq[rev_pos:rev_pos+len(ref_seq)]
            prot = str(Seq.Seq(extracted).translate())
            ref_prot = str(Seq.Seq(ref_seq).translate())
            matches = sum(1 for a,b in zip(prot, ref_prot) if a==b)
            print(f"  {gene} (revcomp strand, pos {rev_pos}): {matches}/{min(len(prot), len(ref_prot))} matching residues")
        else:
            print(f"  {gene}: NOT FOUND")
