"""Deep debug: why '-' strand voting finds wrong position"""
from pathlib import Path
from collections import Counter, defaultdict
from Bio import SeqIO, Seq

PROJECT_DIR = Path(__file__).resolve().parent.parent
GENOME_DIR = PROJECT_DIR / "data" / "genomes"
REF_FASTA = PROJECT_DIR / "reference" / "H37Rv.fasta"
if not REF_FASTA.exists():
    REF_FASTA = PROJECT_DIR / "reference" / "H37Rv.fna"

KMER_LEN = 20
KMER_STRIDE = 5

ref = SeqIO.to_dict(SeqIO.parse(str(REF_FASTA), "fasta"))
chrom_id = list(ref.keys())[0]
chrom = str(ref[chrom_id].seq)

# Focus on katG (reverse strand)
start, end = 2153889, 2156111
katG_coding = str(Seq.Seq(chrom[start-1:end]).reverse_complement())
print(f"katG coding len: {len(katG_coding)}")
print(f"katG coding first 30bp: {katG_coding[:30]}")
print(f"katG coding revcomp (forward strand start): {str(Seq.Seq(katG_coding[:30]).reverse_complement())}")

# Check where this revcomp seed is in the reference forward strand
rev30 = str(Seq.Seq(katG_coding[:30]).reverse_complement())
print(f"\nSearching rev30 in reference forward strand...")
pos = chrom.find(rev30)
print(f"Found at reference position: {pos} (expected: {start})")

# Also check each k-mer from the coding sequence
# For coding k-mer at offset Ki*5: revcomp should be at start + Ki*5 in forward strand
print(f"\nk-mer matching analysis (first 50 k-mers, stride={KMER_STRIDE}):")
n_kmers = (len(katG_coding) - KMER_LEN) // KMER_STRIDE + 1

kmer_positions = []
for ki in range(min(n_kmers, 100)):
    kstart = ki * KMER_STRIDE
    kmer = katG_coding[kstart:kstart+KMER_LEN]
    rev_kmer = str(Seq.Seq(kmer).reverse_complement())
    
    # Expected position in reference forward strand
    exp_pos = start + kstart  # The revcomp should be here
    
    # Check if rev_kmer actually appears here in reference
    actual_rev_pos = chrom.find(rev_kmer)
    
    # Also check if the kmer itself appears anywhere (coding seq in forward - shouldn't)
    actual_fwd_pos = chrom.find(kmer)
    
    # Check nearby positions
    nearby_match = chrom[exp_pos-2:exp_pos+KMER_LEN+2].find(rev_kmer)
    
    kmer_positions.append({
        'ki': ki,
        'kstart': kstart,
        'rev_kmer': rev_kmer,
        'exp_pos': exp_pos,
        'nearby': nearby_match == 2,
        'actual_rev_pos': actual_rev_pos,
        'actual_fwd_pos': actual_fwd_pos,
        'at_expected': chrom[exp_pos:exp_pos+KMER_LEN] == rev_kmer,
    })

# Count how many match at expected position
at_expected = sum(1 for kp in kmer_positions if kp['at_expected'])
exact_at_expected = sum(1 for kp in kmer_positions if kp['actual_rev_pos'] == kp['exp_pos'])
print(f"  k-mers with revcomp at expected position: {at_expected}/{len(kmer_positions)}")
print(f"  k-mers with revcomp exactly at expected: {exact_at_expected}/{len(kmer_positions)}")

# For those NOT at expected, where are they?
print(f"\n  K-mers NOT at expected position:")
for kp in kmer_positions:
    if not kp['at_expected'] and kp['actual_rev_pos'] != -1:
        print(f"    k-mer {kp['ki']} (offset {kp['kstart']}): expected {kp['exp_pos']}, found at {kp['actual_rev_pos']} (delta={kp['actual_rev_pos']-kp['exp_pos']})")

# Now check the actual assembly
print(f"\n\n=== Checking assembly ===")
for gf in sorted(GENOME_DIR.glob("*.fasta"))[:2]:
    records = list(SeqIO.parse(str(gf), "fasta"))
    assembly_seq = ''.join(str(r.seq) for r in records).upper()
    
    # Check if the expected k-mer positions match in the assembly
    print(f"\n{gf.name}:")
    
    rev_matches_at_expected = 0
    rev_matches_total = 0
    
    for ki in range(min(n_kmers, 100)):
        kstart = ki * KMER_STRIDE
        kmer = katG_coding[kstart:kstart+KMER_LEN]
        rev_kmer = str(Seq.Seq(kmer).reverse_complement())
        
        asm_pos = assembly_seq.find(rev_kmer)
        if asm_pos != -1:
            rev_matches_total += 1
            exp_pos = start + kstart
            if abs(asm_pos - exp_pos) < 5:
                rev_matches_at_expected += 1
    
    print(f"  Revcomp k-mers found: {rev_matches_total}/{len(kmer_positions)}")
    print(f"  At expected positions: {rev_matches_at_expected}/{len(kmer_positions)}")
