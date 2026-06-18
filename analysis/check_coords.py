from Bio import SeqIO
import os

genome_dir = 'data/genomes'

# Check what happens with the original coordinates vs new coordinates
# The key gene that changed was inhA: 5423122 -> 1674202
# Let's see which coordinate system the assemblies use

# Test assemblies that worked before vs now
test_files = ['GCF_000023625.1.fasta', 'GCF_040208985.1.fasta',  # Old genomes
              'GCF_020684585.1.fasta', 'GCF_900654255.2.fasta']    # New downloads

for fname in test_files:
    fpath = os.path.join(genome_dir, fname)
    if not os.path.exists(fpath):
        continue
    try:
        record = SeqIO.read(fpath, 'fasta')
        seq = str(record.seq)
        print(f'\n=== {fname} ({record.id}, {len(seq)} bp) ===')
        
        # Try both coordinate systems for rpoB
        # rpoB should be around 759807-763325 in H37Rv coordinates
        # Check if we can find the rpoB start
        for anchor_region, label in [
            (seq.find('GTGGCTTCGATGGAGCGGCTGGAGGGCTTCAACGTCGACAACCCGCTGTCG'), 'rpoB N-term'),
            (seq.find('ATGACCGACGAGCACGCCAAGCAGTCC'), 'inhA start (old 5423122)'),
            (seq.find('ATGACCGACGAGCACGCCAAGCAGTCC', 1670000), 'inhA start (H37Rv ~1.67M)'),
            (seq.find('ATGACCGACGAGCACGCCAAGCAGTCC', 5420000), 'inhA start (old ~5.42M)'),
        ]:
            if anchor_region != -1:
                print(f'  FOUND: {label} at position {anchor_region}')
            else:
                print(f'  NOT FOUND: {label}')
                
    except Exception as e:
        print(f'{fname}: {e}')
