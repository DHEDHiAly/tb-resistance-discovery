from Bio import SeqIO

# Check assemblies that worked vs failed
import os

genome_dir = 'data/genomes'

# Test one that worked and one that failed
for fname in ['GCF_000023625.1.fasta', 'GCF_020684585.1.fasta', 'GCF_040208985.1.fasta']:
    fpath = os.path.join(genome_dir, fname)
    if not os.path.exists(fpath):
        print(f'{fname}: NOT FOUND')
        continue
    
    try:
        records = list(SeqIO.parse(fpath, 'fasta'))
        print(f'{fname}: {len(records)} contigs')
        for i, r in enumerate(records[:3]):
            print(f'  [{i}] {r.id}: {len(r.seq)} bp - starts: {str(r.seq[:80])}')
        
        # Try to find rpoB start sequence
        rpoB_start = 'GTGGCTTCGATGGAGCGGCTGGAGGGCTTCAACGTCGACAACCCGCTGTCG'
        first_contig = str(records[0].seq)
        for i, r in enumerate(records):
            s = str(r.seq)
            pos = s.find(rpoB_start[:40])
            if pos != -1:
                print(f'  rpoB found in contig [{i}] at position {pos}')
                break
        else:
            print(f'  rpoB NOT FOUND in any contig')
    except Exception as e:
        print(f'{fname}: ERROR: {e}')
    print()
