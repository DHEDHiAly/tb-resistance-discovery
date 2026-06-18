from Bio import SeqIO
import os

genome_dir = 'data/genomes'

# Find which assemblies actually have the rpoB gene by trying to find 
# multiple short conserved sequences

# rpoB is highly conserved in TB complex
# Try just the first 40bp of rpoB from H37Rv
rpoB_seed = 'GTGGCTTCGATGGAGCGGCTGGAGGGCTTCAACGTCGACAAC'

for fname in sorted(os.listdir(genome_dir)):
     if not fname.endswith('.fasta'):
         continue
     fpath = os.path.join(genome_dir, fname)
     try:
         records = list(SeqIO.parse(fpath, 'fasta'))
         full_seq = ''.join(str(r.seq) for r in records)
         pos = full_seq.find(rpoB_seed)
         if pos != -1:
             print(f'{fname:50s} CONTAINS rpoB at {pos}')
     except:
         pass

print()
print('Genomes without rpoB seed:')
for fname in sorted(os.listdir(genome_dir)):
     if not fname.endswith('.fasta'):
         continue
     fpath = os.path.join(genome_dir, fname)
     try:
         records = list(SeqIO.parse(fpath, 'fasta'))
         full_seq = ''.join(str(r.seq) for r in records)
         pos = full_seq.find(rpoB_seed)
         if pos == -1:
             print(f'  {fname}')
     except:
         print(f'  {fname}: ERROR')
