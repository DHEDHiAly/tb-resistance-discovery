"""Check available tools for MSA/Shannon entropy computation."""
import Bio
print(f"BioPython version: {Bio.__version__}")
print(f"pairwise2: {hasattr(Bio, 'pairwise2')}")
try:
    from Bio import pairwise2
    print("pairwise2 import OK")
except:
    print("pairwise2 import FAILED")

# Check if we can do local multiple alignment
try:
    from Bio.Align import MultipleSeqAlignment
    print("MultipleSeqAlignment available")
except:
    print("MultipleSeqAlignment NOT available")

# Check the genomes
import os
genomes_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "genomes")
if os.path.exists(genomes_dir):
    files = [f for f in os.listdir(genomes_dir) if f.endswith(".fasta")]
    print(f"\nGenomes found: {len(files)}")
    for f in sorted(files):
        path = os.path.join(genomes_dir, f)
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"  {f}: {size_mb:.1f} MB")
else:
    print(f"\nGenomes dir not found at {genomes_dir}")
