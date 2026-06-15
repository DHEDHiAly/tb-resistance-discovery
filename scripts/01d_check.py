"""Check what we have"""
import csv
from pathlib import Path

genomes = list(Path("data/genomes").glob("*.fasta"))
print(f"Total genomes downloaded: {len(genomes)}")
for g in sorted(genomes):
    print(f"  {g.name} ({g.stat().st_size/1e6:.1f} MB)")

print()
meta = Path("data/metadata/all_tb_metadata.csv")
with open(meta, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f"Metadata entries: {len(rows)}")
print(f"Columns: {reader.fieldnames}")
for r in rows[:5]:
    print(f'  {r["assembly_acc"]} | {r["organism"][:50]} | {r["assembly_level"]}')
