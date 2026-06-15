"""Summarize CRyPTIC phenotype data"""
import csv
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent.parent
META_DIR = PROJECT_DIR / "data" / "metadata"

pheno_path = META_DIR / "cryptic_phenotypes.csv"
with open(pheno_path, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total samples: {len(rows)}")
print()

# Count resistances for key drugs
drugs = ["RIF", "INH", "EMB", "MDR", "AMI", "MXF", "BDQ", "KAN", "LEV", "LZD"]
drug_columns = {d: f"{d}_BINARY_PHENOTYPE" for d in drugs}

print("=== Resistance counts per drug ===")
for drug, col in drug_columns.items():
    counts = Counter(r.get(col, "NA") for r in rows)
    resistant = counts.get("R", 0)
    susceptible = counts.get("S", 0)
    unknown = counts.get("NA", 0) + counts.get("NA", 0)
    print(f"  {drug}: {resistant} R / {susceptible} S / {unknown} NA")

# Count MDR (resistant to at least RIF + INH)
mdr_count = 0
for r in rows:
    rif = r.get("RIF_BINARY_PHENOTYPE", "NA")
    inh = r.get("INH_BINARY_PHENOTYPE", "NA")
    if rif == "R" and inh == "R":
        mdr_count += 1
print(f"\nMDR-TB (RIF+R resistant): {mdr_count}")

# Find samples with VCF files
with_vcf = sum(1 for r in rows if r.get("VCF", "").strip())
print(f"Samples with VCF files: {with_vcf}")

# Show unique sample: resistant to at least one drug
any_resistant = 0
all_susceptible = 0
for r in rows:
    is_r = any(r.get(f"{d}_BINARY_PHENOTYPE", "NA") == "R" for d in ["RIF", "INH", "EMB", "AMI", "MXF", "BDQ", "KAN", "LEV", "LZD"])
    if is_r:
        any_resistant += 1
    else:
        all_susceptible += 1
print(f"\nAny drug resistance: {any_resistant}")
print(f"All susceptible: {all_susceptible}")

# Show first 5 samples with phenotype summary
print("\n=== First 10 samples ===")
for r in rows[:10]:
    phenos = "".join(r.get(f"{d}_BINARY_PHENOTYPE", "?") for d in ["RIF", "INH", "EMB", "MXF"])
    print(f"  {r['ENA_RUN']} | {r['UNIQUEID'][:40]} | {phenos}")
