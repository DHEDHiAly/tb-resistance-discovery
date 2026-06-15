"""Examine CRyPTIC phenotype data and find genome accessions"""
import csv, json, requests
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
requests.packages.urllib3.disable_warnings()

PROJECT_DIR = Path(__file__).resolve().parent.parent
META_DIR = PROJECT_DIR / "data" / "metadata"

# Read phenotype CSV
pheno_path = META_DIR / "cryptic_phenotypes.csv"
with open(pheno_path, newline="") as f:
    reader = csv.reader(f)
    rows = list(reader)

print(f"=== CRyPTIC Phenotype Table ===")
print(f"Rows: {len(rows)}")
print(f"Columns ({len(rows[0])}):")
for i, col in enumerate(rows[0]):
    print(f"  [{i}] {col}")
print(f"\nFirst data row:")
for i, val in enumerate(rows[1]):
    print(f"  [{i}] {rows[0][i]} = {val[:100] if len(val) > 100 else val}")

print(f"\nSample rows (2-5):")
for r in rows[2:6]:
    print(f"  UniqID: {r[0]}, Drug resistance: {r[3][:80] if len(r) > 3 else 'N/A'}")

# Check if UK BioBank / ERS accessions are present
print(f"\n=== Looking for genome accessions ===")
sample_cols = [i for i, c in enumerate(rows[0]) if "accession" in c.lower() or "genome" in c.lower() or "sra" in c.lower() or "ers" in c.lower() or "srr" in c.lower() or "ena" in c.lower()]
print(f"Columns with 'accession/genome/SRA/ENA': {sample_cols}")
for i in sample_cols:
    print(f"  Col [{i}] {rows[0][i]}: {rows[1][i]}")

# Check reproducibility index JSON
print(f"\n=== Checking reproducibility index ===")
base = "https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022"
index_url = f"{base}/reproducibility/cryptic-index_20231027.json"
r = requests.head(index_url, verify=False)
print(f"Index file: {r.status_code}, Size: {r.headers.get('Content-Length', 'unknown')} bytes")
if r.status_code == 200:
    # Download just first part to see structure
    r = requests.get(index_url, stream=True, verify=False)
    chunk = r.iter_content(chunk_size=50000).__next__()
    print(f"First 5000 chars: {chunk[:5000].decode('utf-8')}")
