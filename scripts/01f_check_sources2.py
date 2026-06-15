"""Check response content and try more data sources"""
import requests
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
requests.packages.urllib3.disable_warnings()

PROJECT_DIR = Path(__file__).resolve().parent.parent
META_DIR = PROJECT_DIR / "data" / "metadata"

# Check TB Portal raw response
url = "https://tbportals.niaid.nih.gov/api/v1/isolates?limit=5"
r = requests.get(url, verify=False, timeout=30)
print(f"=== TB Portal API ===")
print(f"URL: {url}")
print(f"Status: {r.status_code}")
print(f"Headers: {dict(r.headers)}")
print(f"Content (first 2000 chars):")
print(r.text[:2000])

# Try CRyPTIC - correct URL
print(f"\n=== CRyPTIC (corrected) ===")
for u in [
    "https://ftp.ebi.ac.uk/pub/databases/cryptic/",
    "https://ftp.ebi.ac.uk/pub/software/cryptic/",
]:
    r = requests.get(u, verify=False, timeout=30)
    print(f"{u} -> {r.status_code}")

# BV-BRC / PATRIC API
print(f"\n=== BV-BRC API ===")
bvbrc_url = "https://www.bv-brc.org/api/genomes/?eq(taxon_lineage_names,Mycobacterium tuberculosis)&select(genome_id,genome_name,antibiotic_resistances)&limit(5)"
r = requests.get(bvbrc_url, verify=False, timeout=30)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    print(f"Content (first 2000 chars): {r.text[:2000]}")
