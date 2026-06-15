"""Try TB Portal API for resistance-labelled genomes"""
import requests
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
requests.packages.urllib3.disable_warnings()

PROJECT_DIR = Path(__file__).resolve().parent.parent
META_DIR = PROJECT_DIR / "data" / "metadata"

# TB Portal API - NIAID/NIH
print("=== TB Portal (NIAID) - Drug Resistance Dataset ===")
print()

# Method 1: Try their public API
urls = [
    "https://tbportals.niaid.nih.gov/api/v1/isolates?limit=10",
    "https://tbportals.niaid.nih.gov/api/isolates?limit=10",
    "https://api.tbportals.niaid.nih.gov/v1/isolates",
]

for url in urls:
    try:
        r = requests.get(url, verify=False, timeout=30)
        print(f"  {url} -> HTTP {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  Response: {json.dumps(data, indent=2)[:2000]}")
            break
    except Exception as e:
        print(f"  {url} -> Error: {e}")

print()

# Method 2: CRyPTIC Consortium data (public)
print("=== CRyPTIC Consortium (drug resistance data) ===")
url = "https://ftp.ebi.ac.uk/pub/databases/cryptic/release/"
try:
    r = requests.get(url, verify=False, timeout=30)
    print(f"  {url} -> HTTP {r.status_code}")
    if r.status_code == 200:
        print(f"  Content (first 2000 chars): {r.text[:2000]}")
except Exception as e:
    print(f"  Error: {e}")

print()

# Method 3: ReSeqTB / NCBI BioSample for phenotype data
print("=== NCBI Biosample phenotype data ===")
# We can query BioSample for M. tuberculosis with resistance phenotype
esearch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
params = {
    "db": "biosample",
    "term": '"Mycobacterium tuberculosis"[Organism] AND "drug resistance"[Attribute]',
    "retmax": 10,
    "retmode": "json",
    "email": "farasatdhedhi@example.com",
}
r = requests.get(esearch, params=params, verify=False)
data = r.json()
ids = data.get("esearchresult", {}).get("idlist", [])
print(f"  BioSamples with drug resistance: {len(ids)} found")
if ids:
    # Fetch one sample to see structure
    efetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "biosample", "id": ids[0], "retmode": "xml", "email": "farasatdhedhi@example.com"}
    r = requests.get(efetch, params=params, verify=False)
    print(f"  Sample {ids[0]} XML (first 2000 chars): {r.text[:2000]}")

print()
print("=== Done ===")
