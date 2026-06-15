"""
Search for more TB genome assemblies to expand the dataset
"""
import requests
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
requests.packages.urllib3.disable_warnings()

PROJECT_DIR = Path(__file__).resolve().parent.parent
META_DIR = PROJECT_DIR / "data" / "metadata"
META_DIR.mkdir(parents=True, exist_ok=True)

VERIFY_SSL = False
EMAIL = "farasatdhedhi@example.com"

url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

queries = [
    ('"Mycobacterium tuberculosis"[Organism] AND latest[filter] AND "complete genome"[filter]', 200),
    ('"Mycobacterium tuberculosis"[Organism] AND latest[filter] AND "contig"[filter]', 200),
    ('"Mycobacterium tuberculosis"[Organism] AND latest[filter] AND "scaffold"[filter]', 200),
]

all_ids = []
for q, retmax in queries:
    params = {
        "db": "assembly",
        "term": q,
        "retmax": retmax,
        "retmode": "json",
        "email": EMAIL,
    }
    r = requests.get(url, params=params, verify=VERIFY_SSL)
    data = r.json()
    ids = data.get("esearchresult", {}).get("idlist", [])
    total = data.get("esearchresult", {}).get("count", "0")
    print(f"Query: {q[:70]}... => {len(ids)} IDs (total available: {total})")
    all_ids.extend(ids)
    time.sleep(0.5)

unique_ids = list(set(all_ids))
print(f"\nTotal unique assembly IDs: {len(unique_ids)}")

with open(META_DIR / "all_tb_ids.json", "w") as f:
    json.dump(unique_ids, f)

print(f"Saved to {META_DIR / 'all_tb_ids.json'}")
