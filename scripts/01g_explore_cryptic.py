"""Explore CRyPTIC and find TB resistance data"""
import requests
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
requests.packages.urllib3.disable_warnings()

# CRyPTIC FTP listing
print("=== CRyPTIC FTP Contents ===")
url = "https://ftp.ebi.ac.uk/pub/databases/cryptic/"
r = requests.get(url, verify=False, timeout=30)
print(r.text[:3000])

# Try the WHO mutation catalogue
print(f"\n=== WHO TB Mutation Catalogue ===")
who_urls = [
    "https://www.who.int/publications/i/item/9789240028173",
    "https://cdn.who.int/media/docs/default-source/documents/tuberculosis/who-mutation-catalogue.xlsx",
]
for u in who_urls:
    r = requests.get(u, verify=False, timeout=30)
    print(f"{u.split('/')[-1]} -> {r.status_code}")

# NCBI Datasets command-line (check if available)
print(f"\n=== NCBI Datasets CLI ===")
import shutil
for cmd in ["datasets", "datasets.exe"]:
    path = shutil.which(cmd)
    print(f"{cmd}: {path}")

# Check if we can download the CRyPTIC phenotype file
print(f"\n=== Looking for phenotype files ===")
pheno_urls = [
    "https://ftp.ebi.ac.uk/pub/databases/cryptic/phenotypes.csv",
    "https://ftp.ebi.ac.uk/pub/databases/cryptic/phenotypes.tsv",
    "https://ftp.ebi.ac.uk/pub/databases/cryptic/release_metadata.csv",
    "https://ftp.ebi.ac.uk/pub/databases/cryptic/release/",
]
for u in pheno_urls:
    try:
        r = requests.get(u, verify=False, timeout=30)
        print(f"{u.split('/')[-1]} -> {r.status_code}")
        if r.status_code == 200:
            print(f"  First 500 chars: {r.text[:500]}")
    except Exception as e:
        print(f"{u.split('/')[-1]} -> Error: {e}")
