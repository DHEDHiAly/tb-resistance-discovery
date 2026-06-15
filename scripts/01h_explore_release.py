"""Explore CRyPTIC release directory"""
import requests
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
requests.packages.urllib3.disable_warnings()

# List CRyPTIC release dir
base = "https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/"
r = requests.get(base, verify=False, timeout=30)
print("=== CRyPTIC release_june2022/ ===")
print(r.text[:4000])

# Check for common file types
import re
lines = r.text.split("\n")
for line in lines:
    href_match = re.search(r'<a href="([^"]+)"', line)
    if href_match:
        print(f"  Found: {href_match.group(1)}")
