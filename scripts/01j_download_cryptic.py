import requests, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
requests.packages.urllib3.disable_warnings()

PROJECT_DIR = Path(__file__).resolve().parent.parent
META_DIR = PROJECT_DIR / "data" / "metadata"
META_DIR.mkdir(parents=True, exist_ok=True)

base = "https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022"

# Download the latest reuse table (phenotype data)
files = {
    "reuse/CRyPTIC_reuse_table_20231107.csv": "cryptic_phenotypes.csv",
    "reuse/CRyPTIC_reuse_table_20231107_col_names.txt": "cryptic_columns.txt",
    "reuse/README": "cryptic_reuse_readme.txt",
    "reproducibility/README": "cryptic_repro_readme.txt",
    "README": "cryptic_top_readme.txt",
}

for remote, local in files.items():
    url = f"{base}/{remote}"
    outpath = META_DIR / local
    if outpath.exists():
        print(f"[SKIP] {local} exists")
        continue
    print(f"Downloading {remote}...", end=" ")
    r = requests.get(url, verify=False, timeout=120)
    if r.status_code == 200:
        with open(outpath, "wb") as f:
            f.write(r.content)
        size = outpath.stat().st_size / 1e6
        print(f"OK ({size:.1f} MB)")
    else:
        print(f"HTTP {r.status_code}")
