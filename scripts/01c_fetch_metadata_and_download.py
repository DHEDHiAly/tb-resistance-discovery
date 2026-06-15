"""
Fetch metadata for TB assemblies and download a batch
"""
import requests
import json
import csv
import gzip
import shutil
import time
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
requests.packages.urllib3.disable_warnings()

PROJECT_DIR = Path(__file__).resolve().parent.parent
META_DIR = PROJECT_DIR / "data" / "metadata"
DOWNLOAD_DIR = PROJECT_DIR / "data" / "genomes"
VERIFY_SSL = False
EMAIL = "farasatdhedhi@example.com"

# Load IDs
with open(META_DIR / "all_tb_ids.json") as f:
    all_ids = json.load(f)

print(f"Loaded {len(all_ids)} assembly IDs")

# Fetch summaries in batches of 10
url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
assemblies = []

for i in range(0, min(len(all_ids), 100), 10):
    batch = all_ids[i:i+10]
    params = {
        "db": "assembly",
        "id": ",".join(batch),
        "retmode": "json",
        "email": EMAIL,
    }
    r = requests.get(url, params=params, verify=VERIFY_SSL)
    data = r.json()
    results = data.get("result", {})

    for uid in batch:
        doc = results.get(uid, {})
        if doc:
            assemblies.append({
                "uid": uid,
                "assembly_acc": doc.get("assemblyaccession", ""),
                "assembly_name": doc.get("assemblyname", ""),
                "organism": doc.get("organism", ""),
                "ftp_path": doc.get("ftppath_refseq", doc.get("ftppath_genbank", "")),
                "biosample": doc.get("biosampleaccn", ""),
                "bioproject": doc.get("projectaccession", ""),
                "assembly_level": doc.get("assemblylevel", ""),
                "submitter": doc.get("submitterorganization", ""),
            })

    print(f"  Batch {i//10 + 1}/10: {len(batch)} IDs fetched")
    time.sleep(0.5)

print(f"\nTotal assemblies with metadata: {len(assemblies)}")

# Save metadata
csv_path = META_DIR / "all_tb_metadata.csv"
with open(csv_path, "w", newline="") as f:
    if assemblies:
        w = csv.DictWriter(f, fieldnames=assemblies[0].keys())
        w.writeheader()
        w.writerows(assemblies)
print(f"Metadata saved to {csv_path}")

# Download first 10 genomes
print(f"\nDownloading 10 genomes...")
downloaded = []
for i, a in enumerate(assemblies[:10]):
    ftp = a["ftp_path"]
    if not ftp:
        continue
    acc = a["assembly_acc"]
    name = ftp.split("/")[-1]
    out_file = DOWNLOAD_DIR / f"{acc}.fasta"

    if out_file.exists():
        print(f"  [{i+1}/10] {acc} exists, skipping")
        downloaded.append(str(out_file))
        continue

    fasta_url = f"https://{ftp.split('://')[-1]}/{name}_genomic.fna.gz"
    print(f"  [{i+1}/10] Downloading {acc}...", end=" ")

    try:
        r = requests.get(fasta_url, stream=True, verify=VERIFY_SSL, timeout=120)
        if r.status_code == 200:
            gz_path = DOWNLOAD_DIR / f"{acc}.fasta.gz"
            with open(gz_path, "wb") as fh:
                for chunk in r.iter_content(8192):
                    fh.write(chunk)
            with gzip.open(gz_path, "rb") as fh_in:
                with open(out_file, "wb") as fh_out:
                    shutil.copyfileobj(fh_in, fh_out)
            gz_path.unlink()
            downloaded.append(str(out_file))
            print(f"OK ({out_file.stat().st_size/1e6:.1f} MB)")
        else:
            print(f"HTTP {r.status_code}")
    except Exception as e:
        print(f"Error: {e}")

    time.sleep(0.3)

print(f"\nDownloaded {len(downloaded)} genomes to {DOWNLOAD_DIR}")
print(f"Total genomes in directory: {len(list(DOWNLOAD_DIR.glob('*.fasta')))}")
