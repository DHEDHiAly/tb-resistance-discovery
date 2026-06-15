"""
Phase 1: Download TB genome data using NCBI API
- Downloads H37Rv reference genome
- Discovers MDR-TB genome assemblies from NCBI Assembly database
- Downloads a batch of assembled genomes for variant calling
"""

import os
import sys
import json
import time
import csv
import gzip
import shutil
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from io import StringIO

import requests
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# Disable SSL warnings for lab network
requests.packages.urllib3.disable_warnings()

# ============ CONFIG ============
EMAIL = "farasatdhedhi@example.com"

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REFERENCE_DIR = PROJECT_DIR / "reference"
DOWNLOAD_DIR = DATA_DIR / "genomes"
META_DIR = DATA_DIR / "metadata"

for d in [DATA_DIR, REFERENCE_DIR, DOWNLOAD_DIR, META_DIR]:
    d.mkdir(parents=True, exist_ok=True)

VERIFY_SSL = False  # Lab network has self-signed certs


def ncbi_esearch(db, term, retmax=20):
    """Search NCBI E-utilities using requests"""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": db,
        "term": term,
        "retmax": retmax,
        "retmode": "json",
        "email": EMAIL,
    }
    r = requests.get(url, params=params, verify=VERIFY_SSL)
    r.raise_for_status()
    return r.json()


def ncbi_esummary(db, ids):
    """Fetch summaries from NCBI E-utilities"""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": db,
        "id": ",".join(ids),
        "retmode": "json",
        "email": EMAIL,
    }
    r = requests.get(url, params=params, verify=VERIFY_SSL)
    r.raise_for_status()
    return r.json()


def ncbi_efetch(db, id, rettype="fasta", retmode="text"):
    """Fetch a record from NCBI E-utilities"""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": db,
        "id": id,
        "rettype": rettype,
        "retmode": retmode,
        "email": EMAIL,
    }
    r = requests.get(url, params=params, verify=VERIFY_SSL)
    r.raise_for_status()
    return r.text


def download_file(url, outpath, desc=""):
    """Download a file with progress"""
    try:
        r = requests.get(url, stream=True, verify=VERIFY_SSL, timeout=300)
        if r.status_code == 200:
            with open(outpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"    [OK] {desc or outpath.name}")
            return True
        else:
            print(f"    ERROR: HTTP {r.status_code} for {url}")
            return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


# ============ STEP 1: Download H37Rv reference ============
def download_reference():
    """Download M. tuberculosis H37Rv reference genome"""
    ref_fasta = REFERENCE_DIR / "H37Rv.fasta"

    if ref_fasta.exists():
        print(f"[OK] Reference already exists at {ref_fasta}")
        return ref_fasta

    print("[1/5] Downloading H37Rv reference genome...")

    # Direct download from NCBI RefSeq FTP
    urls = [
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/195/955/GCF_000195955.2_ASM19595v2/GCF_000195955.2_ASM19595v2_genomic.fna.gz",
        "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/Mycobacterium_tuberculosis/reference/GCF_000195955.2_ASM19595v2/GCF_000195955.2_ASM19595v2_genomic.fna.gz",
    ]

    gz_path = REFERENCE_DIR / "H37Rv.fasta.gz"
    success = False

    for url in urls:
        if download_file(url, gz_path, f"H37Rv reference from {url.split('/')[2]}"):
            success = True
            break

    if not success:
        print("  Trying NCBI Datasets API as fallback...")
        api_url = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/GCF_000195955.2/download?include_annotation_type=GENOME_GFF3&filename=H37Rv.zip"
        r = requests.get(api_url, verify=VERIFY_SSL, allow_redirects=True)
        if r.status_code == 200:
            zip_path = REFERENCE_DIR / "H37Rv.zip"
            with open(zip_path, "wb") as f:
                f.write(r.content)
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(REFERENCE_DIR)
            # Find the fasta
            for f in REFERENCE_DIR.rglob("*.fna"):
                f.rename(ref_fasta)
                break
            zip_path.unlink()
            success = ref_fasta.exists()

    if success and gz_path.exists():
        with gzip.open(gz_path, "rb") as f_in:
            with open(ref_fasta, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()
        print(f"  [OK] Reference saved to {ref_fasta}")

    return ref_fasta if ref_fasta.exists() else None


# ============ STEP 2: Find MDR-TB assemblies ============
def search_mdr_tb_assemblies(max_results=50):
    """Search for MDR-TB genome assemblies"""
    print(f"\n[2/5] Searching for MDR-TB genome assemblies (max: {max_results})...")

    query = (
        "Mycobacterium tuberculosis[Organism] "
        "AND (drug resistance[All Fields] OR MDR[All Fields] OR multi-drug resistant[All Fields]) "
        "AND latest[filter]"
    )

    result = ncbi_esearch("assembly", query, retmax=max_results)
    id_list = result.get("esearchresult", {}).get("idlist", [])
    total = result.get("esearchresult", {}).get("count", "0")
    print(f"  Found {total} total assemblies, fetching details for {len(id_list)}...")

    if not id_list:
        query = "Mycobacterium tuberculosis[Organism] AND latest[filter]"
        result = ncbi_esearch("assembly", query, retmax=max_results)
        id_list = result.get("esearchresult", {}).get("idlist", [])

    assemblies = []
    for i in range(0, len(id_list), 10):
        batch = id_list[i:i+10]
        summary = ncbi_esummary("assembly", batch)
        docs = summary.get("result", {})

        for uid in batch:
            doc = docs.get(uid, {})
            if doc:
                assm = doc.get("assemblyaccession", "")
                assemblies.append({
                    "assembly_acc": assm,
                    "assembly_name": doc.get("assemblyname", ""),
                    "organism": doc.get("organism", ""),
                    "ftp_path": doc.get("ftppath_refseq", doc.get("ftppath_genbank", "")),
                    "biosample": doc.get("biosampleaccn", ""),
                    "bioproject": doc.get("projectaccession", ""),
                    "assembly_level": doc.get("assemblylevel", ""),
                })
        time.sleep(0.5)

    print(f"  [OK] Retrieved {len(assemblies)} assembly records")
    return assemblies


# ============ STEP 3: Save metadata ============
def save_metadata(assemblies, filepath):
    """Save metadata to CSV"""
    with open(filepath, "w", newline="") as f:
        if assemblies:
            writer = csv.DictWriter(f, fieldnames=assemblies[0].keys())
            writer.writeheader()
            writer.writerows(assemblies)
    print(f"  [OK] Metadata saved to {filepath}")
    return filepath


# ============ STEP 4: Download genome assemblies ============
def download_assemblies(assemblies, n=5):
    """Download N genome assemblies"""
    print(f"\n[3/5] Downloading {n} genome assemblies...")

    downloaded = []
    for i, assembly in enumerate(assemblies[:n]):
        ftp = assembly["ftp_path"]
        if not ftp:
            continue

        acc = assembly["assembly_acc"]
        name = ftp.split("/")[-1]
        out_file = DOWNLOAD_DIR / f"{acc}.fasta"

        if out_file.exists():
            print(f"  [{i+1}/{n}] {acc} already exists, skipping")
            downloaded.append(str(out_file))
            continue

        # Try HTTPS (some NCBI FTP paths work as HTTPS)
        for scheme in ["https://", "ftp://"]:
            fasta_url = f"{scheme}{ftp.split('://')[-1]}/{name}_genomic.fna.gz"
            print(f"  [{i+1}/{n}] Trying: {fasta_url}")
            gz_path = DOWNLOAD_DIR / f"{acc}.fasta.gz"

            r = requests.get(fasta_url, stream=True, verify=VERIFY_SSL, timeout=120)
            if r.status_code == 200:
                with open(gz_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                with gzip.open(gz_path, "rb") as f_in:
                    with open(out_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                gz_path.unlink()
                downloaded.append(str(out_file))
                print(f"    [OK] Saved {out_file.name}")
                break
            else:
                print(f"    HTTP {r.status_code}")

        time.sleep(0.3)

    print(f"  [OK] Downloaded {len(downloaded)} genomes")
    return downloaded


# ============ STEP 5: Verify ============
def verify_downloads(reference, genome_files):
    """Verify downloaded files"""
    print("\n[4/5] Verifying downloads...")

    if reference and reference.exists():
        size = reference.stat().st_size / 1e6
        # Count sequences (first pass with simple line count of >)
        with open(reference) as f:
            seq_count = sum(1 for line in f if line.startswith(">"))
        print(f"  Reference H37Rv: {seq_count} sequences, {size:.1f} MB")

    for f in genome_files[:3]:
        path = Path(f)
        if path.exists():
            with open(path) as fh:
                seq_count = sum(1 for line in fh if line.startswith(">"))
            print(f"  {path.name}: {seq_count} contigs, {path.stat().st_size / 1e6:.1f} MB")


# ============ MAIN ============
if __name__ == "__main__":
    print("=" * 60)
    print("TB Resistance Discovery - Phase 1: Data Acquisition")
    print("=" * 60)

    ref = download_reference()

    assemblies = search_mdr_tb_assemblies(max_results=50)

    if assemblies:
        save_metadata(assemblies, META_DIR / "tb_assemblies.csv")
        genome_files = download_assemblies(assemblies, n=5)
        verify_downloads(ref, genome_files)

    print("\n[DONE] Phase 1 complete!")
    print(f"  Reference: {REFERENCE_DIR}")
    print(f"  Genomes: {DOWNLOAD_DIR}")
    print(f"  Metadata: {META_DIR}")
