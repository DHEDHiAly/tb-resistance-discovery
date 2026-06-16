"""
Download CRyPTIC MUTATIONS table (1.4 GB) for prospective validation.

The full MUTATIONS.csv.gz (1.4 GB) is too large to commit to git.
This script downloads it from the CRyPTIC public repository.

Usage:
    python scripts/download_cryptic_data.py

Requires ~2 GB free disk space.
"""

import gzip
import os
import sys
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CRYPTIC_DIR = BASE / "data" / "cryptic"
CRYPTIC_DIR.mkdir(parents=True, exist_ok=True)

# Try multiple mirrors in order of preference
MUTATIONS_URLS = [
    "https://zenodo.org/records/15679731/files/MUTATIONS.csv.gz?download=1",
    "https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/reuse/MUTATIONS.csv.gz",
    "ftp://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/reuse/MUTATIONS.csv.gz",
]

EXPECTED_MD5 = None  # Not available for all mirrors; verify by size (~1.4 GB)

def verify_gzip(path):
    """Verify the file is a valid gzip archive."""
    try:
        with gzip.open(path, "rb") as f:
            f.read(1024)
        return True
    except Exception:
        return False

def download_file(url, dest):
    """Download with progress."""
    print(f"Downloading from:\n  {url}")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 / total
                        mb_dl = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        print(f"  {mb_dl:.0f}/{mb_total:.0f} MB ({pct:.0f}%)", end="\r")
                    else:
                        mb_dl = downloaded / (1024 * 1024)
                        print(f"  {mb_dl:.0f} MB downloaded", end="\r")
        print()
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

def main():
    dest = CRYPTIC_DIR / "MUTATIONS.csv.gz"
    if dest.exists():
        size_gb = os.path.getsize(dest) / (1024**3)
        print(f"MUTATIONS.csv.gz already exists ({size_gb:.1f} GB)")
        if verify_gzip(dest):
            print("  Valid gzip archive.")
            return
        else:
            print("  Corrupted — re-downloading.")
            dest.unlink()

    for url in MUTATIONS_URLS:
        print(f"\nTrying mirror {MUTATIONS_URLS.index(url) + 1}/{len(MUTATIONS_URLS)}...")
        if download_file(url, dest):
            if verify_gzip(dest):
                size_gb = os.path.getsize(dest) / (1024**3)
                print(f"\nDownloaded successfully: {size_gb:.1f} GB")
                print(f"  Saved to: {dest}")
                return
            else:
                print("  Downloaded file is not a valid gzip — trying next mirror.")
                dest.unlink()
        else:
            print("  Connection failed — trying next mirror.")

    print("\nERROR: Could not download MUTATIONS.csv.gz from any mirror.")
    print("Manual download options:")
    print("  1. Zenodo: https://zenodo.org/records/15679731")
    print("  2. EBI FTP: ftp://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/reuse/")
    sys.exit(1)

if __name__ == "__main__":
    main()
