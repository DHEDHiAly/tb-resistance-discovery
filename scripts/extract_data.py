"""
Extract committed data archives for reproducibility.

Usage:
    python scripts/extract_data.py
"""

import tarfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"

def extract_archive(name, subdir):
    path = DATA_DIR / name
    if not path.exists():
        print(f"  SKIP: {name} not found")
        return
    print(f"Extracting {name} ({path.stat().st_size / 1024:.0f} KB)...")
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
    print(f"  Done -> data/{subdir}/")

def main():
    print("Extracting committed data archives...")
    extract_archive("pdbs.tar.gz", "pdb")
    extract_archive("cryptic_supplement.tar.gz", "cryptic")
    print("\nAll archives extracted. Ready to run pipeline.")
    print("NOTE: MUTATIONS.csv.gz (1.4 GB) must be downloaded separately:")
    print("  python scripts/download_cryptic_data.py")

if __name__ == "__main__":
    main()
