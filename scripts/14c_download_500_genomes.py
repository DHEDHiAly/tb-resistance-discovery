"""
Download 500 TB genomes using NCBI E-utilities + FTP
"""
import requests, gzip, time, os, concurrent.futures
from pathlib import Path

requests.packages.urllib3.disable_warnings()

PROJECT_DIR = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = PROJECT_DIR / "data" / "genomes"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
EMAIL = "tb-research@example.com"
VERIFY = False

# Get existing assemblies
existing = set(f.name.split(".")[0].replace(".fasta","") for f in DOWNLOAD_DIR.glob("*.fasta") if f.is_file())
# Also check .gz files that might already exist
for f in DOWNLOAD_DIR.glob("*.fasta.gz"):
    existing.add(f.name.split(".")[0])
print(f"Existing: {len(existing)} assemblies")

# Get all TB assembly IDs
print("Searching NCBI for TB assemblies...")
r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
    params={"db": "assembly", "term": "Mycobacterium tuberculosis[Organism] AND latest[filter]",
            "retmax": 5000, "retmode": "json", "email": EMAIL},
    verify=VERIFY, timeout=60)
idlist = r.json()["esearchresult"]["idlist"]
print(f"Found {len(idlist)} assembly UIDs")

def download_via_ftp(uid):
    """Get FTP link and download for one assembly UID."""
    try:
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "assembly", "id": uid, "retmode": "json", "email": EMAIL},
            verify=VERIFY, timeout=30)
        if r.status_code != 200:
            return None
        
        data = r.json().get("result", {}).get(uid, {})
        assembly_acc = data.get("assemblyaccession", "")
        if not assembly_acc:
            return None
        
        if assembly_acc in existing:
            return None
        
        ftp = data.get("ftppath_refseq", "") or data.get("ftppath_genbank", "")
        if not ftp:
            return None
        
        # Convert FTP to HTTPS
        https_ftp = ftp.replace("ftp://", "https://")
        base = ftp.split("/")[-1]
        
        for ext in ["_genomic.fna.gz", "_genomic.fna"]:
            url = f"{https_ftp}/{base}{ext}"
            r2 = requests.get(url, stream=True, verify=VERIFY, timeout=600)
            if r2.status_code == 200:
                out_path = DOWNLOAD_DIR / f"{assembly_acc}.fasta"
                gz_path = out_path.with_suffix(".fasta.gz")
                
                with open(gz_path, "wb") as f:
                    for chunk in r2.iter_content(8192*32):
                        if chunk:
                            f.write(chunk)
                
                if ext.endswith(".gz"):
                    with gzip.open(gz_path, "rb") as src:
                        with open(out_path, "wb") as dst:
                            dst.write(src.read())
                    gz_path.unlink()
                else:
                    gz_path.rename(out_path)
                
                size = out_path.stat().st_size
                if size > 50000:
                    return f"{assembly_acc}: {size/1e6:.1f}MB"
                else:
                    out_path.unlink(missing_ok=True)
                    return None
    except Exception as e:
        pass
    return None

# Download in parallel with delay to avoid rate limiting
target = min(600, len(idlist))
batch = idlist[:target]

downloaded = 0
failed = 0

print(f"Attempting to download {len(batch)} assemblies ({target} target new)...")

# Sequential (parallel with NCBI FTP causes issues)
for i, uid in enumerate(batch):
    result = download_via_ftp(uid)
    if result:
        downloaded += 1
        print(f"  [{downloaded}] {result}")
    else:
        failed += 1
    
    if downloaded >= 500:
        break
    
    if (i + 1) % 50 == 0:
        print(f"  Progress: {downloaded} downloaded, {failed} failed, {i+1}/{len(batch)} tried")
    
    time.sleep(0.5)  # Rate limiting

total = len(list(DOWNLOAD_DIR.glob("*.fasta")))
print(f"\nDone: {downloaded} new, {total} total assemblies in directory")
