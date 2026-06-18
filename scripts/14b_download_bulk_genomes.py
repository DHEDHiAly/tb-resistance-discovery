"""Search NCBI for available TB genomes and download more"""
import requests, json, os, time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DOWNLOAD_DIR = DATA_DIR / "genomes"

EMAIL = "tb-research@example.com"
VERIFY = False
requests.packages.urllib3.disable_warnings()

# 1. Check how many TB complete genomes are available
r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
    params={"db": "assembly", "term": "Mycobacterium tuberculosis[Organism] AND complete genome[filter]",
            "retmax": 0, "retmode": "json", "email": EMAIL}, verify=VERIFY, timeout=30)
j = r.json()
total_available = int(j["esearchresult"]["count"])
print(f"Total TB complete genome assemblies available: {total_available}")

# Get IDs (first 10000)
r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
    params={"db": "assembly", "term": "Mycobacterium tuberculosis[Organism] AND complete genome[filter]",
            "retmax": 10000, "retmode": "json", "email": EMAIL}, verify=VERIFY, timeout=30)
ids = r.json()["esearchresult"]["idlist"]
print(f"Assembly IDs retrieved: {len(ids)}")

# How many do we already have?
existing = set(f.name.split(".")[0] for f in DOWNLOAD_DIR.glob("*.fasta") if f.is_file())
print(f"Already downloaded: {len(existing)}")

# Download in batches of 100
# First, get summaries for all IDs to find FTP links
new_count = 0
batch_size = 200

for i in range(0, min(len(ids), 5000), batch_size):  # Cap at 5000
    batch = ids[i:i+batch_size]
    r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={"db": "assembly", "id": ",".join(batch), "retmode": "json", "email": EMAIL},
        verify=VERIFY, timeout=60)
    data = r.json()
    
    for uid in batch:
        if uid not in data.get("result", {}):
            continue
        result = data["result"][uid]
        assembly_acc = result.get("assemblyaccession", "")
        if not assembly_acc:
            continue
        if assembly_acc in existing:
            continue
        
        # Get FTP link
        ftp_path = result.get("ftppath_refseq", "") or result.get("ftppath_genbank", "")
        if not ftp_path:
            continue
        
        # Download the genomic FASTA
        fasta_url = f"{ftp_path}/{ftp_path.split('/')[-1]}_genomic.fna.gz"
        outgz = DOWNLOAD_DIR / f"{assembly_acc}.fasta.gz"
        outfa = DOWNLOAD_DIR / f"{assembly_acc}.fasta"
        
        if outfa.exists() or outgz.exists():
            continue
        
        try:
            print(f"  Downloading {assembly_acc}...", end="", flush=True)
            r = requests.get(fasta_url, stream=True, verify=VERIFY, timeout=300)
            if r.status_code != 200:
                # Try .fna instead of .fna.gz
                fasta_url = f"{ftp_path}/{ftp_path.split('/')[-1]}_genomic.fna"
                r = requests.get(fasta_url, stream=True, verify=VERIFY, timeout=300)
                if r.status_code != 200:
                    print(f" not found (HTTP {r.status_code})")
                    continue
            
            with open(outgz, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            
            # Decompress
            import gzip
            with gzip.open(outgz, "rb") as gz:
                with open(outfa, "wb") as f:
                    f.write(gz.read())
            outgz.unlink()
            
            new_count += 1
            print(f" OK ({new_count} new)")
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f" Error: {e}")
    
    time.sleep(3)  # Between batches

print(f"\nDownloaded {new_count} new assemblies (total in dir: {len(list(DOWNLOAD_DIR.glob('*.fasta')))})")
