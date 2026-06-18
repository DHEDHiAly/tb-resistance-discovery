"""Count assemblies and download next batch"""
from pathlib import Path
import requests, gzip, time
requests.packages.urllib3.disable_warnings()

d = Path("data/genomes")
existing = set(f.name.split(".")[0] for f in d.glob("*.fasta") if f.is_file())
print(f"Current assemblies: {len(existing)}")

# Search for the NEXT batch (skip first 6000 results)
EMAIL = "tb-research@example.com"
VERIFY = False

print("Searching for more TB assemblies (offset 6000)...")
r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
    params={"db": "assembly", "term": "Mycobacterium tuberculosis[Organism] AND latest[filter]",
            "retmax": 5000, "retstart": 6000, "retmode": "json", "email": EMAIL},
    verify=VERIFY, timeout=60)
idlist = r.json()["esearchresult"]["idlist"]
print(f"Found {len(idlist)} more assembly UIDs")

downloaded = 0
for uid in idlist:
    try:
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "assembly", "id": uid, "retmode": "json", "email": EMAIL},
            verify=VERIFY, timeout=30)
        if r.status_code != 200:
            continue
        
        data = r.json().get("result", {}).get(uid, {})
        assembly_acc = data.get("assemblyaccession", "")
        if not assembly_acc or assembly_acc in existing:
            continue
        
        ftp = data.get("ftppath_refseq", "") or data.get("ftppath_genbank", "")
        if not ftp:
            continue
        
        https_ftp = ftp.replace("ftp://", "https://")
        base = ftp.split("/")[-1]
        
        found = False
        for ext in ["_genomic.fna.gz", "_genomic.fna"]:
            url = f"{https_ftp}/{base}{ext}"
            r2 = requests.get(url, stream=True, verify=VERIFY, timeout=600)
            if r2.status_code == 200:
                out_path = d / f"{assembly_acc}.fasta"
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
                    downloaded += 1
                    existing.add(assembly_acc)
                    print(f"  [{downloaded}] {assembly_acc}: {size/1e6:.1f}MB")
                    found = True
                else:
                    out_path.unlink(missing_ok=True)
                break
    
    except Exception as e:
        pass
    
    if downloaded >= 400:
        break
    
    time.sleep(0.5)

total = len(list(d.glob("*.fasta")))
print(f"\nDownloaded {downloaded} new, {total} total assemblies")
