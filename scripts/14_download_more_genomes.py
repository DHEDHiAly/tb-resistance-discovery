"""
Download additional TB genomes for homoplasy computation.
Sources:
1. NCBI Assembly DB - MDR-TB genome assemblies
2. European Nucleotide Archive - TB VCFs
3. Pre-computed mutation frequency data
"""

import os, sys, json, time, csv, gzip, shutil, requests, warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Unverified HTTPS request")
requests.packages.urllib3.disable_warnings()

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DOWNLOAD_DIR = DATA_DIR / "genomes"
META_DIR = DATA_DIR / "metadata"
VARIANTS_DIR = PROJECT_DIR / "variants"

for d in [DOWNLOAD_DIR, META_DIR, VARIANTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

VERIFY_SSL = False
EMAIL = "tb-research@example.com"


def ncbi_esearch(db, term, retmax=200):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": db, "term": term, "retmax": retmax, "retmode": "json", "email": EMAIL}
    r = requests.get(url, params=params, verify=VERIFY_SSL, timeout=60)
    r.raise_for_status()
    return r.json()


def ncbi_esummary(db, ids):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {"db": db, "id": ",".join(ids), "retmode": "json", "email": EMAIL}
    r = requests.get(url, params=params, verify=VERIFY_SSL, timeout=60)
    r.raise_for_status()
    return r.json()


def download_file(url, outpath, desc=""):
    try:
        r = requests.get(url, stream=True, verify=VERIFY_SSL, timeout=300)
        if r.status_code == 200:
            with open(outpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  [OK] {desc or outpath.name} ({outpath.stat().st_size/1e6:.1f} MB)")
            return True
        else:
            print(f"  ERROR: HTTP {r.status_code} for {url}")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def search_tb_assemblies(max_results=200):
    """Search for M. tuberculosis assemblies with strain diversity"""
    print(f"\nSearching for TB assemblies (max: {max_results})...")

    # Broader search: get diverse TB assemblies
    queries = [
        "Mycobacterium tuberculosis[Organism] AND latest[filter] AND (complete genome[Assembly Level] OR chromosome[Assembly Level])",
        "Mycobacterium tuberculosis[Organism] AND latest[filter] AND (MDR[All Fields] OR drug resistance[All Fields])",
    ]

    seen_ids = set()
    all_assemblies = []

    for query in queries:
        result = ncbi_esearch("assembly", query, retmax=max_results)
        id_list = result.get("esearchresult", {}).get("idlist", [])
        print(f"  Found {len(id_list)} assemblies for query")

        for i in range(0, len(id_list), 10):
            batch = [x for x in id_list[i:i+10] if x not in seen_ids]
            if not batch:
                continue
            seen_ids.update(batch)
            summary = ncbi_esummary("assembly", batch)
            docs = summary.get("result", {})
            for uid in batch:
                doc = docs.get(uid, {})
                if doc and doc.get("assemblyaccession", ""):
                    acc = doc["assemblyaccession"]
                    all_assemblies.append({
                        "assembly_acc": acc,
                        "assembly_name": doc.get("assemblyname", ""),
                        "organism": doc.get("organism", ""),
                        "ftp_path": doc.get("ftppath_refseq", doc.get("ftppath_genbank", "")),
                        "biosample": doc.get("biosampleaccn", ""),
                        "bioproject": doc.get("projectaccession", ""),
                        "assembly_level": doc.get("assemblylevel", ""),
                    })
            time.sleep(0.5)

    print(f"  Total unique assemblies: {len(all_assemblies)}")
    return all_assemblies


def download_assemblies(assemblies, max_download=50):
    """Download genome assemblies"""
    print(f"\nDownloading up to {max_download} assemblies...")

    existing = set()
    for f in DOWNLOAD_DIR.glob("*.fasta"):
        existing.add(f.stem)

    downloaded = []
    skipped = 0
    for i, assembly in enumerate(assemblies):
        if len(downloaded) >= max_download:
            break
        acc = assembly["assembly_acc"]
        if acc in existing:
            skipped += 1
            continue

        ftp = assembly["ftp_path"]
        if not ftp:
            continue
        name = ftp.split("/")[-1]
        out_file = DOWNLOAD_DIR / f"{acc}.fasta"

        success = False
        for scheme in ["https://", "ftp://"]:
            fasta_url = f"{scheme}{ftp.split('://')[-1]}/{name}_genomic.fna.gz"
            gz_path = DOWNLOAD_DIR / f"{acc}.fasta.gz"
            try:
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
                    print(f"  [{len(downloaded)}/{max_download}] {acc} ({out_file.stat().st_size/1e6:.1f} MB)")
                    success = True
                    break
            except Exception as e:
                print(f"    {scheme}: {e}")
        time.sleep(0.3)

    print(f"  Downloaded: {len(downloaded)}, Skipped (exist): {skipped}")
    return downloaded


def download_cryptic_excluded_vcfs(n=100):
    """Download VCFs from CRyPTIC excluded samples (no phenotype data, not in validation)"""
    print(f"\nDownloading up to {n} CRyPTIC excluded sample VCFs...")

    url = "https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/reuse/CRyPTIC_excluded_samples_20220607.tsv"
    resp = requests.get(url, verify=VERIFY_SSL, timeout=60)

    lines = resp.text.strip().split("\n")
    print(f"  Found {len(lines)-1} excluded samples")

    base_url = "https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/reuse/vcfs/"
    downloaded = 0
    for i, line in enumerate(lines[1:]):
        if downloaded >= n:
            break
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        sample_id, ena_sample, vcf_path, regeno_path = fields[:4]
        vcf_url = base_url + vcf_path
        out_name = f"cryptic_excluded_{sample_id.replace('.','_')}.vcf.gz"
        out_path = VARIANTS_DIR / out_name

        if out_path.exists():
            continue

        try:
            r = requests.get(vcf_url, stream=True, verify=VERIFY_SSL, timeout=120)
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded += 1
                size_mb = out_path.stat().st_size / 1e6
                print(f"  [{downloaded}/{n}] {out_name} ({size_mb:.1f} MB)")
            else:
                print(f"  HTTP {r.status_code} for {sample_id}")
        except Exception as e:
            print(f"  Error downloading {sample_id}: {e}")
        time.sleep(0.2)

    print(f"  Downloaded {downloaded} CRyPTIC excluded VCFs")
    return downloaded


def download_afro_tb_metadata():
    """Download Afro-TB dataset metadata and mutation table"""
    print("\nDownloading Afro-TB metadata...")

    # Try to access the Afro-TB data
    urls = [
        "https://bioinformatics.um6p.ma/AfroTB/api/samples",
        "https://bioinformatics.um6p.ma/AfroTB/data/metadata.csv",
        "https://figshare.com/ndownloader/files/22160002",
    ]

    for url in urls:
        try:
            r = requests.get(url, verify=VERIFY_SSL, timeout=30)
            if r.status_code == 200:
                print(f"  [OK] {url} - {len(r.content)/1e6:.1f} MB")
            else:
                print(f"  HTTP {r.status_code} for {url}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("TB Data Acquisition for Scale-up")
    print("=" * 60)

    # Step 1: Search and download NCBI assemblies
    print("\n--- STEP 1: NCBI Assemblies ---")
    assemblies = search_tb_assemblies(max_results=200)
    if assemblies:
        meta_path = META_DIR / "tb_assemblies_extended.csv"
        with open(meta_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=assemblies[0].keys())
            w.writeheader()
            w.writerows(assemblies)
        print(f"  Metadata saved to {meta_path}")

        download_assemblies(assemblies, max_download=50)

    # Step 2: Download CRyPTIC excluded VCFs (no phenotype data, OK for features)
    print("\n--- STEP 2: CRyPTIC Excluded Sample VCFs ---")
    download_cryptic_excluded_vcfs(n=50)

    # Step 3: Check Afro-TB
    print("\n--- STEP 3: Afro-TB ---")
    download_afro_tb_metadata()

    print("\n[DONE] Data acquisition complete!")
