import requests
requests.packages.urllib3.disable_warnings()

def get_title(d):
    return d.get("title", "N/A")

def get_files(d):
    return d.get("files", [])

def get_name(f):
    return f.get("name", "?")

def get_size_mb(f):
    s = f.get("size", 0)
    return s / 1e6

def get_download_url(f):
    return f.get("download_url", "?")

def get_key(f):
    return f.get("key", "?")

print("=== 35k Dataset (GitHub) ===")
r = requests.get("https://api.github.com/repos/thatseattlevis/tb_35k_NG/contents", verify=False, timeout=30)
if r.status_code == 200:
    for item in r.json():
        print("  {} {}".format(item["type"], item["name"]))
else:
    print("  Status: {}".format(r.status_code))

print("\n=== Afro-TB (Figshare) ===")
r = requests.get("https://api.figshare.com/v2/articles/23521883", verify=False, timeout=30)
if r.status_code == 200:
    data = r.json()
    print("  Title: {}".format(get_title(data)))
    files = get_files(data)
    print("  Files: {}".format(len(files)))
    for f in files:
        print("    {}: {:.0f}MB - {}".format(get_name(f), get_size_mb(f), get_download_url(f)))
else:
    print("  Status: {}".format(r.status_code))

print("\n=== Italian collection (Zenodo) ===")
r = requests.get("https://zenodo.org/api/records/10845532", verify=False, timeout=30)
if r.status_code == 200:
    data = r.json()
    print("  Title: {}".format(get_title(data)))
    files = data.get("files", [])
    print("  Files: {}".format(len(files)))
    for f in files[:10]:
        print("    {}: {:.0f}MB".format(get_key(f), get_size_mb(f)))
    if len(files) > 10:
        print("    ... and {} more".format(len(files)-10))
else:
    print("  Status: {}".format(r.status_code))

print("\n=== NCBI Assembly count ===")
r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
    params={"db": "assembly", "term": "Mycobacterium tuberculosis[Organism] AND complete genome[filter]",
            "retmax": 0, "retmode": "json", "email": "tb@research.org"}, verify=False, timeout=30)
j = r.json()
count = int(j["esearchresult"]["count"])
print("  Complete TB assemblies: {}".format(count))
