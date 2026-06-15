import requests, warnings
warnings.filterwarnings("ignore")
requests.packages.urllib3.disable_warnings()

b = "https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022"
for d in ["README", "pubs/", "reproducibility/", "reuse/"]:
    r = requests.get(f"{b}/{d}", verify=False, timeout=30)
    print(f"=== {d} ===")
    print(r.text[:2000])
    print()
