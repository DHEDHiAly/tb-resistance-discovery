import requests
import json
import sys

# 1. Afro-TB Figshare
print("=== AFRO-TB DATASET ===")
resp = requests.get('https://api.figshare.com/v2/articles/22160002', verify=False, timeout=30)
data = resp.json()
print(f"Title: {data.get('title','N/A')}")
files = data.get('files', [])
print(f"Files: {len(files)}")
for f in files[:30]:
    print(f"  {f['name']}: {f['size']/1e6:.0f} MB")

print()

# 2. Check Figshare for the article files
print("=== AFRO-TB FILE LINKS ===")
for f in files[:5]:
    print(f"  {f['name']}: {f.get('download_url', 'N/A')}")

print()

# 3. Check 35k dataset from Pruthi et al.
print("=== 35K DATASET (Pruthi et al. 2024) ===")
print("GitHub: https://github.com/SSID08/TB-ML")
# Try to check the GitHub repo
resp = requests.get('https://api.github.com/repos/SSID08/TB-ML', verify=False, timeout=15)
if resp.status_code == 200:
    data = resp.json()
    print(f"  Stars: {data.get('stargazers_count', 'N/A')}")
    print(f"  Description: {data.get('description', 'N/A')}")
    # Check the releases
    resp2 = requests.get('https://api.github.com/repos/SSID08/TB-ML/releases', verify=False, timeout=15)
    releases = resp2.json()
    print(f"  Releases: {len(releases)}")
    for r in releases[:3]:
        print(f"    {r['tag_name']}: {r['name']}")
else:
    print(f"  HTTP {resp.status_code}")

print()

# 4. Check Italian WGS dataset (Zenodo)
print("=== ITALIAN WGS DATASET ===")
resp = requests.get('https://zenodo.org/api/records/14780239', verify=False, timeout=15)
if resp.status_code == 200:
    data = resp.json()
    print(f"  Title: {data.get('metadata',{}).get('title','N/A')}")
    files = data.get('files', [])
    print(f"  Files: {len(files)}")
    for f in files[:10]:
        print(f"    {f['key']}: {f['size']/1e6:.0f} MB")
else:
    print(f"  HTTP {resp.status_code}")
