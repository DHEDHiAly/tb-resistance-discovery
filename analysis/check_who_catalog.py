import requests, os

# Check WHO mutation catalogue GitHub
url = 'https://api.github.com/repos/GTB-tbsequencing/mutation-catalogue-2023/contents/Input%20data%20files%20for%20Solo%20algorithms'
r = requests.get(url, verify=False, timeout=15)
print(f'HTTP {r.status_code}')
if r.status_code == 200:
    for item in r.json():
        size_mb = item.get('size', 0) / 1e6
        print(f'  {item["type"]:5} {item["name"]:50} {size_mb:.1f} MB')
        
        # If it's a directory, check its contents
        if item['type'] == 'dir':
            r2 = requests.get(item['url'], verify=False, timeout=15)
            if r2.status_code == 200:
                for sub in r2.json()[:10]:
                    print(f'         {sub["name"]:45} {sub.get("size",0)/1e6:.1f} MB')

# Also check the Final Result Files
print()
url2 = 'https://api.github.com/repos/GTB-tbsequencing/mutation-catalogue-2023/contents/Final%20Result%20Files'
r2 = requests.get(url2, verify=False, timeout=15)
print(f'Final Results: HTTP {r2.status_code}')
if r2.status_code == 200:
    for item in r2.json():
        size_mb = item.get('size', 0) / 1e6
        print(f'  {item["name"]:60} {size_mb:.1f} MB')
