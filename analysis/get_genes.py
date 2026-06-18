import re
with open('scripts/04_resistance_forecasting.py') as f:
    content = f.read()
idx = content.find('RESISTANCE_GENES = [')
depth = 1
i = idx + len('RESISTANCE_GENES = [')
while depth > 0 and i < len(content):
    if content[i] == '[': depth += 1
    elif content[i] == ']': depth -= 1
    i += 1
block = content[idx:i]
pattern = r'\(\"(\w+)\",\s*\"(\w+)\",\s*\"(\w+)\"'
matches = re.findall(pattern, block)
for g, locus, drug in matches:
    print(f'{g:8s} {locus:12s} {drug}')
