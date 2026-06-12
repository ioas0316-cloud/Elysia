import re

log_path = r'C:\Users\USER\.gemini\antigravity-ide\brain\de8c57de-e945-4002-a88a-a65a05330a1c\.system_generated\tasks\task-1848.log'

with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Try cp949 if utf-8 fails
if '�' in content[:100]:
    try:
        with open(log_path, 'r', encoding='cp949', errors='replace') as f:
            content = f.read()
    except:
        pass

# Extract key patterns
patterns = [
    r'.*꿈.*Ego-Ideal.*',
    r'.*목표 지향.*',
    r'.*백일몽.*',
    r'.*기각.*',
    r'.*단기 관심.*',
    r'.*가치관 진화.*',
    r'.*렌즈.*변이.*',
    r'--- Cycle [0-9]+ ---',
    r'.*나의 꿈.*',
]

lines = content.split('\n')
results = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    for pat in patterns:
        if re.search(pat, line):
            results.append(line)
            break

# Write results
with open(r'C:\Elysia\scripts\log_extract.txt', 'w', encoding='utf-8') as out:
    out.write(f"Total lines in log: {len(lines)}\n")
    out.write(f"Matching lines: {len(results)}\n\n")
    for r in results:
        out.write(r + '\n')

print(f"Extracted {len(results)} key lines from {len(lines)} total.")
