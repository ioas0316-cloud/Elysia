"""Quick analysis of Unknown files"""
import json
from pathlib import Path

dna_dir = Path("data/CodeDNA")
unknowns = []

for f in dna_dir.glob("*.dna.json"):
    if f.name.startswith("_"):
        continue
    try:
        data = json.loads(f.read_text(encoding='utf-8'))
        if data.get("phase") == "unknown":
            unknowns.append(data)
    except:
        pass

# Analyze locations
locations = {}
for u in unknowns:
    parts = u["path"].replace("\\", "/").split("/")
    loc = parts[0] if len(parts) > 1 else "root"
    locations[loc] = locations.get(loc, 0) + 1

print("📍 Unknown Files by Location:")
for loc, count in sorted(locations.items(), key=lambda x: -x[1])[:10]:
    print(f"   {loc}: {count}")

# Check hermits (no imports = likely dead code)
hermits = [u for u in unknowns if u["imports"] == 0]
print(f"\n🏝️ Unknown + Hermit (no imports): {len(hermits)}/{len(unknowns)}")

# Sample names
print("\n📝 Sample Unknown Files:")
for u in sorted(unknowns, key=lambda x: -x["lines"])[:10]:
    print(f"   {u['name']:40} {u['lines']:4} lines, {u['imports']} imports")
