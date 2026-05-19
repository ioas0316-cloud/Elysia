"""Scan Legacy for treasures"""
from pathlib import Path

legacy = Path("Legacy")
treasures = []

for py in legacy.rglob("*.py"):
    if "__pycache__" in str(py) or "__init__" in py.name:
        continue
    try:
        content = py.read_text(encoding="utf-8", errors="ignore")
        lines = len(content.splitlines())
        has_class = "class " in content
        keywords = ["spiderweb", "cell", "world", "concept", "graph", "network", 
                   "dialogue", "saga", "story", "emotion", "memory", "learn", 
                   "vocabulary", "grammar", "lexicon", "symbol"]
        found = [k for k in keywords if k in content.lower()]
        if lines > 30 or found:
            treasures.append({
                "name": py.name,
                "path": str(py.relative_to(legacy)),
                "lines": lines,
                "keywords": found[:4]
            })
    except:
        pass

print(f"🏆 Legacy Treasures ({len(treasures)} files):\n")
for t in sorted(treasures, key=lambda x: -x["lines"])[:25]:
    kw = ", ".join(t["keywords"]) if t["keywords"] else "-"
    print(f"   {t['lines']:4} lines  {t['name']:40} [{kw}]")
