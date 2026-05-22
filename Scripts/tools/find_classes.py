import os
import re

target_classes = [
    "DoubleHelixRotor", "VortexField", "SovereignInterferometer", 
    "FogField", "PrismaticRefractor", "RotorNode", 
    "UniversalConstants", "SpecializedRotor"
]

root_dirs = ["c:\\Elysia", "c:\\elysia_seed", "c:\\eye", "c:\\Archive"]
found = {}

for root_dir in root_dirs:
    if not os.path.exists(root_dir):
        continue
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    for tc in target_classes:
                        if re.search(r"class\s+" + tc, content):
                            if tc not in found:
                                found[tc] = []
                            found[tc].append(path)
                except Exception as e:
                    pass

print("Search Results:")
for tc in target_classes:
    if tc in found:
        print(f"✅ {tc}: {found[tc]}")
    else:
        print(f"❌ {tc}: NOT FOUND")
