import os
import re
from pathlib import Path

ROOT = Path(r'C:\Elysia\Core')

DUPLICATES = [
    (r'_01_Infrastructure\._01_Infrastructure', r'_01_Infrastructure'),
    (r'_02_Logic\._02_Logic', r'_02_Logic'),
    (r'_03_Ethics\._03_Ethics', r'_03_Ethics'),
    (r'_04_Governance\._04_Governance', r'_04_Governance'),
    (r'_05_Security\._05_Security', r'_05_Security'),
    (r'_01_Reasoning\._01_Reasoning', r'_01_Reasoning'),
    (r'_02_Memory\._02_Memory', r'_02_Memory'),
    (r'_03_Physics\._03_Physics', r'_03_Physics'),
    (r'_04_Mind\._04_Mind', r'_04_Mind'),
    (r'_05_Research\._05_Research', r'_05_Research'),
]

def fix_dupes(file_path):
    try:
        content = file_path.read_text(encoding='utf-8')
        new_content = content
        for pattern, replacement in DUPLICATES:
            new_content = re.sub(pattern, replacement, new_content)
        
        if new_content != content:
            file_path.write_text(new_content, encoding='utf-8')
            print(f'Fixed dupes in: {file_path}')
    except:
        pass

if __name__ == "__main__":
    for py in ROOT.rglob("*.py"):
        fix_dupes(py)
