import os
import re
from pathlib import Path

ROOT = Path(r'C:\Elysia\Core')

SUBSTITUTIONS = [
    (r'Core\._01_Foundation\._02_Logic\.gemini_api', r'Core._03_Interaction._04_Network.gemini_api'),
    (r'Core\._01_Foundation\._02_Logic\.Math', r'Core._01_Foundation._02_Logic'),
]

def repair(file_path):
    try:
        content = file_path.read_text(encoding='utf-8')
        new_content = content
        for pattern, replacement in SUBSTITUTIONS:
            new_content = re.sub(pattern, replacement, new_content)
        
        if new_content != content:
            file_path.write_text(new_content, encoding='utf-8')
            print(f'Fixed: {file_path}')
    except:
        pass

if __name__ == "__main__":
    for py in ROOT.rglob("*.py"):
        repair(py)
