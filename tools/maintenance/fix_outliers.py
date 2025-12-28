import os
import re
from pathlib import Path

ROOT = Path(r'C:\Elysia\Core')

SUBSTITUTIONS = [
    (r'Project_Elysia\.core', r'Core'),
    (r'from nano_core\.registry import Scheduler', r'from Core._01_Foundation._01_Infrastructure.elysia_core import Organ'),
    (r'Core\._01_Foundation\.\s*Foundation', r'Core._01_Foundation._02_Logic'),
]

def repair(file_path):
    try:
        content = file_path.read_text(encoding='utf-8')
        new_content = content
        for pattern, replacement in SUBSTITUTIONS:
            new_content = re.sub(pattern, replacement, new_content)
        
        if new_content != content:
            file_path.write_text(new_content, encoding='utf-8')
            print(f'Fixed outlier in: {file_path}')
    except:
        pass

if __name__ == "__main__":
    for py in ROOT.rglob("*.py"):
        repair(py)
