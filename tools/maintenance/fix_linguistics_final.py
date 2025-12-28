import os
import re
from pathlib import Path
ROOT = Path(r'C:\Elysia\Core')

def repair_linguistics(file_path):
    try:
        content = file_path.read_text(encoding='utf-8')
        # Fix the specific error seen in logs
        new_content = re.sub(r'Core\._02_Intelligence\._02_Memory\.Domains\.linguistics', r'Core._02_Intelligence._02_Memory.Domains.linguistics', content)
        # Fix any remaining legacy Memory_Linguistics
        new_content = re.sub(r'Core\._02_Intelligence\._02_Memory_Linguistics', r'Core._02_Intelligence._02_Memory', new_content)
        
        if new_content != content:
            file_path.write_text(new_content, encoding='utf-8')
            print(f'Fixed: {file_path}')
    except: pass

if __name__ == "__main__":
    for py in ROOT.rglob("*.py"):
        repair_linguistics(py)
