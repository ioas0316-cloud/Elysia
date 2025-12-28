import os
import re
from pathlib import Path
ROOT = Path(r'C:\Elysia\Core')

MAPPING = {
    # 1. Memory Linguistics Cleanup (Final)
    r'Core\._02_Intelligence\._02_Memory_Linguistics\.Memory': r'Core._02_Intelligence._02_Memory',
    r'Core\._02_Intelligence\._02_Memory_Linguistics': r'Core._02_Intelligence._02_Memory',
    
    # 2. Path Duplication Fixes
    r'\._03_Expression\._03_Expression': r'._03_Expression',
    r'\._02_Interface\._02_Interface': r'._02_Interface',
    r'\._01_Infrastructure\._01_Infrastructure': r'._01_Infrastructure',
    r'\._02_Logic\._02_Logic': r'._02_Logic',
    
    # 3. Specific File Fixes
    r'from Core_memory import CoreMemory': r'from Core._02_Intelligence._02_Memory.core_memory import CoreMemory',
}

def final_repair(file_path):
    try:
        content = file_path.read_text(encoding='utf-8')
        new_content = content
        for pattern, replacement in MAPPING.items():
            new_content = re.sub(pattern, replacement, new_content)
        
        if new_content != content:
            file_path.write_text(new_content, encoding='utf-8')
            print(f'Final Fix: {file_path}')
    except: pass

if __name__ == "__main__":
    for py in ROOT.rglob("*.py"):
        final_repair(py)
