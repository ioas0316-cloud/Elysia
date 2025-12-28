import os
import re
from pathlib import Path
ROOT = Path(r'C:\Elysia\Core')

NESTED_MAP = {
    r'Core\._02_Intelligence\._02_Memory\.Domains\.linguistics\.Memory\.unified_experience_core': r'Core._02_Intelligence._02_Memory.unified_experience_core',
    r'Core\._02_Intelligence\._02_Memory\.Domains\.linguistics\.Memory\.Graph\.hyper_graph': r'Core._02_Intelligence._02_Memory.Graph.hyper_graph',
    r'Core\._02_Intelligence\._02_Memory\.Domains\.linguistics\.Memory\.potential_causality': r'Core._02_Intelligence._02_Memory.potential_causality',
    r'Core\._02_Intelligence\._02_Memory\.Domains\.linguistics\.Memory\.Graph\.knowledge_graph': r'Core._02_Intelligence._02_Memory.Graph.knowledge_graph',
}

def deep_repair(file_path):
    try:
        content = file_path.read_text(encoding='utf-8')
        new_content = content
        for pattern, replacement in NESTED_MAP.items():
            new_content = re.sub(pattern, replacement, new_content)
        
        if new_content != content:
            file_path.write_text(new_content, encoding='utf-8')
            print(f'Deep Fix: {file_path}')
    except: pass

if __name__ == "__main__":
    for py in ROOT.rglob("*.py"):
        deep_repair(py)
