import os
import re
from pathlib import Path

ROOT = Path(r'C:\Elysia\Core')

SUBSTITUTIONS = [
    # 1. Root Layer _NN Mapping
    (r'Core\.0(\d)', r'Core._0\1'),
    
    # 2. Level 2 (Sub Layer) Mapping
    (r'Core\._01_Foundation\.(Infrastructure|Base|Logic|Ethics|Governance|Security|Foundation|Philosophy|Math)', 
     lambda m: {
         'Infrastructure': 'Core._01_Foundation._01_Infrastructure',
         'Base': 'Core._01_Foundation._01_Infrastructure',
         'Logic': 'Core._01_Foundation._02_Logic',
         'Math': 'Core._01_Foundation._02_Logic',
         'Philosophy': 'Core._01_Foundation._02_Logic',
         'Foundation': 'Core._01_Foundation._02_Logic',
         'Ethics': 'Core._01_Foundation._03_Ethics',
         'Governance': 'Core._01_Foundation._04_Governance',
         'Security': 'Core._01_Foundation._05_Security'
     }[m.group(1)]),
     
    # More general cleanups
    (r'\.Foundation\.', r'._02_Logic.'),
    (r'\.Logic\.', r'._02_Logic.'),
    (r'\.Math\.', r'._02_Logic.'),
    (r'\.Philosophy\.', r'._02_Logic.'),
    (r'\.Sensory\.', r'._01_Sensory.'),
    (r'\.Interface\.', r'._02_Interface.'),
    (r'\.Expression\.', r'._03_Expression.'),
]

def repair(file_path):
    try:
        content = file_path.read_text(encoding='utf-8')
        new_content = content
        for pattern, replacement in SUBSTITUTIONS:
            if callable(replacement):
                new_content = re.sub(pattern, replacement, new_content)
            else:
                new_content = re.sub(pattern, replacement, new_content)
        
        if new_content != content:
            file_path.write_text(new_content, encoding='utf-8')
            print(f'Fixed: {file_path}')
    except Exception as e:
        pass

if __name__ == "__main__":
    for py in ROOT.rglob("*.py"):
        repair(py)
