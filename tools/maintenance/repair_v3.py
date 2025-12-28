import os
import re
from pathlib import Path

ROOT = Path(r'C:\Elysia\Core')

SUBSTITUTIONS = [
    # 1. Level 1 (Top Layer)
    (r'Core\.01_Foundation', r'Core._01_Foundation'),
    (r'Core\.02_Intelligence', r'Core._02_Intelligence'),
    (r'Core\.03_Interaction', r'Core._03_Interaction'),
    (r'Core\.04_Evolution', r'Core._04_Evolution'),
    (r'Core\.05_Systems', r'Core._05_Systems'),

    # 2. Level 2 (Sub Layer)
    (r'Core\._01_Foundation\.Foundation', r'Core._01_Foundation._02_Logic'),
    (r'Core\._01_Foundation\.Logic', r'Core._01_Foundation._02_Logic'),
    (r'Core\._01_Foundation\.Ethics', r'Core._01_Foundation._03_Ethics'),
    (r'Core\._01_Foundation\.Governance', r'Core._01_Foundation._04_Governance'),
    (r'Core\._01_Foundation\.Security', r'Core._01_Foundation._05_Security'),
    
    (r'Core\._02_Intelligence\.Reasoning', r'Core._02_Intelligence._01_Reasoning'),
    (r'Core\._02_Intelligence\.Memory', r'Core._02_Intelligence._02_Memory'),
    (r'Core\._02_Intelligence\.Physics', r'Core._02_Intelligence._03_Physics'),
    (r'Core\._02_Intelligence\.Mind', r'Core._02_Intelligence._04_Mind'),
    (r'Core\._02_Intelligence\.Research', r'Core._02_Intelligence._05_Research'),
    
    (r'Core\._03_Interaction\.Sensory', r'Core._03_Interaction._01_Sensory'),
    (r'Core\._03_Interaction\.Interface', r'Core._03_Interaction._02_Interface'),
    (r'Core\._03_Interaction\.Expression', r'Core._03_Interaction._03_Expression'),
    (r'Core\._03_Interaction\.Network', r'Core._03_Interaction._04_Network'),
    (r'Core\._03_Interaction\.Integration', r'Core._03_Interaction._05_Integration'),

    (r'Core\._04_Evolution\.Growth', r'Core._04_Evolution._01_Growth'),
    (r'Core\._04_Evolution\.Learning', r'Core._04_Evolution._02_Learning'),
    (r'Core\._04_Evolution\.Creative', r'Core._04_Evolution._03_Creative'),

    # 3. Specific Legacy Patterns
    (r'Core\._01_Foundation\._04_Governance\.Foundation', r'Core._01_Foundation._02_Logic'),
    (r'Core\._01_Foundation\._02_Logic\.Foundation', r'Core._01_Foundation._02_Logic'),
    (r'Core\._02_Intelligence\._01_Reasoning\.Cognition', r'Core._02_Intelligence._01_Reasoning'),

    # 4. Digit normalization (Safety net)
    (r'Core\.(\d{2})', r'Core._\1'),
    (r'\.(\d{2})_', r'._\1_'),
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
    except Exception as e:
        print(f'Error {file_path}: {e}')

if __name__ == "__main__":
    for py in ROOT.rglob("*.py"):
        repair(py)
