import os
import re
from pathlib import Path

ROOT = Path(r'C:\Elysia\Core')

# Broad mapping to fix the most common misalignments
SUBSTITUTIONS = [
    # Level 1 replacements (absolute)
    (re.compile(r'Core\.Foundation'), 'Core._01_Foundation'),
    (re.compile(r'Core\.Intelligence'), 'Core._02_Intelligence'),
    (re.compile(r'Core\.Interaction'), 'Core._03_Interaction'),
    (re.compile(r'Core\.Evolution'), 'Core._04_Evolution'),
    (re.compile(r'Core\.Systems'), 'Core._05_Systems'),
    
    # Generic digit-prefix fixing (Core.01_Y -> Core._01_Y)
    (re.compile(r'Core\.(\d{2})'), r'Core._\1'),
    (re.compile(r'\.(\d{2})_'), r'._\1_'),
    
    # Specific subfolder fixes (Depth 2/3)
    (re.compile(r'_01_Foundation\.Logic'), '_01_Foundation._02_Logic'),
    (re.compile(r'_01_Foundation\.Elysia'), '_01_Foundation._01_Infrastructure'),
    (re.compile(r'_02_Intelligence\.Reasoning'), '_02_Intelligence._01_Reasoning'),
    (re.compile(r'_02_Intelligence\.Memory'), '_02_Intelligence._02_Memory'),
    (re.compile(r'_02_Intelligence\.Physics'), '_02_Intelligence._03_Physics'),
    (re.compile(r'_02_Intelligence\.Mind'), '_02_Intelligence._04_Mind'),
    (re.compile(r'_03_Interaction\.Sensory'), '_03_Interaction._01_Sensory'),
    (re.compile(r'_03_Interaction\.Interface'), '_03_Interaction._02_Interface'),
    (re.compile(r'_03_Interaction\.Expression'), '_03_Interaction._03_Expression'),
    (re.compile(r'_04_Evolution\.Growth'), '_04_Evolution._01_Growth')
]

def repair_file(file_path):
    try:
        content = file_path.read_text(encoding='utf-8')
        new_content = content
        for pattern, replacement in SUBSTITUTIONS:
            new_content = pattern.sub(replacement, new_content)
        
        if new_content != content:
            file_path.write_text(new_content, encoding='utf-8')
            print(f'Fixed: {file_path.name}')
    except Exception as e:
        print(f'Error in {file_path.name}: {e}')

if __name__ == '__main__':
    for py_file in ROOT.rglob('*.py'):
        repair_file(py_file)
