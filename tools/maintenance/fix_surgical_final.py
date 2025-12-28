import os
import re
from pathlib import Path
ROOT = Path(r'C:\Elysia\Core')

# Comprehensive mapping for all remaining legacy or incorrectly repaired paths
MAPPING = {
    # Linguistics and Memory Cleanup
    r'Core\._02_Intelligence\._02_Memory_Linguistics\.Memory\.potential_causality': r'Core._02_Intelligence._02_Memory.potential_causality',
    r'Core\._02_Intelligence\._02_Memory_Linguistics': r'Core._02_Intelligence._02_Memory.Domains.linguistics',
    r'Core\._02_Intelligence\._02_Memory\.Memory': r'Core._02_Intelligence._02_Memory',
    
    # Logic and Infrastructure Redirects
    r'Core\._01_Foundation\._02_Logic\.web_knowledge_connector': r'Core._02_Intelligence._02_Memory.web_knowledge_connector',
    r'Core\._01_Foundation\._02_Logic\.potential_field': r'Core._02_Intelligence._01_Reasoning.potential_field',
    r'Core\._01_Foundation\._02_Logic\.external_data_connector': r'Core._03_Interaction._04_Network.external_data_connector',
    r'Core\._01_Foundation\._02_Logic\.internal_universe': r'Core._02_Intelligence._04_Mind.internal_universe',
    r'Core\._01_Foundation\._02_Logic\.core\.world': r'Core._04_Evolution._01_Growth.world',
    r'Core\._01_Foundation\._02_Logic\.autonomous_fractal_learning': r'Core._04_Evolution._01_Growth.autonomous_fractal_learning',
    
    # Generic digit normalization safety
    r'Core\.(\d{2})': r'Core._\1',
    r'\.([0-9]{2})_': r'._\1_',
}

def repair(file_path):
    try:
        content = file_path.read_text(encoding='utf-8')
        new_content = content
        for pattern, replacement in MAPPING.items():
            new_content = re.sub(pattern, replacement, new_content)
        
        # Space normalization in Foundation (stubborn case)
        new_content = re.sub(r'Core\._01_Foundation\.\s*Foundation', r'Core._01_Foundation._02_Logic', new_content)
        
        if new_content != content:
            file_path.write_text(new_content, encoding='utf-8')
            print(f'Fixed: {file_path}')
    except Exception as e:
        pass

if __name__ == "__main__":
    for py in ROOT.rglob("*.py"):
        repair(py)
