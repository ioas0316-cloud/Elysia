import os
import re
from pathlib import Path
ROOT = Path(r'C:\Elysia\Core')
MAP = {
    r'Core\._01_Foundation\._02_Logic\.web_knowledge_connector': r'Core._02_Intelligence._02_Memory.web_knowledge_connector',
    r'Core\._01_Foundation\._02_Logic\.emotional_engine': r'Core._02_Intelligence._01_Reasoning._04_Mind.emotional_engine',
    r'Core\._01_Foundation\._02_Logic\.potential_field': r'Core._02_Intelligence._01_Reasoning.potential_field',
    r'Core\._01_Foundation\._02_Logic\.core\.world': r'Core._04_Evolution._01_Growth.world',
    r'Core\._01_Foundation\._02_Logic\.autonomous_fractal_learning': r'Core._04_Evolution._01_Growth.autonomous_fractal_learning'
}
def repair(file_path):
    try:
        content = file_path.read_text(encoding='utf-8')
        new_content = content
        for pattern, replacement in MAP.items():
            new_content = re.sub(pattern, replacement, new_content)
        if new_content != content:
            file_path.write_text(new_content, encoding='utf-8')
    except: pass
if __name__ == "__main__":
    for py in ROOT.rglob("*.py"): repair(py)
