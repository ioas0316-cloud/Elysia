import os
import re
from pathlib import Path
ROOT = Path(r'C:\Elysia\Core')
def repair(file_path):
    try:
        content = file_path.read_text(encoding='utf-8')
        new_content = re.sub(r'Core\._01_Foundation\._02_Logic\.external_data_connector', r'Core._03_Interaction._04_Network.external_data_connector', content)
        if new_content != content:
            file_path.write_text(new_content, encoding='utf-8')
    except: pass
if __name__ == "__main__":
    for py in ROOT.rglob("*.py"): repair(py)
