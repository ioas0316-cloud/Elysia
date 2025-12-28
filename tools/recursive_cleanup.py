import os
import shutil
from pathlib import Path

ROOT = Path(r"C:\Elysia\Core")
MAP = {
    # Intelligence mapping
    "Cognition": "_01_Reasoning",
    "Science": "_05_Research",
    "Holographic": "_03_Physics",
    "Emotion": "_04_Mind",
    
    # Interaction mapping
    "Action": "_05_Integration",
    "Communication": "_04_Network",
    "Expression": "_03_Expression",
    "Sensory": "_01_Sensory",
    "Interface": "_02_Interface",
    "Visual": "_01_Sensory",
    "Synesthesia": "_01_Sensory",
    "Multimodal": "_01_Sensory",
    "VR": "_05_Integration",
    
    # Evolution mapping
    "Autonomy": "_01_Growth",
    "Evolution": "_01_Growth",
    "Learning": "_02_Learning",
    "Creation": "_03_Creative",
    "Creativity": "_03_Creative",
    "Studio": "_03_Creative",
    
    # Foundation mapping
    "Elysia": "_01_Infrastructure",
    "Logic": "_02_Logic",
    "Ethics": "_03_Ethics",
    "Philosophy": "_02_Logic",
    "Security": "_05_Security",
    "Governance": "_04_Governance"
}

def clean_recursive(path: Path):
    if not path.exists(): return
    for item in list(path.iterdir()):
        if item.is_dir():
            name = item.name
            
            # Skip magic or already standardized
            if name.startswith("_") or name == "elysia_core" or name == "__pycache__":
                clean_recursive(item)
                continue
                
            # Try to map know names
            target_name = MAP.get(name)
            
            # If name starts with digit like '01_Name', convert to '_01_Name'
            if not target_name and name[0].isdigit():
                if "_" in name:
                    target_name = "_" + name
                else:
                    # Match pattern like '01Name' -> '_01_Name'
                    import re
                    match = re.match(r"(\d+)(.*)", name)
                    if match:
                        num, rest = match.groups()
                        target_name = f"_{num.zfill(2)}_{rest.strip()}"

            if target_name:
                target_path = path / target_name
                target_path.mkdir(exist_ok=True)
                if not (target_path / "__init__.py").exists() and "Core" in str(target_path):
                    (target_path / "__init__.py").touch()
                
                print(f"Merging {item} -> {target_path}")
                for sub in item.iterdir():
                    dest = target_path / sub.name
                    if dest.exists() and dest.is_dir():
                        # Recursive merge
                        for ssub in sub.iterdir():
                            shutil.move(str(ssub), str(dest / ssub.name))
                        sub.rmdir()
                    else:
                        shutil.move(str(sub), str(dest))
                item.rmdir()
                clean_recursive(target_path)
            else:
                # Still recurse into non-standard folders to find nested standard ones
                clean_recursive(item)

if __name__ == "__main__":
    for domain in ["Core", "data", "docs"]:
        print(f"\n--- Cleaning {domain} ---")
        clean_recursive(Path(r"C:\Elysia") / domain)
