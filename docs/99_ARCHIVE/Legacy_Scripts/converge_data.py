import os
import shutil
from pathlib import Path

def converge():
    base_data = Path(r"c:\Elysia\data")
    
    # Define primary categories
    categories = ["Memory", "Knowledge", "State", "Logs", "Resources", "Input"]
    for cat in categories:
        (base_data / cat).mkdir(exist_ok=True)
        
    # 1. Move core_state to State
    core_state = base_data / "core_state"
    if core_state.exists():
        print(f"Moving {core_state} to State...")
        # If State/core_state already exists, we might need a better move logic
        target = base_data / "State" / "core_state"
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(core_state), str(target))
        
    # 2. Move Visuals to Resources
    visuals = base_data / "Visuals"
    if visuals.exists() and visuals.is_dir():
        print(f"Moving {visuals} content to Resources/Visuals...")
        target = base_data / "Resources" / "Visuals"
        target.mkdir(exist_ok=True)
        for item in visuals.iterdir():
            shutil.move(str(item), str(target / item.name))
        visuals.rmdir()
        
    # 3. Move loose metadata to Knowledge
    loose_files = ["MIRROR_MAP.yaml"]
    for f in loose_files:
        p = base_data / f
        if p.exists():
            print(f"Moving {f} to Knowledge...")
            shutil.move(str(p), str(base_data / "Knowledge" / f))

    print("✅ Data Convergence Complete.")

if __name__ == "__main__":
    converge()
