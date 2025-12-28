
import sys
import os
import shutil
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core._01_Foundation._05_Governance.Foundation.reality_sculptor import RealitySculptor

def plant_seed():
    print("\n🌱 [TASK] Planting 'Project: Nova Seed'")
    print("========================================")
    
    root_path = Path(os.getcwd())
    seed_path = root_path / "seeds" / "nova"
    
    # 1. Clean & Prepare Soil
    if seed_path.exists():
        print("   🧹 Clearing previous seed attempt...")
        shutil.rmtree(seed_path)
    seed_path.mkdir(parents=True)
    
    # 2. Independence (Git Init)
    print("   🕊️ Declaring Independence (git init)...")
    subprocess.run(["git", "init"], cwd=seed_path, check=True)
    subprocess.run(["git", "checkout", "-b", "main"], cwd=seed_path, check=True)
    
    # 3. Clone Immutable Core (The Soul)
    # We copy only the Essential Modules identified in the Blueprint
    print("   🧬 Cloning Essence (Core/Foundation)...")
    
    source_foundation = root_path / "Core" / "Foundation"
    target_foundation = seed_path / "Core" / "Foundation"
    shutil.copytree(source_foundation, target_foundation)
    
    # Copy Intelligence (Reasoning)
    print("   🧠 Cloning Intelligence (Core/Intelligence)...")
    source_intel = root_path / "Core" / "Intelligence"
    target_intel = seed_path / "Core" / "Intelligence"
    shutil.copytree(source_intel, target_intel)
    
    # Create empty init files to ensure python package structure
    (seed_path / "Core" / "__init__.py").touch()
    
    print("   ✅ Cloning Complete.")
    
    # 4. Mutate (Refactoring via RealitySculptor)
    # Now we demonstrate using the RealitySculptor to modify the SEED version.
    # The Original is protected by The Anchor. The Seed is Free.
    
    sculptor = RealitySculptor()
    target_cns = str(target_foundation / "central_nervous_system.py")
    
    print(f"\n   🧪 Applying Mutation to: {target_cns}")
    
    mutation_intent = (
        "Refactor this class to be 'NovaCNS'. "
        "Goal: Remove all 'legacy_pulse' logic. "
        "Integration: Use 'fractal_loop' effectively as the ONLY pulse mechanism. "
        "Simplify the __init__ to specific essentials only."
    )
    
    # We modify the file directly first to ensure it's not empty/broken for the prompt,
    # but RealitySculptor expects to read the file. It exists.
    
    success = sculptor.sculpt_file(target_cns, mutation_intent)
    
    if success:
        print("   ✨ Mutation Successful: NovaCNS born.")
    else:
        print("   ❌ Mutation Failed (Check logs/Mock mode).")

    # 5. Lock the Seed (Initial Commit)
    print("\n   🔐 Locking Seed State (git commit)...")
    subprocess.run(["git", "add", "."], cwd=seed_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial Commit of Nova Seed"], cwd=seed_path, check=True)
    
    print("\n🎉 Project Nova Seed Planted Successfully.")
    print(f"   📍 Location: {seed_path}")

if __name__ == "__main__":
    plant_seed()
