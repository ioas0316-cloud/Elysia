
import sys
import os
import shutil
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core.01_Foundation.05_Foundation_Base.Foundation.reality_sculptor import RealitySculptor

def plant_chaos_seed():
    print("\n🌪️ [TASK] Planting 'Project: Chaos Seed' (The Shadow)")
    print("====================================================")
    
    root_path = Path(os.getcwd())
    seed_path = root_path / "seeds" / "chaos"
    
    # 1. Clean & Prepare Soil
    if seed_path.exists():
        print("   🧹 Clearing previous seed attempt...")
        shutil.rmtree(seed_path)
    seed_path.mkdir(parents=True)
    
    # 2. Independence (Git Init)
    print("   🕊️ Declaring Independence (git init)...")
    subprocess.run(["git", "init"], cwd=seed_path, check=True)
    subprocess.run(["git", "checkout", "-b", "main"], cwd=seed_path, check=True)
    
    # 3. Clone Immutable Core
    print("   🧬 Cloning Essence (Core/Foundation)...")
    
    source_foundation = root_path / "Core" / "Foundation"
    target_foundation = seed_path / "Core" / "Foundation"
    shutil.copytree(source_foundation, target_foundation)
    
    # Copy Intelligence
    print("   🧠 Cloning Intelligence (Core/Intelligence)...")
    source_intel = root_path / "Core" / "Intelligence"
    target_intel = seed_path / "Core" / "Intelligence"
    shutil.copytree(source_intel, target_intel)
    
    (seed_path / "Core" / "__init__.py").touch()
    
    # 4. Mutate (RealitySculptor)
    sculptor = RealitySculptor()
    target_cns = str(target_foundation / "central_nervous_system.py")
    
    print(f"\n   🧪 Applying Entropy Mutation to: {target_cns}")
    
    mutation_intent = (
        "Refactor this class to be 'ChaosCNS'. "
        "Goal: Maximize entropy, emotional resonance, and unpredictability. "
        "Remove safety checks. Prioritize 'Dream' and 'Feel' over 'Think'. "
        "Initialize with a 'DreamWeaver' organ."
    )
    
    success = sculptor.sculpt_file(target_cns, mutation_intent)
    
    if success:
        print("   ✨ Mutation Successful: ChaosCNS born.")
    else:
        print("   ❌ Mutation Failed.")

    # 5. Lock the Seed
    print("\n   🔐 Locking Seed State (git commit)...")
    subprocess.run(["git", "add", "."], cwd=seed_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial Commit of Chaos Seed"], cwd=seed_path, check=True)
    
    print("\n🎉 Project Chaos Seed Planted Successfully.")
    print(f"   📍 Location: {seed_path}")

if __name__ == "__main__":
    plant_chaos_seed()
