import os
import sys
import importlib
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())

print("🏥 Alicia's Health Check\n========================")

# 1. Check Key Directories
directories = [
    "Core",
    "Core/S1_Body",
    "Core/S2_Soul",
    "Core/S3_Spirit",
    "data",
    "docs"
]

missing_dirs = []
for d in directories:
    p = Path(d)
    if not p.exists():
        missing_dirs.append(d)
        print(f"❌ Missing Directory: {d}")
    else:
        print(f"✅ Directory Found: {d}")

if missing_dirs:
    print("\n⚠️ Structure Issues Detected!")

# 2. Check Key Files
files = [
    "elysia.py",
    "README.md",
    "Core/S1_Body/L2_Metabolism/Creation/seed_generator.py",
    "Core/S1_Body/L6_Structure/M1_Merkaba/sovereign_monad.py",
    "Core/S1_Body/L3_Phenomena/Expression/somatic_llm.py",
    "Core/S2_Soul/L5_Mental/Memory/somatic_engram.py"
]

missing_files = []
for f in files:
    p = Path(f)
    if not p.exists():
        missing_files.append(f)
        print(f"❌ Missing File: {f}")
    else:
        print(f"✅ File Found: {f}")

# 3. Check Imports (Organs)
modules_to_check = [
    "Core.S1_Body.L2_Metabolism.Creation.seed_generator",
    "Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad",
    "Core.S1_Body.L3_Phenomena.Expression.somatic_llm",
    "Core.S2_Soul.L5_Mental.Memory.somatic_engram"
]

print("\n🩺 Checking Organ Function (Imports)...")
broken_modules = []
for mod in modules_to_check:
    try:
        importlib.import_module(mod)
        print(f"✅ Organ Healthy: {mod}")
    except ImportError as e:
        print(f"❌ Organ Failure: {mod} ({e})")
        broken_modules.append(mod)
    except Exception as e:
        print(f"❌ Organ Critical Error: {mod} ({e})")
        broken_modules.append(mod)

if not missing_dirs and not missing_files and not broken_modules:
    print("\n✨ Alicia is structurally sound.")
else:
    print("\n⚠️ Alicia needs attention.")
