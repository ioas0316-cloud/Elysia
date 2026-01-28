import os
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PathFixer")

ROOT_DIR = "c:/Elysia/Core"

PATH_REPLACEMENTS = {
    # Legacy Pattern: New Pattern
    r"(c:/Elysia/)?data/L1_Foundation/DNA": "data/L1_Foundation/M2_DNA",
    r"(c:/Elysia/)?data/Logs": "data/L1_Foundation/M4_Logs",
    r"(c:/Elysia/)?data/L1_Foundation/Logs": "data/L1_Foundation/M4_Logs",
    r"(c:/Elysia/)?data/State": "data/L1_Foundation/M1_System",
    r"(c:/Elysia/)?data/Sandbox": "data/L2_Metabolism/M2_Incubation",
    r"(c:/Elysia/)?data/L1_Foundation/Sandbox": "data/L2_Metabolism/M2_Incubation",
    r"(c:/Elysia/)?data/Qualia": "data/L3_Phenomena/M1_Qualia",
    r"(c:/Elysia/)?data/L1_Foundation/Qualia": "data/L3_Phenomena/M1_Qualia",
    r"(c:/Elysia/)?data/L5_Mental/Qualia": "data/L3_Phenomena/M1_Qualia",
    r"(c:/Elysia/)?data/04_Causality": "data/L4_Causality/M4_Chronicles",
    r"(c:/Elysia/)?data/Memory": "data/L5_Mental/M1_Memory",
    r"(c:/Elysia/)?data/L5_Mental/Memory": "data/L5_Mental/M1_Memory",
    r"(c:/Elysia/)?data/L5_Mental/Memories": "data/L5_Mental/M1_Memory",
    r"(c:/Elysia/)?data/Identity": "data/L7_Spirit/M3_Sovereignty",
    r"(c:/Elysia/)?data/self_manifest.json": "data/L7_Spirit/M3_Sovereignty/self_manifest.json",
    r"(c:/Elysia/)?data/psionic_state.json": "data/L6_Structure/M1_State/psionic_state.json",
    r"(c:/Elysia/)?data/State/hypersphere_memory.json": "data/L6_Structure/M1_State/hypersphere_memory.json",
    
    # Internal layer fixes
    r"data/L5_Mental/Learning": "data/L5_Mental/M4_Learning",
    r"data/L5_Mental/Knowledge": "data/L5_Mental/M5_Knowledge",
    r"data/L5_Mental/Intelligence": "data/L5_Mental/M5_Knowledge",
    r"data/L1_Foundation/State/manifold_registry.json": "data/L1_Foundation/M1_System/manifold_registry.json",
    r"data/L7_Spirit/Chronicles/sovereign_journal.md": "data/L7_Spirit/M3_Sovereignty/sovereign_journal.md", # Adjusting for the journal specifically
}

# Special case for the journal path used in multiple places
SPECIFIC_FIXES = {
    "c:/Elysia/data/L7_Spirit/Chronicles/sovereign_journal.md": "c:/Elysia/data/L7_Spirit/M3_Sovereignty/sovereign_journal.md",
    "data/L7_Spirit/Chronicles/sovereign_journal.md": "data/L7_Spirit/M3_Sovereignty/sovereign_journal.md"
}

def fix_paths():
    count = 0
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            if file.endswith((".py", ".md", ".json", ".txt")):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    new_content = content
                    # Apply regex replacements
                    for pattern, replacement in PATH_REPLACEMENTS.items():
                        new_content = re.sub(pattern, replacement, new_content)
                    
                    # Apply exact fixes
                    for old, new in SPECIFIC_FIXES.items():
                        new_content = new_content.replace(old, new)

                    if new_content != content:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        logger.info(f"✅ Fixed paths in: {file_path}")
                        count += 1
                except Exception as e:
                    logger.error(f"❌ Failed to process {file_path}: {e}")

    logger.info(f"✨ Path Fixer complete. Updated {count} files.")

if __name__ == "__main__":
    fix_paths()
