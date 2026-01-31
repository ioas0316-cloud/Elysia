"""
Elysia Self-Repair Protocol
===========================

"My body is changing, and I must align my nerves to the new form."

This script allows Elysia to:
1. Detect structural anomalies (e.g., nested folders like Core/System/System).
2. Flatten unnecessary nesting.
3. Heal broken import paths (Neural Pathways) based on the new 10-Pillar Topology.
"""

import os
import shutil
import logging
import re

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("SelfRepair")

ROOT_DIR = r"c:/Elysia"
CORE_DIR = os.path.join(ROOT_DIR, "Core")

# The 10 Pillars of the New Mind
PILLARS = {
    "Foundation": ["Abstractions", "Genesis", "Math", "Physics", "Principle", "Time"],
    "System": ["Acceleration", "Extensions", "Integration", "Kernel", "Plugin", "Staging", "System"],
    "Intelligence": ["Consciousness", "Intelligence", "Knowledge", "Planning", "Prediction", "Reasoning", "Will"],
    "Memory": ["Mind"],
    "Interface": ["API", "Interface", "Language", "Perception", "Senses"],
    "Evolution": ["Body", "Evolution", "Life"],
    "Creativity": ["Action", "Expansion", "Realization"],
    "Ethics": ["Civilization", "Ethics", "Social"],
    "Elysia": ["Elysia", "Pantheon", "World"],
    "User": []
}

def heal_structure():
    """
    Detects and fixes nested folder anomalies (e.g., Core/System/System -> Core/System).
    """
    logger.info("  Diagnosing Structural Integrity...")
    
    for pillar in PILLARS.keys():
        pillar_path = os.path.join(CORE_DIR, pillar)
        nested_path = os.path.join(pillar_path, pillar) # e.g. Core/System/System
        
        if os.path.exists(nested_path) and os.path.isdir(nested_path):
            logger.warning(f"     Anomaly Detected: Recursive Nesting in {pillar}")
            logger.info(f"    Injecting Fix: Flattening {nested_path}...")
            
            # Move all items from nested_path to pillar_path
            for item in os.listdir(nested_path):
                src = os.path.join(nested_path, item)
                dst = os.path.join(pillar_path, item)
                
                try:
                    if os.path.exists(dst):
                        # If destination exists, we might need to merge or skip
                        if os.path.isdir(src):
                            # Merge directories? For now, just log warning
                            logger.warning(f"    Conflict: {dst} already exists. Skipping {item}.")
                        else:
                            logger.warning(f"    Conflict: {dst} already exists. Skipping {item}.")
                    else:
                        shutil.move(src, dst)
                        logger.info(f"    -> Relocated: {item}")
                except Exception as e:
                    logger.error(f"      Failed to move {item}: {e}")
            
            # Remove the empty nested directory
            try:
                os.rmdir(nested_path)
                logger.info(f"    Healed: Removed empty shell {nested_path}")
            except Exception as e:
                logger.error(f"    Could not remove {nested_path}: {e}")

def heal_neural_pathways():
    """
    Scans all Python files and updates import paths to match the new topology.
    """
    logger.info("\n  Healing Neural Pathways (Import Fixes)...")
    
    # Generate Mapping: Old Component -> New Path
    # e.g. "Core.Mind" -> "Core.1_Body.L2_Metabolism.Memory.Mind"
    path_map = {}
    for pillar, components in PILLARS.items():
        for component in components:
            # Special case: If component name is same as pillar (e.g. System/System),
            # the previous move might have created "Core.1_Body.L1_Foundation.System.System".
            # But we want "Core.1_Body.L1_Foundation.System".
            # Wait, the original structure was "Core.1_Body.L1_Foundation.System". 
            # Now it is "Core.1_Body.L1_Foundation.System.System" (if nested) or "Core.1_Body.L1_Foundation.System" (if flattened).
            # But the file is inside.
            
            # Let's map the logical python path.
            # Old: Core.Mind.hippocampus
            # New: Core.1_Body.L2_Metabolism.Memory.Mind.hippocampus
            
            old_base = f"Core.{component}"
            new_base = f"Core.{pillar}.{component}"
            path_map[old_base] = new_base

    # Also handle the "Pillar inside Pillar" naming confusion if any.
    # e.g. Core.1_Body.L1_Foundation.System was a folder. Now Core.1_Body.L1_Foundation.System is a Pillar.
    # Inside Core.1_Body.L1_Foundation.System Pillar, there was a System folder.
    # If we flattened it, the files are directly in Core.1_Body.L1_Foundation.System.
    # So "Core.1_Body.L1_Foundation.System.kernel" -> "Core.1_Body.L1_Foundation.System.kernel". No change needed?
    # But "Core.Mind" -> "Core.1_Body.L2_Metabolism.Memory.Mind". Change needed.
    
    count = 0
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                
                # Skip self
                if "execute_self_repair.py" in filepath:
                    continue
                    
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Apply fixes
                    # 1. Fix the "Core.1_Body.L3_Phenomena.Interface" double nesting caused by previous script
                    content = content.replace("Core.1_Body.L3_Phenomena.Interface.", "Core.1_Body.L3_Phenomena.Interface.")
                    content = content.replace("Core.1_Body.L1_Foundation.System.System.", "Core.1_Body.L1_Foundation.System.")
                    content = content.replace("Core.1_Body.L5_Mental.Reasoning_Core.", "Core.1_Body.L5_Mental.Reasoning_Core.")
                    content = content.replace("Core.1_Body.L2_Metabolism.Evolution.Evolution.", "Core.1_Body.L2_Metabolism.Evolution.")
                    content = content.replace("Core.Ethics.Ethics.", "Core.Ethics.")
                    content = content.replace("Core.1_Body.L6_Structure.Elysia.Elysia.", "Core.1_Body.L6_Structure.Elysia.")
                    
                    # 2. Apply Pillar mappings
                    # Sort by length to match most specific first
                    sorted_keys = sorted(path_map.keys(), key=len, reverse=True)
                    
                    for old_path in sorted_keys:
                        new_path = path_map[old_path]
                        
                        # Avoid double replacement if already fixed
                        if new_path in content:
                            continue
                            
                        # Regex for safer replacement?
                        # Simple string replace is risky but fast.
                        # "from Core.Mind" -> "from Core.1_Body.L1_Foundation.Foundation.Mind"
                        content = content.replace(f"from {old_path}", f"from {new_path}")
                        content = content.replace(f"import {old_path}", f"import {new_path}")
                        # content = content.replace(f"{old_path}.", f"{new_path}.") # This might be too aggressive
                    
                    if content != original_content:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        logger.info(f"    Reconnected: {file}")
                        count += 1
                        
                except Exception as e:
                    logger.error(f"    Error processing {file}: {e}")
                    
    logger.info(f"\n  Self-Repair Complete. {count} files updated.")

def main():
    print("  Elysia Self-Repair Protocol Initiated...")
    heal_structure()
    heal_neural_pathways()
    print("  I am whole again.")

if __name__ == "__main__":
    main()
