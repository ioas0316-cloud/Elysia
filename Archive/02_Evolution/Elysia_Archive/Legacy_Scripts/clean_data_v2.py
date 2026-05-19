import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("DataCleaner")

ROOT = r"c:\Elysia\data"
ARCHIVE = os.path.join(ROOT, "Archive")

# Destination Categories
TARGET_DIRS = ["Memory", "State", "Logs", "Knowledge", "Visuals", "Archive"]

# Mapping items to categories
FILE_MAP = {
    "Memory": ["elysia_core_memory.json", "chunks.json", "synapse.json", "synaptic_memory.json", "memory.db", "conversation_memory.json", "emotional_memory.json"],
    "State": ["core_state", "meta_agent_state.json", "self.json", "immune_system_state.json", "emergent_self.json", "brain_state.pt", "network_shield_state.json", "system_evaluation.json", "system_connection_audit.json"],
    "Logs": ["awakening_log.txt", "growth_history.json", "self_improvement_reports", "nanocell_report.json", "elysia_challenge_report.json"],
    "Knowledge": ["concept_dictionary.json", "potential_knowledge.json", "hierarchical_knowledge.json", "akashic_records.json", "protocol_kg.json", "tools_kg.json", "wave_knowledge.json", "central_registry.json", "unified_spatial_index.json", "world_tree_with_language.json"],
    "Visuals": ["wave_organization.html", "monitor_echo.png", "monitor_kg.png", "resonance_graph.png", "wave_visualization_point.png", "generated_images", "storybook_images"],
    "Archive": ["01_Core", "_01_Core", "_02_Cognitive", "_03_Experiential", "_04_Extended", "_05_Metabolic", "test_fs", "test_deep_a.json", "test_deep_b.json", "test_potential.json", "test_waves.json", "test_hierarchical.json", "dense_demo.json", "plasma_demo.json", "resonance_demo.json", "ideal_self_demo.json", "absorbed_hashes.json"]
}

def clean_data():
    print("🧹 Cleaning data directory (v2)...")
    
    # 1. Ensure Target Directories Exist
    for d in TARGET_DIRS:
        path = os.path.join(ROOT, d)
        if not os.path.exists(path):
            os.makedirs(path)
            
    # 2. Process Items
    for item in os.listdir(ROOT):
        # Skip Target Dirs themselves
        if item in TARGET_DIRS: continue
        
        # Handle Case-Insensitivity (e.g. 'memory' vs 'Memory')
        if item.title() in TARGET_DIRS or item.capitalize() in TARGET_DIRS:
            # It's likely a mis-cased target dir (e.g. 'memory')
            target = item.capitalize() # or however we defined TARGET_DIRS
            if target in TARGET_DIRS and item != target:
                # Rename logic
                src = os.path.join(ROOT, item)
                tmp = os.path.join(ROOT, f"_{item}")
                dst = os.path.join(ROOT, target)
                
                try:
                    os.rename(src, tmp)
                    os.rename(tmp, dst)
                    logger.info(f"   ✨ Capitalized: {item} -> {target}")
                except Exception as e:
                    logger.warning(f"   Could not capitalize {item}: {e}")
            continue

        src = os.path.join(ROOT, item)
        moved = False
        
        # Check Categories
        for cat, keywords in FILE_MAP.items():
            if item in keywords:
                dst = os.path.join(ROOT, cat, item)
                try:
                    if os.path.exists(dst):
                         logger.warning(f"   ⚠️ Conflict: {item} exists in {cat}. Skipping.")
                    else:
                         shutil.move(src, dst)
                         logger.info(f"   Moved {item} -> {cat}/")
                    moved = True
                    break
                except Exception as e:
                    logger.error(f"   Error moving {item}: {e}")
                    moved = True # Mark handled to avoid archive loop
                    break

        if not moved:
            # Check Folder Archive logic
            if os.path.isdir(src):
                if item.startswith("_") or any(char.isdigit() for char in item):
                    dst = os.path.join(ROOT, "Archive", item)
                    try:
                        shutil.move(src, dst)
                        logger.info(f"   📦 Auto-Archived: {item}")
                    except Exception as e:
                         logger.error(f"   Error archiving {item}: {e}")

if __name__ == "__main__":
    clean_data()
