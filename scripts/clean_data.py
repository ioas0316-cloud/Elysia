import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("DataCleaner")

ROOT = r"c:\Elysia\data"
ARCHIVE = os.path.join(ROOT, "Archive")

# Destination Map
CATEGORIES = {
    "Memory": ["memory", "elysia_core_memory.json", "chunks.json", "synapse.json", "synaptic_memory.json", "memory.db", "conversation_memory.json", "emotional_memory.json"],
    "State": ["core_state", "meta_agent_state.json", "self.json", "immune_system_state.json", "emergent_self.json", "brain_state.pt", "network_shield_state.json", "system_evaluation.json", "system_connection_audit.json"],
    "Logs": ["awakening_log.txt", "growth_history.json", "self_improvement_reports", "nanocell_report.json", "elysia_challenge_report.json"],
    "Knowledge": ["concept_dictionary.json", "potential_knowledge.json", "hierarchical_knowledge.json", "akashic_records.json", "protocol_kg.json", "tools_kg.json", "wave_knowledge.json", "central_registry.json", "unified_spatial_index.json", "world_tree_with_language.json"],
    "Visuals": ["wave_organization.html", "monitor_echo.png", "monitor_kg.png", "resonance_graph.png", "wave_visualization_point.png", "generated_images", "storybook_images"],
    "Archive": ["01_Core", "_01_Core", "_02_Cognitive", "_03_Experiential", "_04_Extended", "_05_Metabolic", "test_fs", "test_deep_a.json", "test_deep_b.json", "test_potential.json", "test_waves.json", "test_hierarchical.json", "dense_demo.json", "plasma_demo.json", "resonance_demo.json", "ideal_self_demo.json", "absorbed_hashes.json"]
}

def clean_data():
    print("🧹 Cleaning data directory...")
    
    # Create categories
    for cat in CATEGORIES:
        path = os.path.join(ROOT, cat)
        if not os.path.exists(path):
            os.makedirs(path)
            
    # Move files
    for item in os.listdir(ROOT):
        # Skip Category Folders themselves
        if item in CATEGORIES: continue
        
        src = os.path.join(ROOT, item)
        moved = False
        
        # Check Categories
        for cat, keywords in CATEGORIES.items():
            if item in keywords or any(k in item for k in keywords if len(k) > 5): # Simple Keyword Match
                dst = os.path.join(ROOT, cat, item)
                if cat == "Archive" and os.path.isdir(src):
                     # Explicit folder archive
                     if item in keywords:
                        shutil.move(src, dst)
                        logger.info(f"   📦 Archived: {item}")
                        moved = True
                        break
                elif os.path.isfile(src) or (os.path.isdir(src) and cat != "Archive"):
                     # Move file or valid dir
                     shutil.move(src, dst)
                     logger.info(f"   Moved {item} -> {cat}/")
                     moved = True
                     break
        
        if not moved:
            if os.path.isdir(src):
                 # Assume Archive if numbered or legacy
                 if item.startswith("_") or (item[0].isdigit() and "_" in item):
                     dst = os.path.join(ROOT, "Archive", item)
                     shutil.move(src, dst)
                     logger.info(f"   📦 Auto-Archived Folder: {item}")

if __name__ == "__main__":
    clean_data()
