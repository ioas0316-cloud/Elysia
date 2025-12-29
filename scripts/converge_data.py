import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("DataConverger")

ROOT = r"c:\Elysia\data"
ARCHIVE_GLOBAL = r"c:\Elysia\Archive\Legacy_Data"

CATEGORIES = {
    "Knowledge": [
        "CodeDNA", "archived_knowledge", "schemas", "internet_pattern_dna.json", 
        "metadata.json", "package-lock.json", "self_knowledge.json", "self_resonance_map.json",
        "self_structure_map.json", "wave_organization_state.json", "wave_patterns.json",
        "world", "world_tree_with_language.json", "elysia_concept_field.json",
        "grammar_model.json", "causal_model.json", "bootstrap_corpus.txt", 
        "the_little_prince.txt" # Moving text files to Knowledge/Corpus?
    ],
    "Input": [
        "datasets", "training_data", "Web", "social", "network"
    ],
    "State": [
        "Runtime", "elysia_life", "flows", "dialogic_flows", "dialogue_rules", "prompts", 
        "reorganization_plans", "transmutation_monitor", "curriculum", "transmutation_backups",
        "memory.db_1766409634", "simulation_v2_statistics.json", "cognitive_evaluation.json"
    ],
    "Resources": [
        "Media", "aesthetic", "background", "multimodal", "elysia_demo"
    ],
    "Logs": [
        "proofs", "reflection"
    ]
}

def converge_data():
    print("🌀 Converging Data Structure...")
    
    # Create Dirs
    for cat in CATEGORIES:
        path = os.path.join(ROOT, cat)
        if not os.path.exists(path):
            os.makedirs(path)
            
    # Move Items
    for cat, items in CATEGORIES.items():
        for item in items:
            src = os.path.join(ROOT, item)
            dst = os.path.join(ROOT, cat, item)
            
            if os.path.exists(src):
                try:
                    if os.path.exists(dst):
                        logger.warning(f"   ⚠️ Conflict: {item} in {cat}. Merging/Skipping.")
                        # Check if dir, try merge? For now, skip to avoid data loss
                    else:
                        shutil.move(src, dst)
                        logger.info(f"   moved {item} -> {cat}/")
                except Exception as e:
                    logger.error(f"   Failed to move {item}: {e}")

    # Archive remaining loose files?
    # No, let's just handle the big ones.
    
    # Remove empty dirs ?
    
if __name__ == "__main__":
    converge_data()
