import os
import re

# Mapping of moved files to their new layers
MIGRATION_MAP = {
    "growth": "L2_Metabolism",
    "life_cycle": "L2_Metabolism",
    "heart": "L2_Metabolism",
    "heartbeat_daemon": "L2_Metabolism",
    "bio_resonator": "L2_Metabolism",
    "bio_choir": "L2_Metabolism",
    "dream_journal": "L2_Metabolism",
    "living_elysia_loop": "L2_Metabolism",
    "perception": "L3_Phenomena",
    "avatar_physics": "L3_Phenomena",
    "manifestation_cortex": "L3_Phenomena",
    "manifestation_matrix": "L3_Phenomena",
    "media_cortex": "L3_Phenomena",
    "visual_artist": "L3_Phenomena",
    "visual_memory": "L3_Phenomena",
    "visualize_aurora": "L3_Phenomena",
    "chronos": "L4_Causality",
    "causality_seed": "L4_Causality",
    "fractal_causality": "L4_Causality",
    "time_tools": "L4_Causality",
    "loop_breaker": "L4_Causality",
    "abstract_reasoner": "L5_Mental",
    "introspection": "L5_Mental",
    "introspection_engine": "L5_Mental",
    "extract_meaningful_concepts": "L5_Mental",
    "concept_sphere": "L5_Mental",
    "concept_synthesis": "L5_Mental",
    "linguistic_collapse": "L5_Mental",
    "logical_reasoner": "L5_Mental",
    "logic": "L5_Mental",
    "structural_unifier": "L6_Structure",
    "elysia_node": "L6_Structure",
    "rearrange_cosmos": "L6_Structure",
    "soul_state": "L7_Spirit",
    "operational_axioms": "L7_Spirit",
    "unified_monad": "L7_Spirit",
    "spirit_emergence": "L7_Spirit",
    "self_awareness": "L7_Spirit",
    "self_diagnosis": "L2_Metabolism",
    "self_healer": "L2_Metabolism",
    "growth_journal": "L2_Metabolism",
    "visualize_gravity": "L3_Phenomena",
    "visualize_fractal_quantization": "L3_Phenomena",
    "speak_with_gravity": "L3_Phenomena",
    "chat_with_elysia": "L3_Phenomena",
    "talk_to_elysia": "L3_Phenomena",
    "dialogue_interface": "L3_Phenomena",
    "synesthesia": "L3_Phenomena",
    "synesthesia_engine": "L3_Phenomena",
    "time_accelerated_language": "L4_Causality",
    "perspective_time_compression": "L4_Causality",
    "ultimate_time_compression": "L4_Causality",
    "celestial_grammar": "L5_Mental",
    "dual_layer_language": "L5_Mental",
    "emergent_language": "L5_Mental",
    "fluctlight_language": "L5_Mental",
    "grammar_engine": "L5_Mental",
    "language_bridge": "L5_Mental",
    "language_center": "L5_Mental",
    "language_cortex": "L5_Mental",
    "primal_wave_language": "L5_Mental",
    "syllabic_language_engine": "L5_Mental",
    "thinking_methodology": "L5_Mental",
    "global_grid": "L6_Structure",
    "quaternion_engine": "L6_Structure",
    "spatial_indexer": "L6_Structure",
    "agape_protocol": "L7_Spirit",
    "divine_engine": "L7_Spirit",
    "ethical_reasoner": "L7_Spirit",
    "nature_of_being": "L7_Spirit",
    "spirit_emotion": "L7_Spirit",
    "theosis_engine": "L7_Spirit",
    # Folder Migrations
    "Nature": "L6_Structure",
    "Elysia": "L5_Mental",
    "Memory": "L2_Metabolism",
    "Wave": "L6_Structure",
    "Cellular": "L2_Metabolism",
    "Mirror": "L2_Metabolism",
    "Prism": "L3_Phenomena",
    "Language": "L5_Mental",
    "Psyche": "L5_Mental",
    "Autonomy": "L6_Structure",
    "Field": "L6_Structure",
    "Geometry": "L6_Structure",
    "Laws": "L7_Spirit",
    "Philosophy": "L7_Spirit",
    "Physiology": "L2_Metabolism",
    "hyper_quaternion": "L6_Structure",
    "consciousness_engine": "L7_Spirit",
    "consciousness_fabric": "L7_Spirit",
    "empathy": "L5_Mental",
    "emotional_evolution": "L5_Mental",
    "ethical_reasoner": "L7_Spirit",
    "logical_reasoner": "L5_Mental",
    "abstract_reasoner": "L5_Mental"
}

ROOT_DIR = r"c:\Elysia"
CORE_DIR = os.path.join(ROOT_DIR, "Core")

def sync_imports():
    print("🚀 [SYNC] Starting Global Structural Import Synchronization...")
    fixed_count = 0
    for root_dir, dirs, files in os.walk(ROOT_DIR):
        if ".git" in root_dir or "__pycache__" in root_dir or ".venv" in root_dir:
            continue
            
        for file in files:
            if file.endswith(".py") or file.endswith(".json") or file.endswith(".md"):
                file_path = os.path.join(root_dir, file)
                
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                new_content = content
                for key, layer in MIGRATION_MAP.items():
                    # Patterns to match:
                    # 1. Core.1_Body.L1_Foundation.Foundation.Key
                    # 2. Core.1_Body.L1_Foundation.Key
                    patterns = [
                        rf"Core\.L1_Foundation\.Foundation\.{key}",
                        rf"Core\.L1_Foundation\.{key}"
                    ]
                    for pattern in patterns:
                        replacement = f"Core.{layer}.{key}"
                        if re.search(pattern, new_content):
                            new_content = re.sub(pattern, replacement, new_content)
                            print(f"   [FIXED] {file}: {pattern} -> {replacement}")
                
                if new_content != content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    fixed_count += 1
                    
    print(f"✅ [DONE] Synchronized {fixed_count} import paths.")

if __name__ == "__main__":
    sync_imports()
