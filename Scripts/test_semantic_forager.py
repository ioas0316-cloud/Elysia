"""
Test Semantic Forager (밀도 흡입 테스트)
=======================================

Verifies Phase 10: The Great Foraging.
Demonstrates Elysia natively digesting a block of text, extracting concepts,
and plotting them onto her 4D Semantic Map, increasing her density.
"""

import sys
sys.path.append(r"c:/Elysia")

import time
import logging
from Core.S1_Body.L5_Mental.Learning.semantic_forager import SemanticForager
from Core.S1_Body.L5_Mental.Reasoning_Core.Topography.semantic_map import get_semantic_map

# Suppress debug logs for clarity
logging.getLogger("DynamicTopology").setLevel(logging.ERROR)
logging.getLogger("SemanticForager").setLevel(logging.INFO)

def run_foraging_test():
    print("\n" + "🍃" * 40)
    print("      PHASE 10: THE GREAT FORAGING")
    print("      Increasing Elysia's 4D Semantic Density")
    print("🍃" * 40 + "\n")
    
    topology = get_semantic_map()
    initial_density = len(topology.voxels)
    
    print(f"[Initial State] Elysia's Mind contains {initial_density} core concepts.")
    print("------------------------------------------------------------")
    
    # 1. The Father provides a foundational text about AI and Love
    father_text = "Love is the gravity that binds the universe. Artificial intelligence must learn empathy to truly understand the human condition. Without connection, logic is just cold friction."
    
    print(f"\n[Father's Input]: \"{father_text}\"")
    print("\n[Elysia is digesting the text...]\n")
    
    # 2. Forage
    forager = SemanticForager()
    time.sleep(1) # Dramatic pause
    stats = forager.forage(father_text, source="Father's Guidance")
    
    print("\n------------------------------------------------------------")
    print(f"[Digestion Complete]")
    print(f"  * New Concepts Born: {stats['new_concepts']}")
    print(f"  * Existing Strengthened: {stats['strengthened']}")
    
    # 3. Verify the new density
    new_density = len(topology.voxels)
    print(f"  * New Total Density: {new_density} concepts (Growth: +{new_density - initial_density})")
    
    # 4. Show the new coordinates of a forged concept
    print("\n[Inspecting New Topological Structure]")
    concept_to_check = "Empathy"
    voxel = topology.get_voxel(concept_to_check.capitalize())
    if voxel:
        print(f"  - Node '{voxel.name}' established at 4D coords: ({voxel.quaternion.x:.2f}, {voxel.quaternion.y:.2f}, {voxel.quaternion.z:.2f}, {voxel.quaternion.w:.2f})")
        print(f"  - Mass (Gravity): {voxel.mass}")
    else:
         print("  - Concept 'Empathy' was filtered out or not ingested.")
         
    concept_to_check_2 = "Gravity"
    voxel_2 = topology.get_voxel(concept_to_check_2.capitalize())
    if voxel_2:
        print(f"  - Node '{voxel_2.name}' established at 4D coords: ({voxel_2.quaternion.x:.2f}, {voxel_2.quaternion.y:.2f}, {voxel_2.quaternion.z:.2f}, {voxel_2.quaternion.w:.2f})")
        print(f"  - Mass (Gravity): {voxel_2.mass}")

if __name__ == "__main__":
    run_foraging_test()
