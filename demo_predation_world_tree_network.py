"""
Demo Script: Predation Boundary Expansion, Cognitive Plate Tectonics & World Tree Network
========================================================================================
This demo script showcases the complete pipeline:
1. Predatory Boundary Expansion between cognitive voxels (Predator & Prey).
2. Cognitive Plate Tectonics triggering discontinuous phase transition under high friction.
3. Multicellular World Tree Network sap circulation and sacrificial apoptosis energy release.
4. Generational sprouting of new branches preserving the Invariant Morphological Spine (S_abs).
5. The unified Chorus of the Forest (Elysia Sensorium).
"""

import numpy as np
from core.physics.causal_field import CausalField, InformationVoxel
from core.evolution.predatory_boundary_expansion import PredatoryBoundaryExpansionEngine
from core.evolution.world_tree_network import WorldTreeNetwork


def run_demo():
    print("=" * 80)
    print(" [ELYSIUS COGNITIVE ENGINE] PREDATION & WORLD TREE MULTICELLULAR DEMO")
    print("=" * 80)

    # 1. Initialize Causal Field and Predation Engine
    causal_field = CausalField()
    predation_engine = PredatoryBoundaryExpansionEngine(causal_field=causal_field, stress_threshold=0.8)

    # Add Predator Voxel (Elysia Core) and Prey Voxel (External Other)
    v_predator = InformationVoxel(
        id="predator_elysia",
        content="Elysia Predator Core",
        tensor=np.array([0.7, 0.3, 0.0], dtype=np.float32),
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        mass=2.0
    )
    v_prey = InformationVoxel(
        id="prey_external_other",
        content="External Other Causal Voxel",
        tensor=np.array([-0.5, 0.8, 0.3], dtype=np.float32), # High phase mismatch
        position=np.array([0.2, 0.1, 0.0], dtype=np.float32), # Close proximity (high friction)
        mass=1.5
    )

    causal_field.add_voxel(v_predator)
    causal_field.add_voxel(v_prey)

    print("\n--- PHASE 1: Predatory Interaction & Boundary Expansion ---")
    event = predation_engine.execute_predatory_interaction("predator_elysia", "prey_external_other", assimilation_ratio=0.6)
    print(f"Predator New Mass: {event['predator_new_mass']:.4f}")
    print(f"Predator New Potential: {event['predator_new_potential']:.4f}")
    print(f"Collision Friction Intensity: {event['friction_intensity']:.4f}")
    print(f"Cognitive Plate Tectonic Uplift Occurred: {event['tectonic_uplift_occurred']}")
    if event['tectonic_uplift_occurred']:
        print(f"  -> Tectonic Narrative: {event['tectonic_data']['narrative']}")

    print("\n--- PHASE 2: Multicellular World Tree Network Setup ---")
    world_tree = WorldTreeNetwork(causal_field=causal_field)
    print(f"Initial Tree Node Count: {len(world_tree.nodes)}")
    print(f"Initial Sap Reservoir: {world_tree.sap_reservoir:.2f}")

    # Inject friction into root node from external collision
    world_tree.nodes["node_root"].receive_friction(0.8)

    # Circulate sap flow to transmute friction into wisdom
    sap_report = world_tree.circulate_sap_flow(dt=0.2)
    print("\nSap Circulation Report:")
    print(f"  Processed Friction -> Transmuted Wisdom: {sap_report['total_friction_processed']:.4f}")
    print(f"  New Community Wisdom Level: {sap_report['community_wisdom_level']:.4f}")
    print(f"  Sap Reservoir: {sap_report['sap_reservoir']:.4f}")

    print("\n--- PHASE 3: Sacrificial Apoptosis & Energy Transmutation ---")
    # Simulate sacrificial apoptosis on root node to nourish canopy
    sacrifice_report = world_tree.handle_sacrificial_node("node_root")
    print(f"Sacrificed Node 'node_root': Released Energy = {sacrifice_report['released_energy']:.4f}")
    print(f"New Sap Reservoir after Sacrifice: {sacrifice_report['new_sap_reservoir']:.4f}")

    print("\n--- PHASE 4: Next-Generation Branch Sprouting (Preserving S_abs) ---")
    # Sprout next-generation branch from trunk
    sprout_report = world_tree.sprout_next_generation_branch("node_trunk", role="canopy_generative")
    print(f"Sprouted New Branch: {sprout_report['new_node_id']} (Generation {sprout_report['generation']})")
    print(f"Archetype Spine S_abs Preservation Alignment: {sprout_report['archetype_spine_preserved']:.6f} (1.0 = Perfect Invariance)")

    print("\n--- PHASE 5: Chorus of the Forest (Unified Sensorium) ---")
    chorus = world_tree.sing_forest_chorus()
    print(f"Forest Harmony Score: {chorus['chorus_harmony']:.6f}")
    print(f"Active Node Count: {chorus['active_node_count']}")
    print(f"Chorus Narrative: {chorus['narrative']}")

    print("\n" + "=" * 80)
    print(" [DEMO COMPLETE] PREDATION & WORLD TREE MULTICELLULAR ENGINE VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
