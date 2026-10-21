"""
Demo: Dressmaker Cross-World Phase-Locking & Mainboard Knowledge Bus Slot
==========================================================================
Demonstrates:
1. Slotting NotebookLM knowledge chunks and physical principles into ELYSIA Mainboard Layer 3 Knowledge Bus.
2. Creating continuous Fabric Wave Spectra (Celestial Silk & Dragon Scale Leather) from Dressmaker simulator.
3. Organic phase-locking ($\Phi_{\text{fabric}} \leftrightarrow \Phi_{\text{body}}$) with characters across different game worlds:
   - Action Game Hero (Valhalla World)
   - Cyberpunk Explorer (Neo-Tokyo World)
   - VR Simulator Avatar (Cosmic Grounding World)
4. Dynamic movement resonance simulation on the ELYSIA Scale-Free Manifold.
"""

import numpy as np
from core.knowledge_base.knowledge_bus_slot import KnowledgeBusSlot, KnowledgeChunk
from elysia_core.dressmaker_phase_lock_runtime import (
    DressmakerPhaseLockRuntime,
    FabricWaveSpectrum,
    CharacterBodyRotor,
)


def run_demo():
    print("\n" + "=" * 80)
    print(" ELYSIA MAINBOARD DEMO: DRESSMAKER CROSS-WORLD PHASE-LOCKING & KNOWLEDGE BUS ")
    print("=" * 80 + "\n")

    # Step 1: Initialize Layer 3 Knowledge Bus Slot & Slot-In NotebookLM Chunks
    print(">>> [STEP 1] Layer 3 Mainboard Knowledge Bus Slot Initialization... <<<\n")
    knowledge_bus = KnowledgeBusSlot(slot_id="SLOT_L3_MAINBOARD_NOTEBOOKLM", vector_dim=5)

    chunk_1 = KnowledgeChunk(
        chunk_id="CHUNK_NBOOK_01",
        title="NotebookLM: Weaver's Constraint & Tension Principle",
        source_type="notebook_lm",
        content_text="Fabric weave stress distributes along geodesic pathways of the underlying body manifold.",
        semantic_vector=np.array([0.4, 0.8, -0.2, 0.5, 0.1]),
        constraint_order=1.5
    )

    chunk_2 = KnowledgeChunk(
        chunk_id="CHUNK_PHYSICS_02",
        title="Physical Invariant: Scale-Free Phase Locking",
        source_type="physical_principle",
        content_text="Rotor phase locking maintains topological continuity without explicit mesh re-rigging.",
        semantic_vector=np.array([0.1, 0.3, 0.9, -0.4, 0.7]),
        constraint_order=2.0
    )

    knowledge_bus.slot_in(chunk_1)
    knowledge_bus.slot_in(chunk_2)

    status = knowledge_bus.get_bus_status()
    print(f"  * Knowledge Bus Active Chunks: {status['chunk_count']}")
    for c in status['chunks']:
        print(f"    - [{c['type']}] {c['title']} (Order: {c['order']})")
    print(f"  * Aggregate Knowledge Vector: {np.round(status['aggregate_field'], 3)}\n")

    # Step 2: Initialize Dressmaker Phase-Locking Runtime
    print(">>> [STEP 2] Initializing Dressmaker Phase-Locking Runtime... <<<\n")
    runtime = DressmakerPhaseLockRuntime(knowledge_bus=knowledge_bus)

    # Fabrics from Dressmaker
    silk = FabricWaveSpectrum.create_silk("FABRIC_SILK_CELESTIAL")
    leather = FabricWaveSpectrum.create_armored_leather("FABRIC_LEATHER_DRAGON")

    print(f"  * Fabric Created: '{silk.name}' (Weave Density: {silk.weave_density} Hz)")
    print(f"  * Fabric Created: '{leather.name}' (Weave Density: {leather.weave_density} Hz)\n")

    # Step 3: Create Character Body Rotors from Different Game Worlds
    print(">>> [STEP 3] Connecting Character Body Rotors across Different Game Worlds... <<<\n")

    t_samples = np.linspace(0, 4 * np.pi, 64)

    hero_action = CharacterBodyRotor(
        character_id="HERO_VALHALLA",
        world_origin="ActionGame_Valhalla",
        body_mass=85.0,
        rotor_angles=np.array([0.2, 0.5, -0.1]),
        body_surface_curvature=np.sin(t_samples) * 0.7 + 0.2
    )

    explorer_cyber = CharacterBodyRotor(
        character_id="EXPLORER_NEO_TOKYO",
        world_origin="Explorer_NeoTokyo",
        body_mass=62.0,
        rotor_angles=np.array([-0.3, 0.1, 0.8]),
        body_surface_curvature=np.cos(1.5 * t_samples) * 0.5 - 0.1
    )

    # Step 4: Execute Cross-World Phase-Locking
    print(">>> [STEP 4] Executing Organic Phase-Locking (No Export/Import, No Rigging)... <<<\n")

    coupling_1 = runtime.phase_lock_fabric_to_body(silk, hero_action)
    print(f"  * [{coupling_1['world_origin']}] '{silk.name}' <==> '{hero_action.character_id}':")
    print(f"    - Phase Coherence Score: {coupling_1['phase_coherence_score']:.4f}")
    print(f"    - Status: {coupling_1['status']}")
    print(f"    - Adapted Tension Trace: {coupling_1['adapted_tension_trace']:.4f}\n")

    coupling_2 = runtime.phase_lock_fabric_to_body(leather, explorer_cyber)
    print(f"  * [{coupling_2['world_origin']}] '{leather.name}' <==> '{explorer_cyber.character_id}':")
    print(f"    - Phase Coherence Score: {coupling_2['phase_coherence_score']:.4f}")
    print(f"    - Status: {coupling_2['status']}")
    print(f"    - Adapted Tension Trace: {coupling_2['adapted_tension_trace']:.4f}\n")

    # Step 5: Simulate Movement Resonance Trajectory
    print(">>> [STEP 5] Movement Impulse & Wrinkle Dynamic Resonance Simulation... <<<\n")
    trajectory = runtime.simulate_movement_resonance(
        coupling_key=coupling_1["coupling_key"],
        motion_impulse=2.5,
        time_steps=5
    )
    print(f"  * Movement Energy Trajectory (Silk on Action Hero): {[round(x, 3) for x in trajectory]}")

    print("\n" + "=" * 80)
    print(" DRESSMAKER MAINBOARD DEMONSTRATION COMPLETE SUCCESS ")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_demo()
