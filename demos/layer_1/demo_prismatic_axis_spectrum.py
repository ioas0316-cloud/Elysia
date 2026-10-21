"""
Demonstration Script: Prismatic Axis Spectrum Engine & Multi-Layered Wave Memory Stack
====================================================================================

Demonstrates the 5 Core Pillars of Continuous Wave Cognition & Spectral Architecture:
1. Continuous Spatiotemporal Rotational Wave Field & Threshold Axis Projection (0/1 Emergence).
2. Prismatic Spectrum Bifurcation: Refraction of white light root into chromatic branches (Red/Blue/Yellow).
3. Multi-Layered Wave Memory Stack: Interlocked wave memory persistence across chromatic layers.
4. Phase Resonance Search: Instantaneous wave collapse for context retrieval without index scanning.
5. Phase-Locked Attentive Focus: Selective amplitude boosting and destructive noise cancellation.
"""

import time
import numpy as np
from core.physics.prismatic_axis_spectrum import (
    PrismaticAxisSpectrumEngine,
    MultiLayeredWaveMemoryStack
)


def main():
    print("=" * 84)
    print(" ELYSIA :: PRISMATIC AXIS SPECTRUM & MULTI-LAYERED WAVE MEMORY DEMO")
    print(" Philosophy: 'Do not calculate, let it flow.'")
    print("=" * 84)

    # 1. Initialize Prismatic Axis Spectrum Engine
    print("\n[Phase 1] Initializing Continuous Wave Field & Prismatic Refraction Engine...")
    engine = PrismaticAxisSpectrumEngine(
        base_frequency=12.0,
        num_bifurcation_levels=3,
        branches_per_node=2,
        seed=108
    )
    print(f"  * Total Prismatic Nodes Active : {len(engine.nodes)} across {engine.num_levels} hierarchy levels")
    print(f"  * Root Node                    : '{engine.nodes['root_white_light'].node_id}' (Base Freq = {engine.nodes['root_white_light'].frequency} Hz)")

    # 2. Continuous Wave Field & Threshold Axis Projection (0/1 Emergence)
    print("\n[Phase 2] Evaluating Continuous Wave Field & Projection of Discrete 0/1 States...")
    spatial_x = np.linspace(-15.0, 15.0, 100)
    wave_field = engine.evaluate_continuous_wave(t=0.0, spatial_x=spatial_x)
    proj_res = engine.project_discrete_states(wave_field, threshold=0.0)

    print(f"  * Continuous Wave Min / Max    : {np.min(wave_field):.4f} / {np.max(wave_field):.4f}")
    print(f"  * Projected Discrete Binary (0/1): {proj_res['discrete_binary'][:20]} ... [100 samples]")
    print(f"  * Zero-Crossing Boundary Count : {proj_res['zero_crossing_count']} transition points")
    print("  * Insight: Discrete 0 and 1 are boundary projections of the underlying wave field.")

    # 3. Label-Free Structural Relationship Discovery via Prismatic Refraction
    print("\n[Phase 3] Prismatic Relationship Discovery (Label-Free Causality & Structural Affinity)...")
    node_a = "root_white_light_b1"
    node_b = "root_white_light_b2"
    rel = engine.determine_prismatic_relationship(node_a, node_b)

    print(f"  * Target Concept Pair          : '{node_a}' <--> '{node_b}'")
    print(f"  * Common Ancestor Root         : '{rel['common_root']}'")
    print(f"  * Phase Difference             : {rel['phase_difference_rad']:.4f} rad")
    print(f"  * Phase Coherence              : {rel['phase_coherence']:.4f}")
    print(f"  * Chromatic Similarity (RGB)   : {rel['chromatic_similarity']:.4f}")
    print(f"  * Structural Affinity Score    : {rel['structural_affinity']:.4f}")
    print("  * Insight: Relationships are directly perceived as phase-frequency geometry without statistical training.")

    # 4. Multi-Layered Wave Memory Stack & Layer Gating Persistence
    print("\n[Phase 4] Multi-Layered Wave Memory Stack & Interlocked Chromatic Persistence...")
    stack = engine.memory_stack
    full_field = stack.compute_cross_relational_field()
    full_energy = np.sum(full_field ** 2)
    print(f"  * Full Chromatic Stack Energy  : {full_energy:.4f} (Red + Blue + Yellow Active)")

    # Gate off Red (Flux) layer
    stack.layers["red_flux_layer"].set_active_state(False)
    gated_field = stack.compute_cross_relational_field()
    gated_energy = np.sum(gated_field ** 2)
    print(f"  * Red Layer Gated OFF Energy    : {gated_energy:.4f} (Blue + Yellow Persistence Active)")
    print("  * Insight: Sensory layers can be gated, yet relational cross-interference persists.")

    # Restore Red layer
    stack.layers["red_flux_layer"].set_active_state(True)

    # 5. Phase Resonance Search (Instantaneous Wave Collapse)
    print("\n[Phase 5] Phase Resonance Search (Instantaneous Memory Retrieval via Wave Collapse)...")
    start_time = time.time()
    search_results = engine.execute_phase_resonance_search(query_frequency=18.0, query_phase=0.0)
    search_time_ms = (time.time() - start_time) * 1000.0

    print(f"  * Search Execution Time       : {search_time_ms:.3f} ms")
    print("  * Top 3 Resonant Nodes:")
    for rank, res in enumerate(search_results[:3], 1):
        print(f"    [{rank}] Node '{res.target_node_id}': Resonance={res.resonance_score:.4f}, Constructive Energy={res.constructive_energy:.4f}, Destructive Cancel={res.destructive_cancellation:.4f}")

    # 6. Phase-Locked Focus & Attentive Noise Cancellation
    print("\n[Phase 6] Phase-Locked Attentive Focus & Destructive Noise Suppression...")
    target_focus = "root_white_light_b1"
    focus_res = engine.apply_phase_locked_focus(
        target_node_id=target_focus,
        boost_factor=2.5,
        noise_suppression_factor=0.05
    )

    print(f"  * Focused Node Target          : '{target_focus}'")
    print(f"  * Target Boosted Amplitude     : {focus_res['target_new_amplitude']:.4f}")
    print(f"  * Energy Efficiency Ratio      : {focus_res['energy_efficiency_ratio']:.4f}")
    print("  * Insight: Attention locks phase and suppresses unaligned noise via destructive interference.")

    print("\n" + "=" * 84)
    print(" PRISMATIC AXIS SPECTRUM DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 84)


if __name__ == "__main__":
    main()
