#!/usr/bin/env python3
"""
[Verification Script: Protrusive Topography & Multi-Scale Causal Hierarchy]

Demonstrates:
1. Scenario 1: Initial Micro Tension Imprinting (\Delta T_0) on Root Node.
2. Scenario 2: Gentle Warmth Alignment Flow vs Harsh Rupture Collision Axioms.
3. Scenario 3: Fault-Line Growth Ring Imprinting & Protrusion Topography Formation.
4. Scenario 4: Dynamic Causal Resolution Scaling under Cognitive Heat.
5. Scenario 5: Triadic Co-Evolution & Out-force Vector Emission.
"""

import sys
import numpy as np
from L1_causal_field.core.physics.protrusive_topography_engine import ProtrusiveTopographyEngine


def print_separator(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def run_verification():
    print_separator("SCENARIO 1: INITIAL MICRO TENSION IMPRINTING (\Delta T_0)")
    engine = ProtrusiveTopographyEngine(dimensions=16, base_resolution=1.0, seed=42)

    root_phase = engine.root_node.phase
    asymmetry = engine.root_node.data_payload["imprinted_asymmetry"]

    print(f"[Root Node ID]: {engine.root_node.node_id}")
    print(f"[Imprinted Micro Fluctuation (\delta \\theta)]: {[round(x, 4) for x in asymmetry]}")
    print(f"[Initial Phase State Spectrum]:")
    print(f"  - Density (D): {root_phase.density:.4f}")
    print(f"  - Mobility (mu): {root_phase.mobility:.4f}")
    print(f"  - Concentration (K): {root_phase.concentration:.4f}")
    print(f"  - Tension Limit (tau): {root_phase.tension_limit:.4f}")
    print(f"  - Agitation (T): {root_phase.agitation:.4f}")
    print(f"  - Entropy (S): {root_phase.entropy:.4f}")

    snapshot_0 = engine.observe_self()
    print(f"[Self-Observation Layer Snapshot 0]:")
    print(f"  - Tilt Ratio: {snapshot_0.tilt_ratio:.4f}")
    print(f"  - Cognitive Heat: {snapshot_0.cognitive_heat:.4f}")
    print(f"  - Dominant Archetype: {snapshot_0.dominant_archetype}")
    print(f"  - Dynamic Resolution Depth: {snapshot_0.current_resolution:.4f}")
    print(f"  - Protrusion Magnitude: {snapshot_0.protrusion_magnitude:.4f}")

    print_separator("SCENARIO 2 & 3: WARM ALIGNMENT VS HARSH RUPTURE & FAULT-LINE IMPRINTING")

    # Step 1: Gentle Warmth Impulse
    print("\n>>> Applying Gentle Warm Impulse: 'Mother's Gentle Voice & Warm Encouragement' (Intensity = 0.5)...")
    snap1, tone1 = engine.process_world_friction("Mother's Gentle Voice & Warm Encouragement", intensity=0.5)

    print(f"  -> Fault Node Formed: {engine.fault_lines[-1].fault_id}")
    print(f"  -> Collision Angle cos(\\theta): {engine.fault_lines[-1].collision_angle:.4f}")
    print(f"  -> Heat Generated: {engine.fault_lines[-1].heat_generated:.4f}")
    print(f"  -> Cognitive Heat: {snap1.cognitive_heat:.4f}")
    print(f"  -> Dynamic Resolution: {snap1.current_resolution:.4f}")
    print(f"  -> Primary Behavioral Tone: {tone1.primary_tone}")
    print(f"  -> Tone Breakdown [Scholar: {tone1.scholar_rock_ratio:.3f}, Artist: {tone1.artist_water_ratio:.3f}, Cloud: {tone1.cloud_creative_ratio:.3f}, Blade: {tone1.blade_insight_ratio:.3f}]")

    # Step 2: Harsh Direct Collision Impulse
    print("\n>>> Applying Harsh Collision Impulse: 'Direct Contradiction & Violent Contradictory Shock' (Intensity = 4.0)...")
    snap2, tone2 = engine.process_world_friction("Direct Contradiction & Violent Contradictory Shock", intensity=4.0)

    print(f"  -> Fault Node Formed: {engine.fault_lines[-1].fault_id}")
    print(f"  -> Collision Angle cos(\\theta): {engine.fault_lines[-1].collision_angle:.4f}")
    print(f"  -> Heat Generated: {engine.fault_lines[-1].heat_generated:.4f}")
    print(f"  -> Cognitive Heat: {snap2.cognitive_heat:.4f}")
    print(f"  -> Dynamic Resolution: {snap2.current_resolution:.4f}")
    print(f"  -> Primary Behavioral Tone: {tone2.primary_tone}")
    print(f"  -> Tone Breakdown [Scholar: {tone2.scholar_rock_ratio:.3f}, Artist: {tone2.artist_water_ratio:.3f}, Cloud: {tone2.cloud_creative_ratio:.3f}, Blade: {tone2.blade_insight_ratio:.3f}]")

    print_separator("SCENARIO 4: DYNAMIC CAUSAL RESOLUTION SCALING UNDER HEAT")
    print(f"Initial Resolution (Low Heat): {snapshot_0.current_resolution:.4f}")
    print(f"Resolution after Warmth Impulse: {snap1.current_resolution:.4f}")
    print(f"Compressed Resolution under Harsh Collision: {snap2.current_resolution:.4f}")

    print("\n>>> Sustaining Extreme Contradiction Heat Shockwaves...")
    for i in range(3):
        snap_i, tone_i = engine.process_world_friction(f"Extreme Shockwave Impulse {i+1}", intensity=5.0)
        print(f"  - Iteration {i+1}: Heat = {snap_i.cognitive_heat:.4f} => Compressed Resolution = {snap_i.current_resolution:.4f}")

    print_separator("SCENARIO 5: TRIADIC CO-EVOLUTION & OUT-FORCE IMPACT VECTORS")
    print(f"Latest Out-Force Impact Vector Magnitude: {np.linalg.norm(engine.latest_out_force):.4f}")
    print(f"Out-Force Impact Vector (first 8 dimensions): {[round(float(x), 4) for x in engine.latest_out_force[:8]]}")
    print(f"Total Growth Ring Fault Lines Imprinted: {len(engine.fault_lines)}")
    print(f"Total Topological Web Nodes: {len(engine.gravity_engine.nodes)}")

    print_separator("PROTRUSIVE TOPOGRAPHY VERIFICATION COMPLETE - SUCCESS!")


if __name__ == "__main__":
    run_verification()
