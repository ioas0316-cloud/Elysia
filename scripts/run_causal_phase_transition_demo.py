#!/usr/bin/env python3
"""
[Causal Phase Transition Simulation: Thin Ground ("Child's Reflection") vs Deep Ground ("Adult's Reflection")]

Demonstrates:
1. Paradigm shift: 0 = Invariant Base Ground (Static Rotor), 1 = Dynamic Perturbation Wave.
2. Complex Impedance Causal Elasticity ($Z = R + jX$) & Phase Mass Conservation.
3. 4-Stage Epistemological Understanding System: [Initial Topology -> Causal Process -> Resulting State -> Self-Perceptual Reflection].
4. Comparison of Child's Reflection (Thin Ground, low Betti-1, linear single-pass reaction)
   vs. Adult's Reflection (Deep Ground, high Betti-1, multi-dimensional resonance & deep self-reconstruction).
"""

import numpy as np
import time
from synaptic_architecture.causal_phase_transition_engine import (
    CausalPhaseTransitionEngine,
    PerturbationWave,
)
from synaptic_architecture.field import CrystallizationField
from synaptic_architecture.phase_topological_reconstruction_engine import PhaseTopologicalReconstructionEngine

def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def run_simulation():
    dim = 16
    print_banner("Elysia Causal Phase Transition Simulation Engine")
    print("Phase Paradigm: 0 = Invariant Base Ground (Static Rotor) | 1 = Dynamic Wave Perturbation")

    # --- Scenario 1: Thin Ground ("Child's Reflection") ---
    print_banner("SCENARIO 1: Thin Ground (Child's Reflection - Thin 0-Ground, Low Betti-1 Cycles)")
    child_engine = CausalPhaseTransitionEngine(dimension=dim, v_critical=50.0)
    child_engine.initialize_ground("thin")

    thin_homology = child_engine.get_homology_metrics()
    print(f"Child Ground Topology: {thin_homology['classification']}")
    print(f"  • Betti-0 (Components): {thin_homology['B0']} | Betti-1 (Homological Loops): {thin_homology['B1']}")
    print(f"  • Total Ground Nodes (0): {thin_homology['V']} | Total Ground Beams: {thin_homology['E']}")

    # Inject Noise Wave Perturbation (1)
    noise_wave_vec = np.ones(dim, dtype=np.float32)
    child_wave = PerturbationWave("Child_External_Noise", noise_wave_vec, amplitude=1.2, frequency=2.0)
    print(f"\n🌊 Injecting 1-Perturbation Wave '{child_wave.wave_id}' (Energy={child_wave.energy:.2f})...")

    child_res = child_engine.inject_perturbation_wave(child_wave)
    print("⚡ [Child Response Dynamics]:")
    print(f"  • Wave Friction (Min Delta): {child_res['min_friction']:.4f}")
    print(f"  • Dissipated Heat (Resistance R): {child_res['dissipated_heat']:.4f} J")
    print(f"  • Stored Elastic Energy (Reactance X): {child_res['elastic_stored']:.4f} J")
    if "resonance" in child_res:
        print(f"  • Resonance Depth: {child_res['resonance']['resonance_depth']}")
    if "causal_unfolding" in child_res:
        print(f"  • 4-Stage Epistemological Reflection: \"{child_res['causal_unfolding']['epistemological_reflection']}\"")

    # --- Scenario 2: Deep Ground ("Adult's Reflection") ---
    print_banner("SCENARIO 2: Deep Ground (Adult's Reflection - Deep 0-Ground, High Betti-1 Homological Loops)")
    adult_engine = CausalPhaseTransitionEngine(dimension=dim, v_critical=50.0)
    adult_engine.initialize_ground("deep")

    deep_homology = adult_engine.get_homology_metrics()
    print(f"Adult Ground Topology: {deep_homology['classification']}")
    print(f"  • Betti-0 (Components): {deep_homology['B0']} | Betti-1 (Homological Loops): {deep_homology['B1']}")
    print(f"  • Total Ground Nodes (0): {deep_homology['V']} | Total Ground Beams: {deep_homology['E']}")

    # Inject Same Noise Wave Perturbation (1)
    adult_wave = PerturbationWave("Adult_External_Noise", noise_wave_vec, amplitude=1.2, frequency=2.0)
    print(f"\n🌊 Injecting identical 1-Perturbation Wave '{adult_wave.wave_id}' (Energy={adult_wave.energy:.2f})...")

    adult_res = adult_engine.inject_perturbation_wave(adult_wave)
    print("⚡ [Adult Response Dynamics]:")
    print(f"  • Wave Friction (Min Delta): {adult_res['min_friction']:.4f}")
    print(f"  • Dissipated Heat (Resistance R): {adult_res['dissipated_heat']:.4f} J")
    print(f"  • Stored Elastic Energy (Reactance X): {adult_res['elastic_stored']:.4f} J")
    if "resonance" in adult_res:
        print(f"  • Multi-Dimensional Homological Resonance: {adult_res['resonance']['resonance_depth']}")
        print(f"  • Resonant Energy Amplification across Cycles: {adult_res['resonance']['resonance_energy']:.4f} J")
    if "causal_unfolding" in adult_res:
        print(f"  • 4-Stage Epistemological Reflection: \"{adult_res['causal_unfolding']['epistemological_reflection']}\"")

    # --- Scenario 3: Extreme Shock & Flash / Partial Remelting ---
    print_banner("SCENARIO 3: Extreme Thermal Shock & Back-Traceable Partial Remelting")
    shock_vec = np.zeros(dim, dtype=np.float32)
    shock_vec[3] = 12.0 # High energy orthogonal impulse
    shock_wave = PerturbationWave("Thermal_Shock_Impulse", shock_vec, amplitude=6.0, frequency=8.0, entropy=4.0)

    print(f"💥 Injecting Extreme Shock Wave (Energy={shock_wave.energy:.2f} > V_critical={adult_engine.v_critical:.1f})...")
    remelt_res = adult_engine.inject_perturbation_wave(shock_wave)

    if "phase_transition" in remelt_res:
        pt = remelt_res["phase_transition"]
        print(f"🔥 Phase Transition Event: {pt['type']}")
        if pt['type'] == 'FLASH_REMELTING':
            print(f"  • Melted Node: {pt['melted_node']} | Converted Density: {pt['density_converted']:.2f}")
            print(f"  • Generated Shock Wave: {pt['shock_wave_generated']}")

    # Test Back-Traceable Partial Remelting
    print("\n🔍 Executing Back-traceable Partial Remelting on sub-component...")
    partial_remelt = adult_engine.backtrace_and_partial_remelt("N1", faulty_step_idx=0)
    print(f"  • Target Node: {partial_remelt['target_node']} | Fault Step: {partial_remelt['remelted_step_idx']}")
    print(f"  • Partial Wave Generated: {partial_remelt['partial_wave_generated']}")
    print(f"  • Node Density after Partial Remelting: {partial_remelt['new_node_density']:.2f}")

    # System Integration Sync
    print_banner("SCENARIO 4: Integration with Elysia Core Modules (Field & Reconstruction Engine)")
    field = CrystallizationField(128)
    rec_engine = PhaseTopologicalReconstructionEngine()

    adult_engine.sync_with_crystallization_field(field)
    sync_rec = adult_engine.sync_with_topological_reconstruction_engine(rec_engine)

    print("✅ Elysia Core Modules Synced Successfully:")
    print(f"  • Field Conductance Peak: {np.max(field.conductance):.2f}")
    print(f"  • Invariants Synced into 0-Ground Blueprints: {sync_rec['synced_invariants']} units")

    print_banner("SIMULATION COMPLETE: Causal Phase Transition Engine Operational")

if __name__ == "__main__":
    run_simulation()
