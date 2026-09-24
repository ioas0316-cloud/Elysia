"""
Elysia Demonstration: Morphological Causal Resonance Pipeline

This CLI demonstration showcases the end-to-end loop of:
1. Sensory Grounding via Bidirectional Phase Negotiation
2. Evolutionary Anchor Query and ICE Crystallization
3. Morphological Plasticity convergence under changing physical environment pressures
"""

import math
import time
from core.embodied.phase_negotiation import BidirectionalPhaseNegotiator, SensoryWaveStream
from core.evolution.dna_anchor import CrystallizedDNAAnchor
from core.embodied.morphological_engine import MorphologicalPlasticityEngine, EnvironmentPressure


def run_morphological_demo():
    print("================================================================================")
    print("   🌌 ELYSIA COGNITIVE ENGINE: MORPHOLOGICAL CAUSAL RESONANCE DEMO 🌌")
    print("================================================================================")
    print("  'From random mutations to causal resonance: How environmental pressure shapes morphology'\n")

    engine = MorphologicalPlasticityEngine()

    # Define environmental transition stages
    stages = [
        ("STAGE 1: Dense Fluid Environment (Hydrodynamic Drag Stress)", EnvironmentPressure(fluid_density=3.0, current_velocity=2.5)),
        ("STAGE 2: High Airflow Turbulence (Aerodynamic Lift Pressure)", EnvironmentPressure(air_flow_velocity=6.0, current_velocity=4.0)),
        ("STAGE 3: Severe Resource Scarcity (Hunger & Tool Articulation Demand)", EnvironmentPressure(resource_scarcity=5.0, current_velocity=1.0)),
        ("STAGE 4: Rough Terrestrial Surface (Structural Rigidity Demand)", EnvironmentPressure(terrain_roughness=4.5, current_velocity=1.5)),
    ]

    for stage_name, env in stages:
        print(f"\n🌊 >>> {stage_name}")
        print("--------------------------------------------------------------------------------")

        for step in range(1, 11):
            res = engine.adapt_morphology(env, time_delta=0.1, morph_rate=0.25)
            genome = engine.current_genome

            print(
                f" Step {step:2d} | Stress: {res['stress']:.2f} | q_err: {res['q_err']:+.3f} | "
                f"Resonance: {res['resonance']:.2f} | Target: {res['target_anchor'][:20]}..."
            )
            print(
                f"          ├─ Drag Coeff: {genome.drag_coefficient:.3f} | Lift Coeff: {genome.lift_coefficient:.3f} | "
                f"Grasp Art: {genome.grasp_articulation:.3f} | Rigidity: {genome.structural_rigidity:.3f}"
            )

    print("\n================================================================================")
    print("❄️ [Crystallized DNA Anchors Inventory]")
    for anchor_id, anchor in engine.dna_crystallizer.crystallized_anchors.items():
        print(f"  • [{anchor_id}] {anchor.name} (Resonance Freq: {anchor.resonance_frequency:.1f} Hz)")

    print("\n✨ Morphological Adaptation Pipeline Completed Successfully!")
    print("================================================================================")


if __name__ == "__main__":
    run_morphological_demo()
