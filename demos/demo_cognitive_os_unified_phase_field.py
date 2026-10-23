"""
Project Elysia: Cognitive OS & Unified Phase Field Interactive Demonstration
=============================================================================
Demonstrates Elysia operating as an active Cognitive Operating System:
1. Connecting software applications (Blender, CAD, Photoshop, Game Engines)
   and world fields (Particle/Fluid Sandboxes, Non-Euclidean Spacetime)
   as active sensory-motor organs.
2. Emitting continuous intent waves across organs and probing state transitions.
3. Dynamically discerning the "Boundary of Self" (Markov Blanket / 최초의 자아 경계)
   based on active inference friction error E and free energy gradients.
4. Fine-tuning Clifford phase rotors and accumulating Tacit Knowledge (노하우).
"""

import sys
import os
import numpy as np

# Ensure proper path loading
sys.path.extend(['.', 'L1_causal_field', 'L2_cognitive_engine'])

from elysia_core.cognitive_os_runtime import CognitiveOSRuntime
from L3_interface.simulators.cognitive_os_organ_adapters import (
    ParticleFluidSandboxOrgan,
    NonEuclideanSpacetimeOrgan,
    BlenderMeshOrgan,
    CADConstraintOrgan,
    PhotoshopRetinaOrgan,
    GameEnginePhysicsOrgan
)


def run_cognitive_os_demo():
    print("==========================================================================")
    print(" Project Elysia: Cognitive OS & Unified Phase Field Integration Demo")
    print("==========================================================================\n")

    # 1. Instantiate Cognitive OS Runtime
    print("[1] Initializing Cognitive OS Runtime and Causal Field Substrate...")
    runtime = CognitiveOSRuntime(dimension=5)

    # 2. Register Software & World Organs
    print("[2] Registering External Software & World Fields as Elysia's Sensory Organs...")
    fluid_organ = ParticleFluidSandboxOrgan(organ_id="sandbox_fluid_01")
    noneuclid_organ = NonEuclideanSpacetimeOrgan(organ_id="spacetime_field_01")
    blender_organ = BlenderMeshOrgan(organ_id="blender_tactile_01")
    cad_organ = CADConstraintOrgan(organ_id="cad_structure_01")
    photoshop_organ = PhotoshopRetinaOrgan(organ_id="photoshop_chroma_01")
    game_organ = GameEnginePhysicsOrgan(organ_id="game_physics_01")

    organs = [fluid_organ, noneuclid_organ, blender_organ, cad_organ, photoshop_organ, game_organ]
    for organ in organs:
        runtime.register_organ(organ)
        print(f"  -> Attached Organ: {organ.organ_id:<22} [{organ.organ_type}]")

    print("\n[3] Executing Perception-Action Cognitive Cycles across 10 Ticks...\n")

    intent_waves = [
        [0.8, 0.2, 0.1, 0.5, 0.3], # Cycle 1: Strong directional pulse
        [0.2, 0.9, 0.4, 0.1, 0.2], # Cycle 2: High order blue intention
        [0.5, 0.5, 0.8, 0.2, 0.1], # Cycle 3: High entropy yellow flux
        [0.1, 0.1, 0.2, 0.9, 0.7], # Cycle 4: Deep CAD/Mechanical deformation
        [0.7, 0.3, 0.5, 0.2, 0.9], # Cycle 5: High speed game physics thrust
        [0.4, 0.4, 0.4, 0.4, 0.4], # Cycle 6: Balanced equilibrium wave
        [0.9, 0.1, 0.1, 0.1, 0.1], # Cycle 7: Re-focusing direct control
        [0.3, 0.8, 0.2, 0.6, 0.4], # Cycle 8: Rotational bivector shift
        [0.6, 0.2, 0.7, 0.3, 0.5], # Cycle 9: Fluid phase transition drive
        [0.5, 0.5, 0.5, 0.5, 0.5]  # Cycle 10: Final stabilized state
    ]

    for step_idx, intent in enumerate(intent_waves, start=1):
        result = runtime.execute_cognitive_cycle(intent, context_name="multimodal_exploration")
        report = result["boundary_report"]
        tacit_summary = result["tacit_knowledge_summary"]

        print(f"--- Cycle {step_idx:02d} --- Intent: {intent}")
        print(f"  Total System Free Energy : {report.total_free_energy:.4f}")
        print(f"  Causal Gravity Radius    : {report.causal_gravity_radius:.4f}")
        print(f"  [Self Region (Direct)]   : {report.self_region_organs}")
        print(f"  [Markov Blanket Boundary]: {report.boundary_organs}")
        print(f"  [External Environment]   : {report.environment_organs}")

        # Sample Tacit Knowledge from two organs
        tk_fluid = tacit_summary.get("skill_sandbox_fluid_01", 0.0)
        tk_cad = tacit_summary.get("skill_cad_structure_01", 0.0)
        print(f"  Tacit Knowledge Accumulation -> Fluid Sandbox: {tk_fluid:.4f} | CAD: {tk_cad:.4f}")
        print()

    print("==========================================================================")
    print(" Cognitive OS Execution Complete: Boundary of Self successfully established!")
    print("==========================================================================")


if __name__ == "__main__":
    run_cognitive_os_demo()
