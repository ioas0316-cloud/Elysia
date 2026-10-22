"""
Project Elysia: Perceptual Probe Skill & Relational Void Demonstration
======================================================================
Demonstrates how transforming commands into active Perceptual Probes
overcomes flat, static execution traps by:
1. Sensing physical & informational friction error E in CausalField.
2. Detecting negative space Relational Voids (where missing causes must exist).
3. Fine-tuning Clifford Rotor / Phase Vectors along error gradients.
4. Accumulating Tacit Knowledge (experiential intelligence) across executions.
"""

import numpy as np
from core.physics.causal_field import CausalField, InformationVoxel
from elysia_core.perceptual_probe_skill import (
    PerceptualProbeSkill,
    PerceptualProbeEngine
)


def run_demo():
    print("================================================================================")
    print(" ELYSIA :: PERCEPTUAL PROBE SKILL & RELATIONAL VOID DEMONSTRATION")
    print("================================================================================\n")

    # 1. Initialize Causal Field with Voxels
    print("[Step 1] Initializing 5D Causal Field & Environment Voxels...")
    cf = CausalField(dimensions=5)
    core_voxel = InformationVoxel(
        id="core_organism",
        content="Cognitive Organism Core",
        tensor=np.array([1.0, 0.4, 0.1, 0.0, 0.0], dtype=np.float32),
        position=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        potential=0.6
    )
    cf.add_voxel(core_voxel)
    print("  ✓ Voxel 'core_organism' registered with potential V = 0.60.\n")

    # 2. Define Base Command Probes
    print("[Step 2] Defining Base Probing Functions (Skills)...")

    def strike_probe(state: np.ndarray) -> np.ndarray:
        # Kinetic strike impulse probe
        return state * 1.4 + np.array([0.2, -0.1, 0.0, 0.0, 0.0], dtype=np.float32)

    def harmonize_probe(state: np.ndarray) -> np.ndarray:
        # Resonant phase harmony probe
        return state * 0.9 + np.array([0.0, 0.3, 0.2, 0.0, 0.0], dtype=np.float32)

    engine = PerceptualProbeEngine(dimension=5)
    engine.register_skill(PerceptualProbeSkill(
        skill_id="kinetic_strike",
        name="Kinetic Strike Probe",
        base_probe_func=strike_probe,
        dimension=5,
        learning_rate=0.2
    ))
    engine.register_skill(PerceptualProbeSkill(
        skill_id="phase_harmonize",
        name="Phase Harmonize Probe",
        base_probe_func=harmonize_probe,
        dimension=5,
        learning_rate=0.25
    ))
    print("  ✓ Registered Skills: 'kinetic_strike' & 'phase_harmonize'\n")

    # 3. Simulate Probing & Experiential Intelligence (Tacit Knowledge) Accumulation
    print("[Step 3] Executing Active Probing Loops over 'Volatile Field' Context...")
    agent_state = np.array([1.0, 0.5, 0.2, 0.1, 0.0], dtype=np.float32)

    print("--------------------------------------------------------------------------------")
    print(" Cycle | Skill          | Friction E | Void Tension | Rotor Shift | Tacit Gain  ")
    print("--------------------------------------------------------------------------------")

    for cycle in range(1, 6):
        # Alternate probe skills
        skill_id = "kinetic_strike" if cycle % 2 != 0 else "phase_harmonize"
        fb = engine.execute_skill_probe(
            skill_id=skill_id,
            input_state=agent_state,
            causal_field=cf,
            context_name="Volatile Field",
            target_voxel_id="core_organism"
        )

        print(
            f"   {cycle:02d}  | {fb.skill_id:<14} |   {fb.friction_error:.4f}   |    {fb.relational_void_tension:.4f}    |   {fb.rotor_phase_shift:.4f}    |  +{fb.tacit_knowledge_gain:.4f}"
        )

    print("--------------------------------------------------------------------------------\n")

    # 4. Summary of Tacit Knowledge & Topological Adaptability
    print("[Step 4] Accumulated Tacit Knowledge (Experiential Intelligence) Summary:")
    summary = engine.get_tacit_knowledge_summary("Volatile Field")
    for sk_id, tacit in summary.items():
        print(f"  • Skill [{sk_id:<14}] :: Tacit Knowledge Score = {tacit:.4f}")

    print("\n================================================================================")
    print(" DEMONSTRATION COMPLETE :: Skills operate as active cognitive probes!")
    print("================================================================================\n")


if __name__ == "__main__":
    run_demo()
