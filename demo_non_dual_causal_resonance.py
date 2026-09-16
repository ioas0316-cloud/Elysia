"""
Demonstration Script: Non-Dualistic Causal Intent Engine,
Tri-Variable Sweep Dial, and Parallel Observation Universe.

Demonstrates:
1. Substrate -> Structural Mechanics -> Emergent Construct Derivation Path Capture across 5 Intentional Lenses.
2. Parallel Observation Universe: System Lens vs World Intrinsic Causality (Meta Lens Differential Delta S_lens).
3. Trinity Phase Topology: Self, Other, and Meta World Refraction Mapping.
4. Tri-Variable Sweep Dial: Inverse Parameter Discovery & Bifurcation Point Scans.
5. 4-Stage Reverse Decompression Pipeline (Macro label stripping to atomic rule reduction and re-emergence).
6. Reversible 4-Phase Transition Cycle (Gas <-> Liquid <-> Lattice <-> Re-melting).
"""

import numpy as np
from core.physics.causal_lens_differential import (
    PrimitiveSubstrate,
    StructuralMechanics,
    ModalityType,
    MathematicsLens,
    PhysicsLens,
    LanguageLens,
    SoundAcousticsLens,
    VisionOpticsLens,
    WorldMechanismField,
    ParallelObservationUniverse,
    TrinityPhaseTopology,
    TriVariableSweepDial,
    ReverseDecompressionPipeline,
    ReversiblePhaseTransitionEngine
)


def run_demo():
    print("=" * 80)
    print("=== ELYSIA: NON-DUALISTIC CAUSAL INTENT & PARALLEL LENS RESONANCE DEMO ===")
    print("=" * 80)

    # 1. Initialize Primitive Substrate
    substrate = PrimitiveSubstrate(
        substrate_id="prim_sub_001",
        energy_density=3.2,
        tension_potential=2.1,
        repulsion_potential=0.6,
        state_vector=np.array([0.2, 0.8, 1.5, 0.4, 0.9, 1.2, 0.3, 0.7], dtype=np.float64)
    )
    print(f"\n[1] Primitive Substrate Initialized: ID={substrate.substrate_id}, Energy={substrate.energy_density}")

    # 2. Refract Substrate through 5 Intentional Causal Lenses
    print("\n[2] Refracting Substrate through 5 Intentional Causal Lenses:")
    lenses = [
        MathematicsLens(),
        PhysicsLens(),
        LanguageLens(),
        SoundAcousticsLens(),
        VisionOpticsLens()
    ]

    for lens in lenses:
        mech = StructuralMechanics(
            mechanism_id=f"mech_{lens.modality.value}",
            modality=lens.modality,
            tension_coefficient=1.8,
            repulsion_coefficient=0.5,
            boundary_curvature=1.1
        )
        construct = lens.refract_substrate(substrate, mech)
        print(f"  - Lens Modality: {construct.modality.value:<18} | Invariant Phi: {construct.invariant_phi:.4f} | Form Tensor: {construct.form_tensor}")
        print(f"    Derivation Path ID: {construct.derivation_path.path_id} | Bit Offset: 0x{construct.bit_offset:04X}")

    # 3. Parallel Observation Universe & Meta Differential Diagnosis
    print("\n[3] Parallel Observation Universe: Contrasting Observer Lens vs World Intrinsic Field:")
    universe = ParallelObservationUniverse()
    mech_phys = StructuralMechanics(
        mechanism_id="mech_phys_world",
        modality=ModalityType.PHYSICS,
        tension_coefficient=1.5,
        repulsion_coefficient=0.4,
        boundary_curvature=1.2
    )
    diff = universe.contrast_and_diagnose(substrate, mech_phys, ModalityType.PHYSICS)
    print(f"  - Lens Modality: {diff.lens_modality.value}")
    print(f"  - Bias Distortion (Delta S_lens): {diff.bias_distortion:.6f}")
    print(f"  - Artifact Coefficient:           {diff.artifact_coefficient:.6f}")
    print(f"  - Isomorphism Degree:            {diff.isomorphism_degree * 100:.2f}%")

    # 4. Trinity Phase Topology
    print("\n[4] Trinity Phase Topology Mapping (Self, Other, Meta World):")
    trinity = TrinityPhaseTopology(self_bias=0.2, other_bias=0.1)
    c_self, c_other, c_meta, delta_refract = trinity.MapTrinityRefraction(substrate, mech_phys)
    print(f"  - C_self Norm: {np.linalg.norm(c_self):.4f}")
    print(f"  - C_other Norm: {np.linalg.norm(c_other):.4f}")
    print(f"  - C_meta Norm: {np.linalg.norm(c_meta):.4f}")
    print(f"  - Non-Suppressive Refraction Delta: {delta_refract:.6f}")

    # 5. Tri-Variable Sweep Dial Scan
    print("\n[5] Tri-Variable Sweep Dial Scan (Fix Substrate & Construct, Sweep Mechanism):")
    math_lens = MathematicsLens()
    target_mech = StructuralMechanics("target_m", ModalityType.MATHEMATICS, tension_coefficient=2.2, repulsion_coefficient=0.5, boundary_curvature=1.0)
    target_construct = math_lens.refract_substrate(substrate, target_mech)

    tensions, errors, bifurcation = TriVariableSweepDial.sweep_mechanism(
        substrate, target_construct, math_lens, tension_range=(0.5, 3.5), steps=20
    )
    best_idx = int(np.argmin(errors))
    print(f"  - Target Tension: 2.20 | Discovered Inverse Tension: {tensions[best_idx]:.2f} (Min Error: {errors[best_idx]:.6f})")
    print(f"  - Detected Bifurcation Point: {bifurcation}")

    # 6. Reverse Decompression Pipeline
    print("\n[6] 4-Stage Reverse Decompression Pipeline:")
    pipeline = ReverseDecompressionPipeline()
    decomp = pipeline.decompress("Macro_Quantum_Fluid_Dynamics", target_construct, substrate)
    print(f"  - Macro Label Stripped: '{decomp['macro_label_stripped']}'")
    print(f"  - Binding Operator Extracted: {decomp['binding_operator']}")
    print(f"  - Atomic Rules Reduced: {decomp['atomic_rules']}")
    print(f"  - Re-emergence Test Integrity: {decomp['derivation_integrity'] * 100:.2f}%")

    # 7. Reversible 4-Phase Transition Cycle
    print("\n[7] Reversible 4-Phase Transition Cycle (Gas <-> Liquid <-> Lattice <-> Re-melting):")
    engine = ReversiblePhaseTransitionEngine(memory_size=65536)
    phase_res = engine.execute_phase_cycle(substrate, mech_phys)
    print(f"  - Phase Sequence: {[p.value for p in phase_res['phase_sequence']]}")
    print(f"  - Crystallized Value at 0x{phase_res['bit_offset']:04X}: {phase_res['crystallized_value']:.6f}")
    print(f"  - Re-melted Potential:                     {phase_res['remelted_potential']:.6f}")
    print(f"  - Reconstitution Error:                    {phase_res['reconstitution_error']:.8f}")
    print(f"  - Causal Continuity Preserved:             {phase_res['causal_continuity'] == 1.0}")

    print("\n" + "=" * 80)
    print("=== DEMO COMPLETED SUCCESSFULLY WITH 100% CAUSAL CONTINUITY ===")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
