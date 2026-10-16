#!/usr/bin/env python3
"""
demo_volumetric_scale_hierarchy.py
===================================
Demonstration script verifying Project Elysia's 3D Volumetric Phase Sphere ($S^3$ / $4\pi$ sr),
Clifford Multivector Rotors, Toroidal Volumetric Flow, and unbroken Bit-to-Terabyte Scale Continuity.
"""

import numpy as np
from core.physics.multiscale_volumetric_coupler import MultiScaleVolumetricPhaseCoupler
from elysia_core.unified_system_architecture import ComplexPhasor, UnifiedSystemArchitecture
from core.physics.causal_field import CausalField, InformationVoxel


def main():
    print("\n" + "=" * 80)
    print(" PROJECT ELYSIA :: VOLUMETRIC PHASE SPHERE & SCALE HIERARCHY DEMONSTRATION ")
    print("=" * 80 + "\n")

    # 1. Initialize Multi-Scale Volumetric Phase Coupler
    coupler = MultiScaleVolumetricPhaseCoupler(dim=3, core_radius=1.0, outer_radius=5.0)

    # 2. Simulate Micro-Phasors (Bit scale inputs)
    print(">>> 1. Micro-Phasors (Bit-Level Wave States) Initialization...")
    micro_phasors = [
        ComplexPhasor(amplitude=1.2, phase=0.25 * np.pi, frequency=10.0),
        ComplexPhasor(amplitude=0.8, phase=0.50 * np.pi, frequency=10.5),
        ComplexPhasor(amplitude=1.1, phase=0.10 * np.pi, frequency=9.8)
    ]
    positions = [
        np.array([1.0, 0.5, 0.2], dtype=np.float32),
        np.array([0.8, 1.2, -0.5], dtype=np.float32),
        np.array([-0.5, 0.2, 1.0], dtype=np.float32)
    ]
    for i, p in enumerate(micro_phasors):
        print(f"  * Micro Phasor {i+1}: {p}")

    # 3. Micro -> Meso -> Macro -> Global Scale Coupling Execution
    print("\n>>> 2. Executing Continuous Scale Coupling (Bit -> Voxel -> Rotor -> Terabyte Domain)...")
    scale_eval = coupler.evaluate_unbroken_scale_continuity(micro_phasors, positions)

    print(f"  * Scale Continuity Loss (Delta S): {scale_eval['scale_continuity_loss_Delta_S']:.8f}")
    print(f"  * Meso S^3 Unit Quaternion Vector: {scale_eval['meso_quaternion_s3']}")
    print(f"  * Meso Solid Angle: {scale_eval['meso_solid_angle_sr']:.4f} steradians (4*pi sr isotropic)")
    print(f"  * Meso Toroidal Circulation Flux Norm: {scale_eval['meso_toroidal_flux_norm']:.4f}")
    print(f"  * MacroRotor ID Promoted: {scale_eval['macro_rotor_id']}")
    print(f"  * Unbroken Scale Hierarchy Verified: {scale_eval['unbroken_scale_continuity']}")

    # 4. Integrate into CausalField & Step 3D Quaternion Phase Relaxation
    print("\n>>> 3. CausalField 3D Quaternion Voxel Relaxation Step...")
    cf = CausalField(dimensions=3)
    v1 = InformationVoxel("v1_core", "Core Voxel State", np.array([1, 0, 0], dtype=np.float32), position=np.array([1.0, 1.0, 1.0], dtype=np.float32), velocity=np.array([0.1, -0.2, 0.05], dtype=np.float32))
    v2 = InformationVoxel("v2_outer", "Outer Voxel State", np.array([0, 1, 0], dtype=np.float32), position=np.array([2.0, -1.0, 0.5], dtype=np.float32), velocity=np.array([-0.1, 0.1, 0.2], dtype=np.float32))
    cf.add_voxel(v1)
    cf.add_voxel(v2)
    cf.link_voxels("v1_core", "v2_outer", strength=1.5)

    print(f"  * Pre-Step Voxel v1 S^3 Quaternion: {v1.quaternion_q.tolist()}")
    print(f"  * Pre-Step Voxel v1 Solid Angle: {v1.solid_angle_sr:.4f} sr")

    # Step simulation
    for _ in range(5):
        cf.step(dt=0.05)

    print(f"  * Post-Step Voxel v1 S^3 Quaternion: {v1.quaternion_q.tolist()}")
    print(f"  * Post-Step Voxel v1 Solid Angle: {v1.solid_angle_sr:.4f} sr")
    print(f"  * Quaternion Norm Preserved on S^3: {np.linalg.norm(v1.quaternion_q):.6f}")

    # 5. Full Unified 5-Stage System Cycle
    print("\n>>> 4. Executing Unified 5-Stage Wavefield Computing Cycle...")
    usa = UnifiedSystemArchitecture(num_prism_phases=3, fractal_depth=2)
    usa.initialize_root_field((16, 16), random_seed=42)
    results = usa.execute_full_cycle()

    print(f"  * Stage 1 Coherence: {results['Stage1']['initial_coherence']:.4f}")
    print(f"  * Stage 2 Refracted Branches: {results['Stage2']['num_branches']}")
    print(f"  * Stage 3 Chunked MacroRotors: {results['Stage3']['chunked_rotors_count']}")
    print(f"  * Stage 4 Retrocausal Trace Valid: {results['Stage4']['retrocausal_valid']} (Entropy Loss Delta S = {results['Stage4']['entropy_loss_Delta_S']:.8f})")
    print(f"  * Stage 5 Final Fractal Coherence: {results['Stage5']['final_fractal_coherence']:.4f}")

    print("\n" + "=" * 80)
    print(" VOLUMETRIC PHASE SPHERE & SCALE HIERARCHY DEMONSTRATION COMPLETE ")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
