"""
Demo Rotational Cyclic Morphology (로터적 순환 및 위상 형태학 시연 스크립트)

Demonstrates:
1. Rotational Cyclic Engine: Phasor rotational movement A * e^(i*phi), phase restoration force,
   and energy conservation metrics (E_conservation).
2. Phase Morphology Field: Scale Frame of Reference transformation from discrete cube/simplex
   lattice (Scale < 0.5) to continuous morphing line/curve (0.5 <= Scale <= 2.0) to rotational circle/sphere/curl (Scale > 2.0).
3. Envelope Area Block: Turning discrete 0/1 bit pulse sequences into continuous envelope area blocks
   (Rasterization/Filling) and instantiating them as macro-unit voxels in CausalField with filling ratio & density match metrics.
"""

import numpy as np
from core.physics.causal_field import CausalField
from core.physics.rotational_cyclic_morphology import (
    RotationalCyclicEngine,
    PhaseMorphologyField,
    EnvelopeAreaBlock,
)


def run_demo():
    print("=" * 80)
    print(" [ELYSIUM CORE] ROTATIONAL CYCLIC MORPHOLOGY DEMO")
    print("=" * 80)

    # 1. Rotational Cyclic Engine Demonstration
    print("\n--- 1. ROTATIONAL CYCLIC ENGINE (로터적 순환 엔진) ---")
    rotor_engine = RotationalCyclicEngine(default_frequency=2.0 * np.pi, damping=0.0)
    p1 = rotor_engine.register_phasor("alpha_rotor", amplitude=2.0, phase=0.0)
    p2 = rotor_engine.register_phasor("beta_rotor", amplitude=1.5, phase=np.pi / 2)

    print(f"Registered Phasors: 'alpha_rotor' (A={p1.amplitude}, phi={p1.phase:.2f} rad), 'beta_rotor' (A={p2.amplitude}, phi={p2.phase:.2f} rad)")

    print("\nEvolving Rotational Cycle over 5 steps...")
    for step in range(1, 6):
        diag = rotor_engine.step(dt=0.1)
        print(
            f" Step {diag['cycle']}: Energy Conservation Ratio = {diag['energy_conservation_ratio']:.6f} | "
            f"Alpha Phasor: ({diag['phasors']['alpha_rotor']['real']:.3f} + {diag['phasors']['alpha_rotor']['imag']:.3f}j)"
        )

    print("\nApplying Phase Restoration Force (위상 복원력) towards Equilibrium (phi = 0.0)...")
    rotor_engine.apply_restoration_force("beta_rotor", target_phase=0.0, coupling_k=1.5, dt=0.2)
    diag_restored = rotor_engine.step(dt=0.1)
    print(f" Post-restoration 'beta_rotor' Phase: {diag_restored['phasors']['beta_rotor']['phase']:.3f} rad")

    # 2. Phase Morphology Field Demonstration
    print("\n--- 2. PHASE MORPHOLOGY FIELD (위상 형태 장 & 스케일 관측 변환) ---")
    morph_field = PhaseMorphologyField(observation_scale=1.0)
    sample_wave_pts = np.array(
        [[i * 0.1, np.sin(i * 0.5), np.cos(i * 0.5)] for i in range(10)],
        dtype=np.float32,
    )
    phasor_list = list(rotor_engine.phasors.values())

    scales_to_test = [0.2, 1.0, 3.0]
    for scale in scales_to_test:
        morph_field.set_scale(scale)
        res = morph_field.project_waveform_geometry(sample_wave_pts, phasors=phasor_list)
        print(f" [Scale {scale:.1f}] Morph Type: {res['morph_type']}")
        print(f"             Topological Invariant: {res['topological_invariant']:.4f}")
        print(f"             Sample Geometry Point 0: {res['projected_geometry'][0].round(3).tolist()}")

    # 3. Envelope Area Block & Bit Pulse Morphing
    print("\n--- 3. ENVELOPE AREA BLOCK & BIT PULSE FILLING (포락선 면적 채우기 및 매크로 유닛 변환) ---")
    # Discrete bit pulse train (0s and 1s)
    bit_stream = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1], dtype=np.int32)
    print(f" Input Discrete Bit Pulse Train (16-bit): {bit_stream.tolist()}")

    area_block = EnvelopeAreaBlock(block_id="macro_unit_01", grid_resolution=(8, 8))
    envelope_res = area_block.extract_envelope_and_fill(bit_stream, amplitude_scale=1.0)

    print(f" Envelope Extraction Completed!")
    print(f"  - Filled Area: {envelope_res['filled_area']:.2f} / Ideal Area: {envelope_res['ideal_area']:.2f}")
    print(f"  - Filling Ratio: {envelope_res['filling_ratio']:.4f}")
    print(f"  - Density Match Precision: {envelope_res['density_match_percent']:.2f}%")

    # 4. Integration with CausalField
    print("\n--- 4. CAUSAL FIELD INTEGRATION (인과 장 연동 및 매크로 유닛 승격) ---")
    causal_field = CausalField(dimensions=3)
    created_voxels = area_block.convert_to_causal_voxels(envelope_res, causal_field)
    print(f" Converted Bit Pulse Block into {len(created_voxels)} Continuous Macro-Unit Voxels in CausalField.")

    # Advance causal step
    causal_field.step(dt=0.1)
    print(f" Causal Field Step Executed. Total Active Voxels: {len(causal_field.voxels)}, Active Beams: {len(causal_field.beams)}")

    print("\n" + "=" * 80)
    print(" ROTATIONAL CYCLIC MORPHOLOGY DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
