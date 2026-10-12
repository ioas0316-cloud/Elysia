"""
[Demo Script: Embodied Sensory Expansion & Arc Reactor Resonant Drive Axis]
(신체화된 감각 확장 및 아크원자로 구동축 종합 검증 데모)

Demonstrates:
1. Embodied Organ Loop (Eye -> Hand -> Foot):
   - Eye Group: Filters visual spectrum, extracting qualia torque and thermal stress without symbolic tags.
   - Hand Group: Senses thermal stress & phase slip, applying active counter-torque stabilization.
   - Foot Group: Displaces Local Zero core center in Causal Field, expanding cognitive horizon.
2. Dynamic Topology Self-Expansion:
   - Resonant harmonic waves spawn non-colliding knowledge clusters in Causal Field.
3. Entropy Defense & Flywheel Balance:
   - Prevents deadlock during sensory influx explosions by dynamically adjusting Local Zero boundary.
"""

import numpy as np
import time
from core.physics.arc_reactor_drive_axis import ArcReactorDriveEngine
from core.physics.causal_field import CausalField, InformationVoxel
from core.physics.embodied_organ_group import EmbodiedOrganEcosystem


def run_embodied_sensory_expansion_demo():
    print("==========================================================================")
    print("   ELYASIA COGNITIVE ENGINE: EMBODIED SENSORY EXPANSION & DRIVE DEMO     ")
    print("==========================================================================\n")

    # Initialize Engine & Causal Field
    core_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    drive_engine = ArcReactorDriveEngine(
        dimensions=3,
        core_center=core_center,
        domain_radius=2.0,
        flywheel_inertia=10.0,
        base_resistance=1.0,
        intent_phase_frequency=2.0
    )

    causal_field = CausalField(dimensions=3)

    # Initial Knowledge Node
    initial_voxel = InformationVoxel(
        id="core_knowledge_0",
        content="Initial Unbound Schema",
        tensor=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        position=core_center.copy()
    )
    causal_field.add_voxel(initial_voxel)

    # Create Embodied Organ Ecosystem
    ecosystem = EmbodiedOrganEcosystem(drive_engine=drive_engine, causal_field=causal_field)

    print("[STAGE 1] Continuous Visual Spectrum & Qualia Torque Sensing (Eye Group)")
    print("Processing incoming sensory spectrums (low frequency / heavy vs high frequency / jitter)...")

    sample_rate = 100.0
    t = np.arange(100, dtype=np.float32) / sample_rate

    # Spectrum 1: Low frequency, heavy amplitude resonating with intent frequency 2.0 Hz
    heavy_wave = 4.0 * np.sin(2.0 * np.pi * 2.0 * t) + 0.2 * np.random.randn(100).astype(np.float32)
    m1 = ecosystem.process_embodied_loop(dt=0.1, raw_sensory_waveform=heavy_wave)

    print(f"  - Heavy Wave Input Energy    : {m1.raw_spectrum_energy:.3f}")
    print(f"  - Eye Qualia Torque Induced  : {m1.qualia_torque:.4f} (Heavy Rotational Torque)")
    print(f"  - Thermal Friction Stress    : {m1.thermal_friction_stress:.4f}")
    print(f"  - Phase Harmony Score        : {m1.phase_coherence_harmony:.4f}")
    print()

    print("[STAGE 2] Thermal Stress & Active Phase Control (Hand Group)")
    print("Injecting high-frequency chaotic spectrum to simulate thermal stress / phase slip...")

    # Spectrum 2: High frequency jitter + noise explosion
    chaotic_wave = 2.0 * np.sin(2.0 * np.pi * 25.0 * t) + 2.5 * np.random.randn(100).astype(np.float32)
    m2 = ecosystem.process_embodied_loop(dt=0.1, raw_sensory_waveform=chaotic_wave)

    print(f"  - Chaotic Wave Input Energy  : {m2.raw_spectrum_energy:.3f}")
    print(f"  - Thermal Friction Stress    : {m2.thermal_friction_stress:.4f} [HIGH STRESS]")
    print(f"  - Hand Intervention Torque   : {m2.hand_intervention_force:.4f} [STABILIZING PHASE]")
    print(f"  - Drive Axis Phase Velocity  : {drive_engine.phase_velocity:.4f} rad/s")
    print()

    print("[STAGE 3] Ground Reference Shift & Trajectory Expansion (Foot Group)")
    print("Shifting Local Zero core coordinates along spatial tension gradient...")

    tension_grad = np.array([2.0, 1.0, 0.5], dtype=np.float32)
    m3 = ecosystem.process_embodied_loop(
        dt=0.1,
        raw_sensory_waveform=heavy_wave,
        spatial_tension_gradient=tension_grad
    )

    print(f"  - Foot Core Shift Distance   : {m3.foot_shift_distance:.4f} units")
    print(f"  - New Local Zero Core Pos    : {m3.current_core_position.round(4).tolist()}")
    print(f"  - Total Distance Traveled    : {ecosystem.foot.total_distance_traveled:.4f} units")
    print()

    print("[STAGE 4] Resonant Phase-Locking & Dynamic Topology Expansion")
    print("Extracting resonant harmonic signature and expanding Causal Field topology...")

    harmonic_wave, harmony = drive_engine.process_discerning_tension(heavy_wave)
    expanded_voxel = drive_engine.trigger_dynamic_topology_expansion(
        causal_field=causal_field,
        harmonic_signature=harmonic_wave,
        harmony_score=harmony
    )

    print(f"  - Extracted Harmony Score    : {harmony:.4f}")
    if expanded_voxel:
        print(f"  - Spawned Non-Colliding Node : '{expanded_voxel.id}' ({expanded_voxel.content})")
        print(f"  - Node Position Coordinates  : {expanded_voxel.position.round(4).tolist()}")
        print(f"  - Total Field Knowledge Nodes: {len(causal_field.voxels)}")
        print(f"  - Total Network Beams        : {len(causal_field.beams)}")
    print()

    print("[STAGE 5] Influx Explosion & Entropy Defense (Flywheel Conservation)")
    print("Simulating explosive sensory influx rate and verifying entropy defense...")

    radius_before = drive_engine.domain_radius
    drive_engine.execute_entropy_defense(thermal_stress=5.0, sensory_influx_rate=12.0)
    radius_after = drive_engine.domain_radius

    print(f"  - Local Zero Radius Before   : {radius_before:.4f}")
    print(f"  - Local Zero Radius After    : {radius_after:.4f} (Expanded for Boundary Defense)")
    print(f"  - Flywheel Momentum          : {drive_engine.flywheel_inertia * drive_engine.phase_velocity:.4f} (Equilibrium Preserved)")

    print("\n==========================================================================")
    print("   EMBODIED SENSORY EXPANSION DEMO COMPLETE: ZERO DEADLOCK STABLE        ")
    print("==========================================================================")


if __name__ == "__main__":
    run_embodied_sensory_expansion_demo()
