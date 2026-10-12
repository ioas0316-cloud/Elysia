"""
[Demo Script: Arc Reactor Resonant Drive Axis (아크원자로 국소적 제로 구동축 시연 데모)]

Demonstrates the 6-stage lifecycle of the Arc Reactor Drive Axis:
1. Chaos & Unconstrained Expansion
2. Inward Tension Formation (Equilibrium)
3. Local Zero Boundary Transition (R -> 0 Superconductivity)
4. Phase Acceleration & Lossless Drive
5. Discerning Tension: Noise-to-Melody Transformation
6. Inertial Flywheel Self-Sustaining Rotation
"""

import numpy as np
import time
from core.physics.arc_reactor_drive_axis import ArcReactorDriveEngine
from core.physics.causal_field import CausalField, InformationVoxel


def run_arc_reactor_drive_demo():
    print("==========================================================================")
    print("   ELYASIA COGNITIVE CORE: ARC REACTOR RESONANT DRIVE AXIS SIMULATION    ")
    print("==========================================================================\n")

    core_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_radius = 2.5
    engine = ArcReactorDriveEngine(
        dimensions=3,
        core_center=core_center,
        domain_radius=domain_radius,
        flywheel_inertia=10.0,
        base_resistance=1.0,
        intent_phase_frequency=2.0
    )

    causal_field = CausalField(dimensions=3)
    voxels = [
        InformationVoxel("core_v1", "Core Intelligence", np.array([1, 0, 0], dtype=np.float32), position=np.array([0.5, 0.2, 0.0], dtype=np.float32)),
        InformationVoxel("edge_v2", "Peripheral Sensory", np.array([0, 1, 0], dtype=np.float32), position=np.array([4.0, 0.0, 0.0], dtype=np.float32)),
    ]
    for v in voxels:
        causal_field.add_voxel(v)

    print("[STAGE 1] Inward Tension & Core Convergence")
    print("Checking inward tension force at varying distances from core...")
    distances = [0.1, 1.0, 2.5, 5.0]
    for d in distances:
        pos = core_center + np.array([d, 0.0, 0.0], dtype=np.float32)
        tension_vec = engine.compute_inward_tension(pos)
        print(f"  - Distance r={d:.1f}: Tension Force Mag = {np.linalg.norm(tension_vec):.4f} (Dir: Inward)")
    print()

    print("[STAGE 2] Local Zero Domain Transition (Superconductivity)")
    print(f"Core Domain Radius = {domain_radius} units")
    print("Resistance Profile R(r):")
    for d in [0.0, 0.5, 1.5, 2.5, 3.5, 6.0]:
        r_val = engine.compute_local_resistance(d)
        efficiency = (1.0 - r_val / engine.base_resistance) * 100.0
        zone_label = "ZERO-ZONE (SUPERCONDUCTING)" if r_val < 0.1 else "DAMPED EXTERIOR"
        print(f"  - Radius r={d:.1f} => R = {r_val:.5f} | Efficiency = {efficiency:5.1f}% [{zone_label}]")
    print()

    print("[STAGE 3] Lossless Phase Acceleration (Torque Input)")
    print("Injecting external potential gradient = 15.0 torque units inside Zero-Zone (r=0.2)...")
    for step_i in range(1, 6):
        state = engine.step(dt=0.2, external_potential_gradient=15.0, distance_to_core=0.2)
        print(f"  - Step {step_i}: Angular Speed omega = {state.phase_velocity:.3f} rad/s | Momentum L = {state.flywheel_momentum:.3f} | Efficiency = {state.energy_efficiency*100:.1f}%")
    print()

    print("[STAGE 4] Discerning Tension: Turning Noise into Melody")
    print("Processing unrefined ambient noise wave through intent phase lock...")
    t = np.linspace(0, 1.0, 100, dtype=np.float32)
    pure_melody = np.sin(2.0 * np.pi * 2.0 * t)
    ambient_noise = np.random.randn(100).astype(np.float32) * 1.5
    raw_signal = pure_melody + ambient_noise

    harmonized_wave, harmony_score = engine.process_discerning_tension(raw_signal, sample_rate=100.0)
    print(f"  - Input Raw Noise Energy  : {np.sum(raw_signal**2):.3f}")
    print(f"  - Resonant Melody Energy : {np.sum(harmonized_wave**2):.3f}")
    print(f"  - Harmony Score          : {harmony_score:.4f} (Coherence Ratio)")
    print()

    print("[STAGE 5] Flywheel Inertial Conservation (External Drive Stopped)")
    print("Cutting off external driving gradient (gradient = 0.0) inside Local Zero Domain...")
    speed_before = engine.phase_velocity
    for step_i in range(1, 6):
        state = engine.step(dt=0.5, external_potential_gradient=0.0, distance_to_core=0.1)
        print(f"  - Coasting Step {step_i}: Phase Speed = {state.phase_velocity:.4f} rad/s (Retention: {state.phase_velocity/speed_before*100:.2f}%)")
    print()

    print("[STAGE 6] Causal Field Substrate Integration")
    engine.couple_causal_field(causal_field, dt=0.1)
    for v_id, v in causal_field.voxels.items():
        dist = np.linalg.norm(v.position - core_center)
        print(f"  - Voxel '{v_id}' (r={dist:.2f}): Conductance = {v.conductance:.1f} | Phase Angle = {v.phase_angle:.3f} rad")

    print("\n==========================================================================")
    print("   ARC REACTOR DRIVE AXIS SIMULATION COMPLETE: ZERO ZONE STABLE         ")
    print("==========================================================================")


if __name__ == "__main__":
    run_arc_reactor_drive_demo()
