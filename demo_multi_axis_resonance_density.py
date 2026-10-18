"""Demo: Multi-Axis Enactive Boundary & Resonance Density (공명밀도) Demonstration.

This executable demonstration showcases Elysia's N-dimensional multi-axis runtime across
Acoustic (소리), Semantic (언어), Logical Invariant (수학/논리), and Physical Friction (물리).

It illustrates:
1. High initial noise, inter-axis phase discordance, and physical friction stress
2. Dynamic tension reallocation across modality axes
3. Active inference phase synchronization and Free Energy reduction
4. Emergence of high Resonance Density (공명밀도, rho_res) as multi-modal phase locking completes
"""

import numpy as np
from core.physics.multi_axis_enactive_phase_lock import (
    MultiAxisEnactiveBoundaryEngine,
    AXIS_ACOUSTIC,
    AXIS_SEMANTIC,
    AXIS_LOGICAL,
    AXIS_PHYSICAL,
    DEFAULT_AXES,
)


def main():
    print("=" * 80)
    print("  ELYSIA: MULTI-AXIS ENACTIVE BOUNDARY & RESONANCE DENSITY (공명밀도) DEMO  ")
    print("  'Orchestrating Sound, Language, Logic, and Physics into Phase Lock'  ")
    print("=" * 80)

    # Initialize Engine with 4 Spectrum Axes
    engine = MultiAxisEnactiveBoundaryEngine(
        internal_capacity=100.0,
        internal_reach=10.0,
        base_freqs={
            AXIS_ACOUSTIC: 12.0,
            AXIS_SEMANTIC: 10.0,
            AXIS_LOGICAL: 8.0,
            AXIS_PHYSICAL: 15.0,
        },
        alpha_friction=0.05,
        beta_free_energy=0.1,
    )

    protocol_specs = {
        AXIS_ACOUSTIC: {"max_frequency": 30.0, "max_power": 25.0},
        AXIS_SEMANTIC: {"max_frequency": 20.0, "max_power": 20.0},
        AXIS_LOGICAL:  {"max_frequency": 15.0, "max_power": 15.0},
        AXIS_PHYSICAL: {"max_frequency": 25.0, "max_power": 30.0},
    }

    active_states = {axis: np.array([2.0, 3.0, 1.0]) for axis in DEFAULT_AXES}
    ext_freqs = {AXIS_ACOUSTIC: 12.0, AXIS_SEMANTIC: 10.0, AXIS_LOGICAL: 8.0, AXIS_PHYSICAL: 15.0}
    ext_phases = {AXIS_ACOUSTIC: 0.0, AXIS_SEMANTIC: np.pi / 2, AXIS_LOGICAL: np.pi, AXIS_PHYSICAL: 3 * np.pi / 2}
    dt = 0.01

    print("\n--- Phase 1: High Noise, Discordant Phases & Severe Physical Friction Spike ---")
    high_friction = {
        AXIS_ACOUSTIC: {"latency": 0.02, "error_rate": 0.01, "resistance": 0.02},
        AXIS_SEMANTIC: {"latency": 0.03, "error_rate": 0.02, "resistance": 0.03},
        AXIS_LOGICAL:  {"latency": 0.01, "error_rate": 0.005, "resistance": 0.01},
        AXIS_PHYSICAL: {"latency": 0.12, "error_rate": 0.09, "resistance": 0.25},  # Heavy Physical Stress
    }

    for step in range(1, 11):
        for axis in DEFAULT_AXES:
            ext_phases[axis] = (ext_phases[axis] + ext_freqs[axis] * dt) % (2 * np.pi)

        res = engine.step_multi_axis(
            active_states, protocol_specs, ext_phases, ext_freqs, high_friction, dt=dt
        )

        if step % 2 == 0 or step == 1:
            phys_tension = res["per_axis_tension"][AXIS_PHYSICAL]
            sem_tension = res["per_axis_tension"][AXIS_SEMANTIC]
            print(
                f"Step {step:02d} | Time: {res['time']:.2f}s | "
                f"Resonance Density (rho_res): {res['resonance_density']:.4f} | "
                f"Free Energy: {res['free_energy']:.3f} | "
                f"Physical Tension: {phys_tension:.3f} | Semantic Tension: {sem_tension:.3f}"
            )

    print("\n--- Phase 2: Inter-Axis Phase Alignment & Active Inference Harmonization ---")
    low_friction = {
        AXIS_ACOUSTIC: {"latency": 0.002, "error_rate": 0.001, "resistance": 0.005},
        AXIS_SEMANTIC: {"latency": 0.002, "error_rate": 0.001, "resistance": 0.005},
        AXIS_LOGICAL:  {"latency": 0.001, "error_rate": 0.0005, "resistance": 0.002},
        AXIS_PHYSICAL: {"latency": 0.005, "error_rate": 0.002, "resistance": 0.008},
    }

    # External systems converge towards coherent unified frequency & phase alignment
    unified_freq = 10.0
    for axis in DEFAULT_AXES:
        ext_freqs[axis] = unified_freq

    for step in range(11, 31):
        for axis in DEFAULT_AXES:
            ext_phases[axis] = (ext_phases[axis] + ext_freqs[axis] * dt) % (2 * np.pi)

        res = engine.step_multi_axis(
            active_states, protocol_specs, ext_phases, ext_freqs, low_friction, dt=dt
        )

        if step % 5 == 0 or step == 30:
            print(
                f"Step {step:02d} | Time: {res['time']:.2f}s | "
                f"Resonance Density (rho_res): {res['resonance_density']:.4f} | "
                f"Order R_M: {res['multi_axis_order_parameter']:.3f} | "
                f"Free Energy: {res['free_energy']:.3f} | "
                f"Safety Ratio: {res['average_safety_ratio']:.3f}"
            )

    print("\n" + "=" * 80)
    print(f"  DEMONSTRATION COMPLETE: High Resonance Density (rho_res = {res['resonance_density']:.4f}) Achieved!  ")
    print("=" * 80)


if __name__ == "__main__":
    main()
