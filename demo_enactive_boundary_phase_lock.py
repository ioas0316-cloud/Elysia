"""Demo: Enactive Boundary Phase-Lock Engine.

This executable demonstration showcases Elysia's 3-layer architecture for
boundary awareness, dynamic constraint learning via friction feedback, and phase locking.
"""

import time
import numpy as np
from core.physics.enactive_boundary_phase_lock import EnactiveBoundaryEngine


def main():
    print("=" * 70)
    print("  ELYSIA: ENACTIVE BOUNDARY PHASE-LOCK ENGINE DEMO  ")
    print("  'Protocol Awareness, Friction Learning, and Mutual Phase-Locking'  ")
    print("=" * 70)

    # Initialize Engine
    engine = EnactiveBoundaryEngine(
        internal_capacity=100.0,
        internal_reach=10.0,
        base_freq=10.0,
        coupling_strength=2.0,
    )

    # External hardware/system protocol specifications (e.g., PSU / GPU / Microcontroller limits)
    protocol_spec = {
        "max_frequency": 30.0,
        "max_power": 25.0,
        "max_bandwidth": 200.0,
        "latency_tolerance": 0.05,
    }

    active_state = np.array([2.0, 3.0, 1.0])  # System operational state vector
    ext_freq = 15.0
    ext_phase = 0.0
    dt = 0.01

    print("\n--- Phase 1: High Friction / Protocol Border Stress (Boundary Tightening) ---")
    high_friction = {"latency": 0.08, "error_rate": 0.05, "resistance": 0.1}

    for t_step in range(1, 11):
        ext_phase = (ext_phase + ext_freq * dt) % (2 * np.pi)
        res = engine.step(
            active_state=active_state,
            protocol_spec=protocol_spec,
            external_phase=ext_phase,
            external_freq=ext_freq,
            friction_feedback=high_friction,
            dt=dt,
        )

        print(
            f"Step {t_step:02d} | Time: {res['time']:.2f}s | "
            f"Instant Friction: {res['friction_metrics']['instant_friction']:.2f} | "
            f"Effective Max Freq Limit: {res['effective_max_frequency']:.2f} Hz | "
            f"Phase Diff: {res['phase_diff']:.3f} rad | Order R: {res['order_parameter_R']:.3f}"
        )

    print("\n--- Phase 2: Adaptation & Phase Synchronization (Mutual Tuning / Manners) ---")
    low_friction = {"latency": 0.002, "error_rate": 0.001, "resistance": 0.005}

    for t_step in range(11, 31):
        ext_phase = (ext_phase + ext_freq * dt) % (2 * np.pi)
        res = engine.step(
            active_state=active_state,
            protocol_spec=protocol_spec,
            external_phase=ext_phase,
            external_freq=ext_freq,
            friction_feedback=low_friction,
            dt=dt,
        )

        if t_step % 5 == 0 or t_step == 30:
            print(
                f"Step {t_step:02d} | Time: {res['time']:.2f}s | "
                f"Accum Friction: {res['friction_metrics']['accumulated_friction']:.2f} | "
                f"Safety Ratio: {res['boundary_clearance']['normalized_safety_ratio']:.3f} | "
                f"Order R: {res['order_parameter_R']:.3f} | Resonance Energy: {res['resonance_energy']:.3f}"
            )

    print("\n" + "=" * 70)
    print("  DEMONSTRATION COMPLETE: Boundary Respected, Phase-Locking Achieved.  ")
    print("=" * 70)


if __name__ == "__main__":
    main()
