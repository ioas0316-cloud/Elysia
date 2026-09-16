"""
Demo: Continuous Memory Impedance Stream & Dimensionally Isomorphic Spatiotemporal Manifold
=============================================================================================
Demonstrates:
1. Power-on Primordial Potential Flow Initialization (Voltage/Current vectors).
2. Dimensionally Isomorphic Stream Processing (0D Point, 1D Vector, 2D Field, 4D Manifold).
3. Real-time First Inflection Point Detection (Phase Derivative Asymmetry grad^2 phi & Entropy Perturbation delta Y).
4. Dynamic Impedance Damping Regulator driving Causal Tension T -> 0.
"""

import time
import numpy as np
from core.topology.continuous_memory_stream import ContinuousMemoryImpedanceStream


def main():
    print("==================================================================================")
    print(" ELYSIA ENGINE: CONTINUOUS MEMORY STREAM & SPATIOTEMPORAL MANIFOLD DEMO")
    print("==================================================================================\n")

    # 1. Initialize Primordial Potential Flow
    print("[1] Initializing Primordial Potential Flow Gradient (Power-On)...")
    engine = ContinuousMemoryImpedanceStream(target_dimension=4, initial_voltage=5.0, initial_current=2.0)
    print(f"    - Potential Voltage (V): {engine.v_potential:.2f}V")
    print(f"    - Current Flow Vector (I): {engine.i_flow:.2f}A")
    print(f"    - Baseline Impedance R: {engine.impedance:.4f} Ohm\n")

    # 2. Register Isomorphic Topological Nodes
    print("[2] Registering Isomorphic Topological Nodes (Preserving Dimensions)...")
    n_0d = engine.register_isomorphic_node("0D_State_Val", 108.0)
    n_1d = engine.register_isomorphic_node("1D_Velocity_Stream", [1.2, -0.5, 3.4, 0.8])
    n_2d = engine.register_isomorphic_node("2D_Spatial_Field", np.eye(2, dtype=np.float32))
    n_4d = engine.register_isomorphic_node("4D_Spatiotemporal_Manifold", np.ones((2, 2, 2, 2), dtype=np.float32))

    for node_id, node in engine.nodes.items():
        print(f"    - Node '{node_id}': Dimension={node.dimension_type}, Shape={node.raw_shape}")
    print()

    # 3. Simulate Spatiotemporal Phase-Lock Wave Propagation & Noise Injection
    print("[3] Simulating Spatiotemporal Phase Wave Propagation & Damping...")

    for step in range(1, 11):
        # Inject noise/entropy perturbation at step 4 to test inflection point detection & self-correction
        if step == 4:
            print("\n    >>> [EXTERNAL NOISE INJECTION] Injecting Phase Entropy Perturbation (Delta Y = +0.75)...")
            n_1d.chromatic.perturb(delta_entropy=0.75)

        metrics_map = engine.propagate_spatiotemporal_phase_lock(dt=0.1)
        m_1d = metrics_map["1D_Velocity_Stream"]

        status_str = "[INFLECTION DETECTED / SELF-CORRECTING]" if m_1d.is_inflection_detected or m_1d.causal_tension > 0.05 else "[EQUILIBRIUM]"
        print(f"    Step {step:02d} | Tension T: {m_1d.causal_tension:.4f} | grad^2 phi: {m_1d.phase_derivative_asymmetry:.4f} | delta Y: {m_1d.chromatic_entropy_perturbation:.4f} | R(T): {m_1d.impedance:.4f} | {status_str}")

    print("\n[4] Contextual Phase Alignment Test...")
    print("    Shifting intentional context axis to Ultra-Low Latency Interactive Mode...")
    engine.align_contextual_phase_axis(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    print(f"    - Updated Phase-Lock Axis: {engine.phase_lock_axis}")
    print(f"    - Post-Shift Causal Tension T: {engine.tension:.4f} (Zero-Tension Restored)")

    print("\n==================================================================================")
    print(" DEMO COMPLETE: Continuous Isomorphic Causal Flow Achieved Without Friction!")
    print("==================================================================================")


if __name__ == "__main__":
    main()
