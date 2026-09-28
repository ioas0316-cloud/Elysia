"""
Verification script for Constructive Spacetime Axes in Elysia Engine.
Validates differential behavior across Algebraic, Geometric, and Causal Spacetime Axes.
"""

import numpy as np
from core.physics.constructive_causal_spacetime import ConstructiveSpacetimeAxis


def verify_spacetime_axes():
    print("=== [1/3] Verifying Constructive Spacetime Axes ===")
    axis = ConstructiveSpacetimeAxis(dimension=4)

    # Initial state verification
    state_0 = axis.apply_causal_impulse(np.zeros((4, 4)), dt=0.0)
    assert state_0["algebraic"]["impedance"] == 0.0, "Algebraic axis must have 0 impedance."
    assert state_0["geometric"]["curvature"] == 0.0, "Flat initial curvature expected."
    print("✓ Initial flat state verified.")

    # Apply causal impulse
    impulse = np.array([
        [0.8, 0.2, 0.0, 0.1],
        [0.2, 0.5, 0.1, 0.0],
        [0.0, 0.1, 0.3, 0.0],
        [0.1, 0.0, 0.0, 0.4]
    ])

    steps = 10
    for t in range(steps):
        state = axis.apply_causal_impulse(impulse, dt=0.1)

    # Validate non-trivial responses across 3 distinct axes
    alg_time = state["algebraic"]["time_t"]
    geom_curvature = state["geometric"]["curvature"]
    geom_radius = state["geometric"]["effective_radius"]
    causal_impedance = state["causal"]["impedance"]

    print(f"Algebraic Time (t): {alg_time:.4f} | Impedance: {state['algebraic']['impedance']}")
    print(f"Geometric Curvature: {geom_curvature:.4f} | Radius: {geom_radius:.4f}")
    print(f"Causal Impedance: {causal_impedance:.4f} | Phase Order: {state['causal']['phase_order']:.4f}")

    assert alg_time > 0.0, "Algebraic time must advance."
    assert abs(geom_curvature) > 1e-4, "Geometric curvature must deflect under causal impulse."
    assert causal_impedance > 1e-4, "Causal impedance must reflect physical tension."

    print("✓ Spacetime Axes Differential Behavior Verified Successfully!\n")


if __name__ == "__main__":
    verify_spacetime_axes()
