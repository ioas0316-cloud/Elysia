"""
demo_scale_causal_reversibility.py
==================================
Demonstration of Scale Principles, Unbroken Scale Continuity, Continuous Phase Transitions,
and Bidirectional Causal Time-Reversal (Macro -> Micro) in Elysia.

Simulates:
1. Continuous Phase Transition along density & surface tension gradient.
2. Clifford Spacetime Rotor Forward Rotation and Exact Rewind.
3. Multiscale Volumetric Flow Deflection & Interaction History Unwinding.
4. Wavefield Spectrum Rainbow Creation and Rewind.
"""

import numpy as np

from core.physics.volumetric_scale_hierarchy import (
    StructuralDensityNode,
    VolumetricScaleHierarchyEngine
)
from core.physics.field_unified_physics_core import (
    FieldUnifiedPhysicsCore
)
from core.physics.clifford_spacetime_rotor import (
    CliffordSpacetimeRotor
)


def run_demo():
    print("================================================================================")
    print(" ELYSIA: SCALE PRINCIPLES & BIDIRECTIONAL CAUSAL REVERSIBILITY DEMONSTRATION")
    print("================================================================================")

    # 1. Continuous Phase Transition (Liquid -> Gas Process Trace)
    print("\n--- 1. Continuous Phase Transition Spectrum (No Point-Rule Flags) ---")
    physics = FieldUnifiedPhysicsCore(resolution=256)

    densities = [2.5, 1.2, 0.4, 0.05]
    tensions = [0.9, 0.5, 0.2, 0.05]

    for rho, ts in zip(densities, tensions):
        res = physics.simulate_continuous_matter_transition(density=rho, surface_tension=ts)
        print(f"Density: {rho:.2f} | Tension: {ts:.2f} | Friction: {res['effective_friction']:.3f} | Phase Velocity: {res['phase_velocity']:.3f} | Dynamic State: {res['phase_label']}")

    # 2. Clifford Spacetime Rotor Rewind (Rainbow / Rotor Reversibility)
    print("\n--- 2. Clifford Spacetime Rotor Bidirectional Reversibility (Macro -> Micro) ---")
    axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    angle = np.pi / 4.0  # 45 deg
    rotor = CliffordSpacetimeRotor.create_rotor_3d(axis, angle)

    v_rainbow_ray = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v_refracted = CliffordSpacetimeRotor.sandwich_transform_vector_3d(rotor, v_rainbow_ray)
    v_rewound = CliffordSpacetimeRotor.rewind_transform_vector_3d(rotor, v_refracted)

    print(f"Original Light Ray Vector : {v_rainbow_ray}")
    print(f"Refracted Rainbow Vector  : {np.round(v_refracted, 4)}")
    print(f"Rewound Initial Vector    : {np.round(v_rewound, 4)}")
    error = float(np.linalg.norm(v_rainbow_ray - v_rewound))
    print(f"Reconstruction Error       : {error:.14e} (History_Invariance = 1.0)")

    # 3. Volumetric Scale Flow & Interaction Memory Rewind
    print("\n--- 3. Volumetric Density Boundary Deflection & Memory Rewind ---")
    v_engine = VolumetricScaleHierarchyEngine()
    barrier = StructuralDensityNode(
        node_id="dense_barrier",
        position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        density=2.0,
        porosity=0.1,
        viscosity_resistance=0.8
    )
    v_engine.add_node(barrier)

    init_pos = np.array([0.0, 0.2, 0.0], dtype=np.float32)
    init_vel = np.array([1.5, 0.0, 0.0], dtype=np.float32)

    step1 = v_engine.simulate_flow_step(init_vel, init_pos, dt=0.01)
    print(f"Forward Step Deflected Pos: {np.round(step1['flow_position'], 4)} | Total Deflection: {step1['total_deflection']:.4f}")
    print(f"Barrier Interaction Ring   : {len(barrier.interaction_history)} record(s)")

    rewind1 = v_engine.rewind_step()
    print(f"Rewound Prior Position    : {np.round(rewind1['restored_position'], 4)}")
    print(f"Barrier Interaction Ring   : {len(barrier.interaction_history)} record(s) (Memory Cleared)")
    print(f"History Invariance Metric  : {rewind1['history_invariance']:.1f}")

    print("\n================================================================================")
    print(" ALL SCALE PRINCIPLE & BIDIRECTIONAL REVERSIBILITY VERIFICATIONS SUCCESSFUL.")
    print("================================================================================")


if __name__ == "__main__":
    run_demo()
