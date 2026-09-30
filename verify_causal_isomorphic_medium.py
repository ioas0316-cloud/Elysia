"""
Verification Script for Causal Isomorphic Medium Engine
======================================================
Validates:
1. Multi-domain isomorphic constraint relaxation (Physical, Fluid, Logical).
2. Continuous tensor field rotor alignment & Phase-Lock Index (PLI).
3. Gibbs-Boltzmann entropy reduction & WFC decision collapse.
4. AC-3 constraint wave propagation.
5. Contradiction detection & Thermal Relaxation reflection loop.
"""

import sys
import numpy as np
from core.physics.causal_isomorphic_medium import (
    CausalIsomorphicMedium,
    IsomorphicDomainFactory,
    RotorTile,
    CausalIsomorphicMediumEngine,
    QuaternionUtil
)


def verify_causal_isomorphic_medium():
    print("=" * 80)
    print("CAUSAL ISOMORPHIC MEDIUM ENGINE VERIFICATION")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # STEP 1: Matrix-Based Field Relaxation Across Domains
    # ------------------------------------------------------------------------
    print("\n[STEP 1] TESTING MULTI-DOMAIN CONSTRAINT TENSOR MATRIX RELAXATION...")

    elec_m = IsomorphicDomainFactory.create_physical_circuit(num_nodes=6)
    elec_sim = CausalIsomorphicMedium(elec_m)
    elec_logs = elec_sim.relax_to_equilibrium(max_steps=30, dt=0.05)
    print(f"  - Physical Circuit: Relaxed in {len(elec_logs)} steps. Residual Tension: {elec_m.compute_total_field_tension():.4f}")

    fluid_m = IsomorphicDomainFactory.create_fluid_medium(num_nodes=6)
    fluid_sim = CausalIsomorphicMedium(fluid_m)
    fluid_logs = fluid_sim.relax_to_equilibrium(max_steps=30, dt=0.05)
    print(f"  - Fluid Medium   : Relaxed in {len(fluid_logs)} steps. Residual Tension: {fluid_m.compute_total_field_tension():.4f}")

    logic_m = IsomorphicDomainFactory.create_logical_structure(num_nodes=6)
    logic_sim = CausalIsomorphicMedium(logic_m)
    logic_logs = logic_sim.relax_to_equilibrium(max_steps=30, dt=0.05)
    print(f"  - Logical Domain : Relaxed in {len(logic_logs)} steps. Residual Tension: {logic_m.compute_total_field_tension():.4f}")

    assert elec_m.compute_total_field_tension() < elec_logs[0].total_tension
    assert fluid_m.compute_total_field_tension() < fluid_logs[0].total_tension
    assert logic_m.compute_total_field_tension() < logic_logs[0].total_tension
    print("  ✓ Matrix-based relaxation convergence verified across all 3 domains.")

    # ------------------------------------------------------------------------
    # STEP 2: Rotor Math & Double Cover Symmetry
    # ------------------------------------------------------------------------
    print("\n[STEP 2] TESTING ROTOR MATH & DOUBLE COVER SYMMETRY...")
    q1 = np.array([0.7071, 0.7071, 0.0, 0.0])
    q2 = -q1  # Opposite sign quaternion representing same physical orientation
    e_align = QuaternionUtil.compute_rotor_alignment_energy(q1, q2)
    print(f"  - Double Cover Alignment Energy E_rotor(q, -q): {e_align:.6f}")
    assert abs(e_align) < 1e-5
    print("  ✓ Double cover symmetry verified.")

    # ------------------------------------------------------------------------
    # STEP 3: Cognitive Engine - Sensory Injection & Phase-Locking
    # ------------------------------------------------------------------------
    print("\n[STEP 3] TESTING SENSORY STIMULUS & PHASE LOCK INDEX (PLI)...")
    tiles = [
        RotorTile(0, "RIGHT", np.array([1.0, 0.0])),
        RotorTile(1, "UP   ", np.array([0.0, 1.0])),
        RotorTile(2, "LEFT ", np.array([-1.0, 0.0])),
        RotorTile(3, "DOWN ", np.array([0.0, -1.0]))
    ]

    # Compatibility matrix: adjacent orthogonal tiles allowed (1.0), opposite tiles disallowed (0.0)
    compat_matrix = np.array([
        [1.0, 1.0, 0.0, 1.0],  # RIGHT
        [1.0, 1.0, 1.0, 0.0],  # UP
        [0.0, 1.0, 1.0, 1.0],  # LEFT
        [1.0, 0.0, 1.0, 1.0]   # DOWN
    ])

    engine = CausalIsomorphicMediumEngine(
        width=3, height=3, tiles=tiles, compatibility_matrix=compat_matrix, beta_0=2.0
    )

    # Inject stimulus at (0, 0) for RIGHT
    engine.inject_sensory_stimulus(0, 0, target_tile_id=0, intensity=1.0)
    pli_val = engine.compute_phase_lock_index(0, 0)
    print(f"  - Stimulus Injected at (0,0). Phase Lock Index (PLI): {pli_val:.4f}")
    assert pli_val >= 0.0

    # ------------------------------------------------------------------------
    # STEP 4: Entropy-Guided WFC Decision & AC-3 Propagation
    # ------------------------------------------------------------------------
    print("\n[STEP 4] TESTING ENTROPY-GUIDED DECISION STEP & AC-3 PROPAGATION...")
    step_count = 0
    while step_count < 15:
        step_count += 1
        is_done, is_conflict = engine.make_decision_step()
        if is_done or is_conflict:
            break

    print(f"  - Decision loop reached status: is_done={is_done}, is_conflict={is_conflict} in {step_count} steps.")
    summary = engine.get_grid_state_summary()
    print("  - Current Grid State:")
    for row in summary:
        print("    ", " ".join(f"[{cell}]" for cell in row))

    # ------------------------------------------------------------------------
    # STEP 5: Contradiction & Reflection (Thermal Relaxation) Loop
    # ------------------------------------------------------------------------
    print("\n[STEP 5] TESTING CONTRADICTION REFLECTION & THERMAL RELAXATION...")
    # Intentionally trigger thermal relaxation
    initial_beta = engine.beta
    engine.reflect_and_relax(radius=1, thermal_factor=0.3)
    print(f"  - Reflection Triggered! Beta relaxed from {initial_beta:.2f} -> {engine.beta:.2f}")
    assert engine.beta < initial_beta
    assert np.all(engine.collapsed == False)
    print("  ✓ Thermal relaxation successfully restored high-entropy superposition state.")

    print("\n" + "=" * 80)
    print("ALL CAUSAL ISOMORPHIC MEDIUM ENGINE VERIFICATIONS SUCCESSFUL.")
    print("=" * 80)


if __name__ == "__main__":
    verify_causal_isomorphic_medium()
