#!/usr/bin/env python3
"""
Elysia Core Engine - Meta-Causal Mechanism Ecosystem Demonstration

This script demonstrates the Mechanism-Centric Meta-Causal Architecture:
- Nodes represent active algorithms/constraints ('Mechanisms'), not static data values.
- Edges represent meta-causal bindings where structural changes in one mechanism reconfigure another.
- System converges to equilibrium through relaxation without explicit `if-else` branching.
- Introspection traces which mechanism drove state transitions.
"""

import time
from core.engine.meta_causal_map import (
    DifferentialBoundMechanism,
    HarmonicConservationMechanism,
    MetaCausalEngine,
)


def run_demo():
    print("=" * 80)
    print(" Elysia Core Engine: Meta-Causal Mechanism Ecosystem Demonstration")
    print(" 'Do not calculate, let it flow: Mechanism-as-Node Meta-Causality'")
    print("=" * 80)

    # 1. Initialize Meta-Causal Engine
    engine = MetaCausalEngine()

    # 2. Define Mechanism Nodes
    # Mechanism A: Differential Bound (|x - y| <= 2.0)
    mech_bound = DifferentialBoundMechanism("DifferentialBound_Alpha", max_diff=2.0)
    mech_bound.state[0] = 12.0
    mech_bound.state[1] = 1.0  # Initial diff = 11.0 (Violation!)

    # Mechanism B: Harmonic Conservation Invariant (x + y + z = 10.0)
    mech_harmonic = HarmonicConservationMechanism("HarmonicConservation_Beta", target_sum=10.0)
    mech_harmonic.state[0] = 5.0
    mech_harmonic.state[1] = 5.0
    mech_harmonic.state[2] = 5.0  # Initial sum = 15.0 (Violation!)

    engine.add_mechanism(mech_bound)
    engine.add_mechanism(mech_harmonic)

    # 3. Add Meta-Causal Binding
    # When Mechanism Alpha experiences high residual energy, it reconfigures Mechanism Beta's target parameters
    def meta_coupling_fn(src, tgt, weight):
        if tgt.parameters:
            shift = src.residual_energy * weight * 0.02
            old_param = tgt.parameters[0]
            tgt.parameters[0] = old_param + shift

    engine.add_binding("DifferentialBound_Alpha", "HarmonicConservation_Beta", coupling_weight=0.5, reconfigure_fn=meta_coupling_fn)

    print("\n[Step 1] Initial Ecosystem Setup")
    print(f"  - Mechanism Alpha (|x - y| <= {mech_bound.parameters[0]}): State = {mech_bound.state}")
    print(f"  - Mechanism Beta (x + y + z = {mech_harmonic.parameters[0]}): State = {mech_harmonic.state}")
    initial_residual = engine.compute_total_residual()
    print(f"  - Initial Total Ecosystem Residual Energy: {initial_residual:.4f}")

    # 4. Autonomous Relaxation Convergence Loop
    print("\n[Step 2] Executing Autonomous Relaxation Convergence (No if-else branching)...")
    start_time = time.time()
    iterations = engine.step_convergence(max_iterations=100, tolerance=1e-3, learning_rate=0.5)
    elapsed_ms = (time.time() - start_time) * 1000.0

    final_residual = engine.compute_total_residual()
    print(f"  - Convergence achieved in {iterations} iterations ({elapsed_ms:.2f} ms).")
    print(f"  - Final Total Ecosystem Residual Energy: {final_residual:.6f}")

    print("\n[Step 3] Post-Relaxation Mechanism States")
    print(f"  - Mechanism Alpha: State = {[round(s, 3) for s in mech_bound.state]} | Diff = {abs(mech_bound.state[0] - mech_bound.state[1]):.3f}")
    print(f"  - Mechanism Beta: State = {[round(s, 3) for s in mech_harmonic.state]} | Sum = {sum(mech_harmonic.state):.3f} (Reconfigured Target = {mech_harmonic.parameters[0]:.3f})")

    # 5. Introspection & Inverse Causal Exploration
    print("\n[Step 4] Introspection & Inverse Causal Exploration")
    contributions = engine.introspect_causal_contributions()
    for mech_id, ratio in contributions.items():
        print(f"  - Mechanism [{mech_id}]: {ratio * 100.0:.2f}% contribution to residual equilibrium shift")

    print("\n" + "=" * 80)
    print(" Meta-Causal Mechanism Ecosystem Demonstration Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
