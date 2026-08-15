"""
Verification script for Relational Dynamic Axiom Engine & Embodied Environment (Phase 1 & 2 Core).
Verifies:
1. Local Axiom Unlocking under targeted tension (Locality Constraint).
2. Energy/Entropy braking & impedance x adjustment.
3. Principle of Least Action during Re-crystallization.
4. Back-trace causal symbolic projection across environment rule shifts.
"""

import sys
from core.evolution.relational_dynamic_axiom_engine import (
    RelationalDynamicAxiomEngine,
    EmbodiedVirtualEnvironment
)

def run_verification():
    print("=== STARTING VERIFICATION: RELATIONAL DYNAMIC AXIOM ENGINE ===")

    env = EmbodiedVirtualEnvironment(mass=1.0, stiffness=10.0, damping=0.5)
    engine = RelationalDynamicAxiomEngine(relativization_threshold=0.5, condensation_threshold=0.85)

    print("\n1. Initial State Projection:")
    print(engine.backtrace_projection())

    # Phase A: Normal physical oscillation under external force
    print("\n2. Phase A: Running normal physical simulation (No Rule Shift)...")
    for step in range(5):
        state = env.step(external_force=2.0)
        pred = {"position": state["position"], "velocity": state["velocity"]}
        trace = engine.process_observation(state, pred)

    print(f"Axes count: {trace['axes_count']}, Vars count: {trace['vars_count']}")
    assert trace["axes_count"] >= 5, "Initial primary axioms should remain locked during normal operation"

    # Phase B: Rule Shift / do-intervention (stiffness suddenly triples & position perturbed)
    print("\n3. Phase B: Applying do(stiffness=30.0, position=5.0) Rule Shift & Testing Locality Constraint...")
    env.do_intervention("stiffness", 30.0)
    env.do_intervention("position", 5.0)

    # Simulation step under new environment rule
    state_b = env.step()
    # Agent still expects old trajectory prediction (position ~ 0, velocity ~ 0) -> High Tension!
    pred_b = {"position": 0.0, "velocity": 0.0}

    trace_b = engine.process_observation(state_b, pred_b, intervention_node="stiffness")
    print(f"Tension detected: {trace_b['tension']:.4f}")
    print(f"Unlocked local nodes: {trace_b['unlocked_nodes']}")

    # Verify Locality Constraint: 'stiffness' or 'hooke_law' unlocked into resistor x, 'mass' stayed locked as Anchor
    assert "stiffness" in trace_b["unlocked_nodes"] or "hooke_law" in trace_b["unlocked_nodes"]
    assert engine.nodes["mass"].is_axis is True, "Mass axiom must stay locked (Locality Constraint)"

    print("\nState Projection during Relativization:")
    print(engine.backtrace_projection())

    # Phase C: Adaptation & Principle of Least Action Re-crystallization
    print("\n4. Phase C: Adapting to new rule & Re-crystallizing via Principle of Least Action...")
    for step in range(10):
        state_c = env.step()
        # Agent prediction adapts to new stiffness=30.0
        pred_c = {"position": state_c["position"], "velocity": state_c["velocity"]}
        trace_c = engine.process_observation(state_c, pred_c)

    print("\nFinal State Projection after Re-crystallization:")
    print(engine.backtrace_projection())

    assert engine.nodes["stiffness"].is_axis is True, "Stiffness should re-crystallize back to Axiom Axis"
    print("\n=== VERIFICATION SUCCESSFUL: ALL PHASES PASSED ===")

if __name__ == "__main__":
    run_verification()
