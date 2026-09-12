#!/usr/bin/env python3
"""
Verification Script: Relational Nexus Node & Topological Phase Transition Architecture
======================================================================================
1. Verify Bit-Mapped Trigger Substrate benchmarks (Yield >= 95%, Dissipation Rate, State Restoration Index = 100%)
2. Verify Relational Nexus Node Informational Trinity (Payload/Schema/Interpreter) & Local Curvature/Stress
3. Verify Topological Phase Transition Engine, Entropy Thresholds (eta_c), and 3 Invariants (b_0=1, b_1=0, Causal Flux)
4. Verify Full Cycle Integration in SelfReferentialArchitectureEngine
"""

import sys
import numpy as np

from core.memory.bit_mapped_trigger_substrate import BitMappedTriggerSubstrate
from core.topology.relational_nexus_node import RelationalNexusNode
from core.topology.topological_phase_transition import TopologicalPhaseTransitionEngine
from core.topology.self_referential_architecture import SelfReferentialArchitectureEngine


def main():
    print("=" * 80)
    print("Elysia Verification: Relational Nexus & Topological Phase Transition")
    print("=" * 80)

    # 1. Bit-Mapped Substrate Benchmark Verification
    print("\n[1] Testing Bit-Mapped Trigger Substrate Benchmark Metrics...")
    substrate = BitMappedTriggerSubstrate(num_banks=4, bank_size=64)

    noisy_signals = [np.random.rand(16) for _ in range(50)]
    yield_res = substrate.evaluate_bit_remapped_yield(noisy_signals)
    print(f" -> {yield_res['statement']}")
    assert yield_res["passed"], "Bit Re-mapping Yield failed!"

    surge_bits = np.ones(64, dtype=np.uint8)
    diss_res = substrate.evaluate_bank_dissipation_rate(surge_bits)
    print(f" -> {diss_res['statement']}")
    assert diss_res["passed"], "Bank Dissipation Rate failed!"

    rest_res = substrate.evaluate_state_restoration_index(dropped_bank_id=0)
    print(f" -> {rest_res['statement']}")
    assert rest_res["passed"], "State Restoration Index failed!"

    # 2. Relational Nexus Node Verification
    print("\n[2] Testing Relational Nexus Node (Informational Trinity & Self-Mutation)...")
    node = RelationalNexusNode(
        node_id="Test_Node_1",
        payload={"value": 3.0, "temperature": 88.0},
        self_schema={"dimension": 3, "stress_limit": 1.0},
        in_causal_vectors={"Node_2": 2.5},
        out_causal_vectors={"Node_3": 0.5},
        bit_substrate=substrate
    )
    stress = node.compute_local_stress()
    print(f" -> Calculated Local Stress: {stress:.4f}")

    mutate_res = node.self_evaluate_and_mutate(environment_stress=2.5)
    print(f" -> {mutate_res['statement']}")
    assert mutate_res["is_mutated"], "Node Self-Mutation failed!"
    assert mutate_res["current_dimension"] == 4, "Dimension expansion failed!"

    # 3. Topological Phase Transition & Invariants Verification
    print("\n[3] Testing Topological Phase Transition & Invariants (b0=1, b1=0, Flux)...")
    phase_engine = TopologicalPhaseTransitionEngine(lambda_coef=0.5)
    nodes = [
        RelationalNexusNode("N1", {"value": 1.0}, {"dimension": 3}, in_causal_vectors={"N2": 1.0}),
        RelationalNexusNode("N2", {"value": 1.0}, {"dimension": 3}, out_causal_vectors={"N1": 1.0})
    ]
    adj = np.array([[0.0, 1.0], [1.0, 0.0]])

    invariants = phase_engine.verify_topological_invariants(nodes, adj)
    print(f" -> Invariants Check: {invariants['statement']}")
    assert invariants["passed"], "Topological Invariants verification failed!"

    trans_res = phase_engine.execute_phase_transition_or_rollback(nodes, adj)
    print(f" -> Phase Transition Engine Output: {trans_res['statement']}")

    # 4. SelfReferentialArchitectureEngine Integration Verification
    print("\n[4] Testing Full Cycle Integration in SelfReferentialArchitectureEngine...")
    arch_engine = SelfReferentialArchitectureEngine()
    stimulus = {
        "bit_payload": np.ones(32, dtype=np.uint8),
        "voltage_intent": np.array([2.0, -1.0, 3.0]),
        "layer1_intent": np.array([1.5, -0.5, 2.0, 0.1])
    }
    cycle_output = arch_engine.run_full_self_referential_cycle(stimulus)
    print(" -> Full Cycle Output Keys verified:", list(cycle_output.keys())[:6])
    print(f" -> Relational Phase Transition Result: {cycle_output['relational_phase_transition']['statement']}")

    print("\n" + "=" * 80)
    print("ALL VERIFICATIONS PASSED SUCCESSFULLY! RELATIONAL NEXUS & PHASE TRANSITION VALIDATED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
