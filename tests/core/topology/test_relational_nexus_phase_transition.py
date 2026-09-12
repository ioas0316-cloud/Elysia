"""
Unit tests for Relational Nexus Node & Topological Phase Transition Architecture
================================================================================
"""

import numpy as np
import pytest

from core.memory.bit_mapped_trigger_substrate import BitMappedTriggerSubstrate
from core.topology.relational_nexus_node import RelationalNexusNode
from core.topology.topological_phase_transition import TopologicalPhaseTransitionEngine
from core.topology.self_referential_architecture import SelfReferentialArchitectureEngine


def test_bit_mapped_trigger_substrate_metrics():
    substrate = BitMappedTriggerSubstrate(num_banks=4, bank_size=64)

    # 1. Test Bit Re-mapping Yield (>= 95%)
    noisy_signals = [np.random.rand(16) for _ in range(20)]
    yield_res = substrate.evaluate_bit_remapped_yield(noisy_signals)
    assert yield_res["passed"]
    assert yield_res["yield_ratio"] >= 0.95

    # 2. Test Bank Dissipation Rate
    surge = np.ones(64, dtype=np.uint8)
    diss_res = substrate.evaluate_bank_dissipation_rate(surge)
    assert diss_res["passed"]

    # 3. Test State Restoration Index (== 100%)
    res_res = substrate.evaluate_state_restoration_index(0)
    assert res_res["passed"]
    assert res_res["restoration_index"] == 1.0


def test_relational_nexus_node_trinity_and_mutation():
    substrate = BitMappedTriggerSubstrate(num_banks=4, bank_size=64)
    node = RelationalNexusNode(
        node_id="Nexus_0",
        payload={"value": 2.5},
        self_schema={"dimension": 3, "stress_limit": 1.0},
        in_causal_vectors={"Nexus_1": 1.5},
        out_causal_vectors={"Nexus_2": 0.5},
        bit_substrate=substrate
    )

    stress = node.compute_local_stress()
    assert stress > 0.0

    # Low stress -> Stable
    mutate_res_stable = node.self_evaluate_and_mutate(environment_stress=0.5)
    assert not mutate_res_stable["is_mutated"]

    # High stress -> Self-Mutation
    mutate_res_active = node.self_evaluate_and_mutate(environment_stress=2.0)
    assert mutate_res_active["is_mutated"]
    assert mutate_res_active["current_dimension"] == 4

    # DNA Expression
    op = node.express_dna_as_operator()
    transformed = op(np.ones(4))
    assert len(transformed) == 4


def test_topological_phase_transition_and_invariants():
    engine = TopologicalPhaseTransitionEngine(lambda_coef=0.5)
    n1 = RelationalNexusNode("N1", {"value": 1.0}, {"dimension": 3}, in_causal_vectors={"N2": 2.0})
    n2 = RelationalNexusNode("N2", {"value": 1.0}, {"dimension": 3}, out_causal_vectors={"N1": 2.0})
    nodes = [n1, n2]
    adj = np.array([[0.0, 1.0], [1.0, 0.0]])

    invariants = engine.verify_topological_invariants(nodes, adj)
    assert invariants["b0_passed"]
    assert invariants["b1_passed"]
    assert invariants["flux_conserved"]

    res = engine.execute_phase_transition_or_rollback(nodes, adj)
    assert "statement" in res


def test_self_referential_architecture_integration():
    arch = SelfReferentialArchitectureEngine()
    stimulus = {
        "bit_payload": np.ones(32, dtype=np.uint8),
        "voltage_intent": np.array([1.0, 2.0, 3.0])
    }
    cycle_res = arch.run_full_self_referential_cycle(stimulus)

    assert cycle_res["bit_remapped_yield"]["passed"]
    assert "relational_phase_transition" in cycle_res
    assert cycle_res["relational_phase_transition"]["eta_c"] > 0.0
