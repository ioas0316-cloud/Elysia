"""
Unit tests for Introspective Causal Diagnostics Engine
======================================================
Tests:
1. Structural rejection of incompatible/unbound topological noise (Zero Binding).
2. Residual stress accumulation and detection under constraint conflicts.
3. Introspective causal map backtracking and root-cause proof generation.
4. Layer-wide boundary isomorphic equilibrium convergence and residual stress relaxation.
"""

import pytest
from core.consciousness.introspective_causal_diagnostics import (
    IntrospectiveCausalDiagnosticsEngine,
    CausalNode,
    CausalConstraint,
    BindingState,
    ProtocolType,
    DiagnosticProof
)


def test_metabolic_structural_rejection():
    """1. 검증: 규격에 들어맞지 않는 소음 데이터의 구조적 거부반응 (Zero Binding)"""
    engine = IntrospectiveCausalDiagnosticsEngine()

    incompatible_input = {
        "signature": "UNBOUND_RANDOM_NOISE_VECTOR_12345",
        "protocol_type": "passport"
    }

    result = engine.process_metabolic_ingestion(incompatible_input)

    assert result.is_metabolic_bound is False
    assert result.status == BindingState.REJECTED_INCOMPATIBLE
    assert "Metabolic Binding Failed" in result.root_cause_explanation
    assert "Structurally rejected" in result.root_cause_explanation


def test_residual_stress_accumulation_and_introspective_diagnosis():
    """2 & 3. 검증: 제약 조건 충돌 시 예외 구문이 아닌 잔차 응력 집적 및 인과 지도 자가 진단"""
    engine = IntrospectiveCausalDiagnosticsEngine(yield_threshold=0.5)

    node_a = CausalNode(id="logic_core", name="Logic Core", domain="code", capacity=1.0)
    node_b = CausalNode(id="mem_bus", name="Memory Bus", domain="memory", capacity=1.0)
    engine.add_node(node_a)
    engine.add_node(node_b)

    constraint = CausalConstraint(
        id="c_logic_mem",
        source_node="logic_core",
        target_node="mem_bus",
        protocol_type=ProtocolType.PASSPORT,
        stiffness=2.0,
        tolerance=0.05
    )
    engine.add_constraint(constraint)

    valid_stimulus = {
        "signature": "CAUSAL_TOPOLOGY_V1",
        "protocol_type": "passport",
        "target_node": "logic_core",
        "intensity": 3.0
    }

    result = engine.process_metabolic_ingestion(valid_stimulus)

    assert result.is_metabolic_bound is True
    assert result.initial_stress > 0.5
    assert result.conflict_constraint_pair == ("logic_core", "mem_bus")
    assert "Introspective Diagnosis" in result.root_cause_explanation
    assert "logic_core" in result.affected_nodes


def test_boundary_equilibrium_relaxation():
    """4. 검증: 위상적 이완을 통한 필연적 변형 평형 상태 도출"""
    engine = IntrospectiveCausalDiagnosticsEngine(yield_threshold=1.0, max_introspect_steps=10)

    node_a = CausalNode(id="node1", name="Node 1", domain="protocol", capacity=1.0)
    node_b = CausalNode(id="node2", name="Node 2", domain="hardware", capacity=1.0)
    engine.add_node(node_a)
    engine.add_node(node_b)

    c = CausalConstraint(
        id="c12",
        source_node="node1",
        target_node="node2",
        protocol_type=ProtocolType.CURRENCY,
        stiffness=1.0,
        tolerance=0.1
    )
    engine.add_constraint(c)

    stimulus = {
        "signature": "EQUILIBRIUM_TOKEN",
        "protocol_type": "currency",
        "target_node": "node1",
        "intensity": 2.0
    }

    result = engine.process_metabolic_ingestion(stimulus)

    assert result.is_metabolic_bound is True
    assert result.status == BindingState.EQUILIBRIUM
    assert result.residual_stress <= 1.0
    assert len(result.boundary_absorption_log) > 0
    assert result.predicted_equilibrium_state["is_valid_equilibrium"] is True
    assert result.predicted_equilibrium_state["isomorphic_grid_aligned"] is True


def test_unperturbed_equilibrium_on_zero_tension():
    """5. 검증: 충돌이 없는 조화 자극 시 미섭동 평형 상태 유지"""
    engine = IntrospectiveCausalDiagnosticsEngine(yield_threshold=1.0)

    node_a = CausalNode(id="node_safe", name="Safe Node", domain="code", capacity=10.0)
    engine.add_node(node_a)

    stimulus = {
        "signature": "SEMANTIC_INVARIANT_GRID",
        "protocol_type": "language",
        "target_node": "node_safe",
        "intensity": 0.01
    }

    result = engine.process_metabolic_ingestion(stimulus)

    assert result.is_metabolic_bound is True
    assert result.status == BindingState.BOUND
    assert result.conflict_constraint_pair is None
