"""
Unit tests for CausalStemBranchEngine
======================================
같음과 다름의 줄기와 가지 (Homological Stem & Branch Topology) 해부 엔진 테스트
"""

import pytest
from core.topology.causal_stem_branch_engine import (
    CausalStemBranchEngine,
    CausalTrajectoryGraph,
    CausalNode,
    CausalEdge,
    TrajectoryContext
)


def test_stem_branch_extraction_across_different_mediums():
    """
    서로 다른 매질 (파이썬 코드 vs 물리적 이산 회로)을 통해 구현된 두 인과 궤적 그래프를 대조하여,
    1:1 보존되는 같음의 줄기(Stem)와 매질 전용 가지(Branch)가 정확히 분리되는지 검증.
    """
    # Graph A: Python Code Medium
    graph_A = CausalTrajectoryGraph(
        graph_id="python_pipeline",
        context=TrajectoryContext(medium_type="python_code")
    )
    graph_A.add_node(CausalNode(id="in_val", role="input", abstract_operation="read_signal"))
    graph_A.add_node(CausalNode(id="log_print", role="state", abstract_operation="console_logger")) # Python specific branch
    graph_A.add_node(CausalNode(id="acc_val", role="transform", abstract_operation="accumulate"))
    graph_A.add_node(CausalNode(id="out_val", role="output", abstract_operation="emit_signal"))

    graph_A.add_edge(CausalEdge(source_id="in_val", target_id="log_print", mechanism_type="side_effect"))
    graph_A.add_edge(CausalEdge(source_id="in_val", target_id="acc_val", mechanism_type="direct_flow"))
    graph_A.add_edge(CausalEdge(source_id="acc_val", target_id="out_val", mechanism_type="direct_flow"))

    # Graph B: Physical Circuit Medium
    graph_B = CausalTrajectoryGraph(
        graph_id="physical_circuit",
        context=TrajectoryContext(medium_type="physical_circuit")
    )
    graph_B.add_node(CausalNode(id="sensor_pin", role="input", abstract_operation="read_signal"))
    graph_B.add_node(CausalNode(id="capacitor_charge", role="transform", abstract_operation="accumulate"))
    graph_B.add_node(CausalNode(id="led_heat_sink", role="state", abstract_operation="thermal_dissipation")) # Circuit specific branch
    graph_B.add_node(CausalNode(id="actuator_out", role="output", abstract_operation="emit_signal"))

    graph_B.add_edge(CausalEdge(source_id="sensor_pin", target_id="capacitor_charge", mechanism_type="direct_flow"))
    graph_B.add_edge(CausalEdge(source_id="capacitor_charge", target_id="actuator_out", mechanism_type="direct_flow"))
    graph_B.add_edge(CausalEdge(source_id="actuator_out", target_id="led_heat_sink", mechanism_type="side_effect"))

    engine = CausalStemBranchEngine()
    result = engine.parse_stem_and_branches(graph_A, graph_B)

    # 1. Stem 검증: 매질을 통과하여 보존된 3개의 노드 (input, accum, output) 사상
    stem = result.stem
    assert len(stem.common_subgraph_A_node_ids) == 3
    assert len(stem.common_subgraph_B_node_ids) == 3
    assert stem.node_mapping_A_to_B["in_val"] == "sensor_pin"
    assert stem.node_mapping_A_to_B["acc_val"] == "capacitor_charge"
    assert stem.node_mapping_A_to_B["out_val"] == "actuator_out"

    # 2. Branch 검증: 매질 특수성으로 격리된 가지
    assert len(result.branches_A) == 1
    assert "log_print" in result.branches_A[0].branch_node_ids
    assert result.branches_A[0].attached_stem_node_id == "in_val"

    assert len(result.branches_B) == 1
    assert "led_heat_sink" in result.branches_B[0].branch_node_ids
    assert result.branches_B[0].attached_stem_node_id == "actuator_out"

    # 3. Continuity 검증
    assert result.is_continuous is True
    assert len(result.discontinuity_reasons) == 0


def test_trajectory_discontinuity_detection():
    """
    줄기 궤적 내에 논리적 단절이 존재할 때 연속성 파기(Discontinuity)가 올바르게 감지되는지 검증.
    """
    graph_A = CausalTrajectoryGraph(graph_id="broken_flow_A")
    graph_A.add_node(CausalNode(id="in1", role="input", abstract_operation="read_signal"))
    graph_A.add_node(CausalNode(id="proc1", role="transform", abstract_operation="accumulate"))
    graph_A.add_node(CausalNode(id="out1", role="output", abstract_operation="emit_signal"))
    # No edge connecting proc1 to out1 -> Discontinuous!
    graph_A.add_edge(CausalEdge(source_id="in1", target_id="proc1", mechanism_type="direct_flow"))

    graph_B = CausalTrajectoryGraph(graph_id="broken_flow_B")
    graph_B.add_node(CausalNode(id="in2", role="input", abstract_operation="read_signal"))
    graph_B.add_node(CausalNode(id="proc2", role="transform", abstract_operation="accumulate"))
    graph_B.add_node(CausalNode(id="out2", role="output", abstract_operation="emit_signal"))
    graph_B.add_edge(CausalEdge(source_id="in2", target_id="proc2", mechanism_type="direct_flow"))

    engine = CausalStemBranchEngine()
    result = engine.parse_stem_and_branches(graph_A, graph_B)

    assert result.is_continuous is False
    assert len(result.discontinuity_reasons) > 0
    assert "Discontinuity detected" in result.discontinuity_reasons[0]
