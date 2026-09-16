"""
Unit Tests: Self-Molding Intelligence Engine
============================================
자기 관조, 위상 결함 진단, Stem 보존 검증 및 자가 성형 단위 테스트.
"""

import pytest
from core.topology.self_molding_engine import SelfMoldingEngine


def test_self_introspection():
    engine = SelfMoldingEngine()
    code = """
def update_locomotion(pos, speed):
    new_pos = pos + speed
    return new_pos
"""
    graph = engine.introspect_code(code)
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    assert any("INVARIANT_OP_ADD" in n.invariant_signature for n in graph.nodes.values())


def test_topological_anomaly_diagnosis():
    engine = SelfMoldingEngine()
    # whisker 센서 변수를 읽었으나 아무 곳에도 전달되지 않는 불완전 코드
    incomplete_code = """
def step(pos, speed, whisker_contact):
    whisker_val = whisker_contact
    new_pos = pos + speed
    return new_pos
"""
    graph = engine.introspect_code(incomplete_code)
    anomalies = engine.diagnose_topology(graph)

    # whisker_val이 생성되었으나 후속 연결이 없는 데드엔드 결함 탐지 확인
    assert len(anomalies) > 0
    assert any(a["type"] in ("DANGLING_CAUSAL_DEAD_END", "UNCOUPLED_SENSORY_INPUT") for a in anomalies)


def test_homologically_proven_self_molding():
    engine = SelfMoldingEngine()

    orig_code = """
val = 10
res = val * 2
return res
"""
    # 곱셈 줄기를 그대로 유지하면서 가속도 덧셈 기능이 추가된 진화 코드
    evolved_code = """
val = 10
res = val * 2
final_res = res + 5
return final_res
"""
    evolved_src, proven, metrics = engine.evolve_code(
        original_code=orig_code,
        evolved_code_proposal=evolved_code,
        target_invariant_signature="INVARIANT_OP_ADD"
    )

    assert proven is True
    assert metrics["is_proven"] is True
    assert metrics["has_target_feature"] is True
    assert metrics["preserved_stem_nodes"] >= 1
    assert evolved_src == evolved_code


def test_invalid_evolution_rejection():
    engine = SelfMoldingEngine()

    orig_code = """
val = 10
res = val * 2
return res
"""
    # 핵심 곱셈 연산 줄기가 파괴된 가짜/버그 코드
    broken_code = """
val = 10
res = val - 999
return res
"""
    evolved_src, proven, metrics = engine.evolve_code(
        original_code=orig_code,
        evolved_code_proposal=broken_code,
        target_invariant_signature="INVARIANT_OP_MULTIPLY" # 곱셈이 없으므로 거부되어야 함
    )

    # Stem 불일치 또는 타겟 부재로 안전하게 거부(Rejection)
    assert proven is False
    assert evolved_src == orig_code
