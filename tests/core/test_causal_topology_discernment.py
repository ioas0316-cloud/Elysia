"""
Verification Test for Elysia Causal Topology Foundation
======================================================
고립된 수치 점, 하드코딩 사전, f-string 텍스트 템플릿의 거짓을 완전히 배제하고,
위상 대조(Comparison) -> 마찰 인지(Tension) -> 관계 내재화(Replication)를 통해
시스템이 외부 세계의 인과적 구조를 스스로 판별하고 학습해 내는지를 물리적으로 검증합니다.
"""

import numpy as np
import pytest

from core.topology.causal_structure import (
    CausalNumber,
    CausalSymbol,
    InformationTopology,
    TopologyLink
)
from core.topology.topological_comparer import TopologicalComparer
from core.topology.relational_replicator import RelationalStructureReplicator
from core.topology.causal_discernment_engine import CausalDiscernmentEngine


def test_causal_number_and_symbol_topology():
    """
    1. 숫자(Number)와 기호(Symbol)가 단순 스칼라/문자열이 아닌,
       양·순서·차이·색채 및 교차 차원 관계망을 가진 위상 객체임을 검증.
    """
    num1 = CausalNumber(id="num_1", value=1.0, sequence_index=1, magnitude=1.0, gradient_tension=0.1, chromatic_vector=np.array([0.1, 0.9, 0.1]))
    num2 = CausalNumber(id="num_2", value=2.0, sequence_index=2, magnitude=2.0, gradient_tension=0.2, chromatic_vector=np.array([0.2, 0.8, 0.2]))
    num10 = CausalNumber(id="num_10", value=10.0, sequence_index=10, magnitude=10.0, gradient_tension=0.8, chromatic_vector=np.array([0.9, 0.1, 0.8]))

    # 인과적 위상 차이 마찰력 검증
    disp_1_2 = num1.calculate_disparity(num2)
    disp_1_10 = num1.calculate_disparity(num10)

    # 순서와 양의 거리가 먼 10과의 마찰력이 2와의 마찰력보다 훨씬 커야 함
    assert disp_1_2 < disp_1_10

    # 기호 4대 교차 차원 서명 검증
    sym_apple = CausalSymbol(
        id="sym_apple",
        name="Apple",
        material_vector=np.array([0.8, 0.2, 0.1, 0.9], dtype=np.float32),
        causal_trajectory=["seed", "tree", "blossom", "fruit"],
        logical_category="Fruit",
        relational_links=[
            TopologyLink("sym_apple", "sym_tree", "causal", strength=0.9, tension=0.1),
            TopologyLink("sym_apple", "sym_red", "material", strength=0.8, tension=0.2)
        ],
        intrinsic_tension=0.05
    )

    sig = sym_apple.get_cross_dimensional_signature()
    assert len(sig) == 5
    assert sig[1] == 4.0  # 인과적 계보 깊이 4


def test_topological_comparison_same_and_different():
    """
    2. 위상 대조기(TopologicalComparer)가 맹목적 매칭 없이
       자기 구조와 세계 구조의 위상적 같음(Coherence)과 다름(Disparity)을 정확히 산출하는지 검증.
    """
    self_topo = InformationTopology("Self")
    self_topo.add_number(CausalNumber(id="n1", value=5.0, sequence_index=5, magnitude=5.0, gradient_tension=0.1, chromatic_vector=np.array([0.5, 0.5, 0.1])))

    # 1) 동일한 위상 구조 대조
    world_same = InformationTopology("WorldSame")
    world_same.add_number(CausalNumber(id="n1", value=5.0, sequence_index=5, magnitude=5.0, gradient_tension=0.1, chromatic_vector=np.array([0.5, 0.5, 0.1])))

    comparer = TopologicalComparer(tolerance=0.15)
    res_same = comparer.compare(self_topo, world_same)

    assert res_same.isomorphism_ratio == 1.0
    assert res_same.disparity_tension == 0.0
    assert len(res_same.coherent_nodes) == 1

    # 2) 완전히 다른 위상 구조 대조
    world_diff = InformationTopology("WorldDiff")
    world_diff.add_number(CausalNumber(id="n1", value=50.0, sequence_index=50, magnitude=50.0, gradient_tension=0.9, chromatic_vector=np.array([0.9, 0.1, 0.9])))

    res_diff = comparer.compare(self_topo, world_diff)
    assert res_diff.isomorphism_ratio < 0.5
    assert res_diff.disparity_tension > 0.3
    assert len(res_diff.disparate_nodes) == 1


def test_relational_replication_true_learning():
    """
    3. 참된 인지/학습 검증:
       시스템이 모르는 외부 세계 자극을 만났을 때,
       차이 마찰(Tension)을 인지하여 그 관계 구조를 자기 내부로 복제·이식(Internalization)하고,
       내재화 이후 동일 자극에 대해 마찰이 해소되고 동형성(Isomorphism)이 1.0으로 수렴함을 증명.
    """
    engine = CausalDiscernmentEngine(tension_threshold=0.10)

    # 모르는 미지의 외부 세계 자극 구축 ("사과" 및 "나무"의 인과 관계망)
    world_stimulus = InformationTopology("WorldAppleStimulus")
    world_stimulus.add_symbol(CausalSymbol(
        id="sym_apple",
        name="Apple",
        material_vector=np.array([0.9, 0.1, 0.8], dtype=np.float32),
        causal_trajectory=["seed", "tree", "fruit"],
        logical_category="Fruit",
        relational_links=[
            TopologyLink("sym_apple", "sym_tree", "causal", strength=0.95, tension=0.05)
        ]
    ))
    world_stimulus.add_symbol(CausalSymbol(
        id="sym_tree",
        name="Tree",
        material_vector=np.array([0.2, 0.9, 0.3], dtype=np.float32),
        causal_trajectory=["seed", "sprout", "tree"],
        logical_category="Plant"
    ))

    # [1차 시도] 초기의 자아는 이 관계망을 전혀 모름 (미지 자극)
    trace_1 = engine.perceive_and_discern(world_stimulus)

    assert trace_1.initial_isomorphism == 0.0           # 동형성 0 (완전 미지)
    assert trace_1.initial_disparity_tension > 0.4       # 높은 차이 긴장 마찰 발생
    assert trace_1.was_internalized is True              # 마찰로 인해 내재화(학습) 유발됨
    assert trace_1.post_isomorphism > 0.8                # 내재화 후 동형성 대폭 상승

    # 자아 위상 다양체 지문의 실질적 변형(Delta) 검증
    assert trace_1.topological_fingerprint_delta["density"] > 0.0

    # [2차 시도] 동일한 외부 자극을 다시 마주함 (학습 성과 확인)
    trace_2 = engine.perceive_and_discern(world_stimulus)

    assert trace_2.initial_isomorphism == 1.0            # 이제 이미 알고 있는 자아 구조이므로 동형성 100%
    assert trace_2.initial_disparity_tension == 0.0      # 마찰 긴장도 0 (완전 통전)
    assert trace_2.was_internalized is False             # 이미 내재화되었으므로 추가 변형 없음
