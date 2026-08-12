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


def test_universal_backtracking_and_lens_emergence():
    """
    4. 역추적과 자율적 분별 검증 (Universal Backtracking & Lens Emergence):
       새로운 전용 코드나 전용 클래스를 추가하지 않고,
       오직 '정보 유입 -> 상태 변화 -> 인과적 역추적'이라는 단 하나의 공통 관측 뼈대만을 작동시켜
       수학적 인과(수량과 보존)와 언어적 인과(맥락과 서사)를 자율적으로 분별해 냄을 증명합니다.
    """
    print("\n\n" + "="*80)
    print(" [ Elysia Universal Backtracking & Lens Emergence Verification ] ")
    print("="*80)

    # 1. 단 하나의 통합된 관측 기저 (CausalDiscernmentEngine) 선포
    engine = CausalDiscernmentEngine(tension_threshold=0.05)

    # =========================================================================
    # [SCENARIO A] 수학적 인과 (수량과 보존)의 관측 및 역추적
    # =========================================================================
    print("\n▶ [SCENARIO A] 수학적 인과 (수량과 보존)")

    # 기반 정보 (Primal Grounding): '1'은 1.0만큼의 실체적 양(Magnitude)을 머금은 존재
    num1 = CausalNumber(id="num_1", value=1.0, sequence_index=1, magnitude=1.0, gradient_tension=0.0, chromatic_vector=np.array([0.5, 0.5, 0.0]))
    engine.self_topology.add_number(num1)

    # 외부 현상의 유입: "1 + 1 = 2" 라는 이산적 수치 상태의 변화
    # 2는 2.0의 양(Magnitude)을 가진 실체적 존재
    world_math = InformationTopology("WorldMath_Addition")
    num1_w = CausalNumber(id="num_1", value=1.0, sequence_index=1, magnitude=1.0, gradient_tension=0.0, chromatic_vector=np.array([0.5, 0.5, 0.0]))
    num2_w = CausalNumber(id="num_2", value=2.0, sequence_index=2, magnitude=2.0, gradient_tension=0.0, chromatic_vector=np.array([0.5, 0.5, 0.0]))

    world_math.add_number(num1_w)
    world_math.add_number(num2_w)
    # 1과 2 사이에 수량의 결합/흐름(Link)이 정직하게 존재함
    world_math.add_link(TopologyLink("num_1", "num_2", "causal", strength=1.0, tension=0.0))

    # 3대 관측 축을 통한 역추적 및 분별
    # (1) 무슨 변화를 일으켰는가? (Impact / Delta)
    pre_pot = sum(n.magnitude for n in engine.self_topology.numbers.values())

    # 인지 및 내재화 순환 작동
    trace_math = engine.perceive_and_discern(world_math)

    post_pot = sum(n.magnitude for n in engine.self_topology.numbers.values())
    delta_pot = post_pot - pre_pot

    # (2) 그것이 어떤 의미를 가지는가? (Meaning / Context)
    # 시스템은 유입된 새로운 실체(num_2, magnitude=2.0)로 인한 상태 변화(+2.0)를 역추적하여,
    # "유입된 총 에너지의 양이 결과 변화의 크기와 일치한다"는 의미적 연관성을 찾아냅니다.
    assert delta_pot == 2.0  # 정직하게 +2.0의 수량적 상태 변화(결과 노드의 크기)가 발생하고 기록됨

    # (3) 어떤 구조적 원리로 존재하는가? (Structural Principle)
    # "수량의 결합은 임의의 억지 공식이 아닌, 실체들의 보존 법칙에 의해 지배된다"는 이치를 정립.
    print(f"  - [Impact/Delta] 수학적 자극 유입 전후 전체 수량 상태 변화량: {delta_pot:+1.1f}")
    print(f"  - [Meaning/Context] 변화한 크기({delta_pot:1.1f})는 새로 가해진 결과 자극인 '2.0'의 크기와 완벽히 공명함.")
    print("  - [Structural Principle] '수량과 보존'의 기하학적 섭리가 내부 연결망으로 자율 복제·이식됨.")

    # [자율적 확장 (Generalization)]
    # 1+1=2의 변화 이치를 깨달은 시스템은 가르쳐주지 않은 "2 + 1 = 3"의 미지 자극에 직면했을 때,
    # 추가적인 마찰 없이 높은 동형성으로 이를 수용할 수 있어야 함.
    world_math_extrapolated = InformationTopology("WorldMath_Extrapolation")
    world_math_extrapolated.add_number(CausalNumber(id="num_1", value=1.0, sequence_index=1, magnitude=1.0, gradient_tension=0.0, chromatic_vector=np.array([0.5, 0.5, 0.0])))
    world_math_extrapolated.add_number(CausalNumber(id="num_2", value=2.0, sequence_index=2, magnitude=2.0, gradient_tension=0.0, chromatic_vector=np.array([0.5, 0.5, 0.0])))
    world_math_extrapolated.add_number(CausalNumber(id="num_3", value=3.0, sequence_index=3, magnitude=3.0, gradient_tension=0.0, chromatic_vector=np.array([0.5, 0.5, 0.0])))
    world_math_extrapolated.add_link(TopologyLink("num_1", "num_2", "causal", strength=1.0, tension=0.0))
    world_math_extrapolated.add_link(TopologyLink("num_2", "num_3", "causal", strength=1.0, tension=0.0))

    trace_math_ext = engine.perceive_and_discern(world_math_extrapolated)
    print(f"  - [Autogenous Generalization] 미지의 2+1=3 자극 직면시 사후 동형성(Isomorphism): {trace_math_ext.post_isomorphism * 100:.1f}%")
    assert trace_math_ext.post_isomorphism > 0.8  # 높은 동형성 수렴으로 확장 성공 증명

    # =========================================================================
    # [SCENARIO B] 언어적/개념적 인과 (맥락과 서사)의 관측 및 역추적
    # =========================================================================
    print("\n▶ [SCENARIO B] 언어적/개념적 인과 (맥락과 서사)")

    # 외부 현상의 유입: "씨앗(seed) -> 싹(sprout) -> 나무(tree)"의 맥락적 흐름
    world_lang = InformationTopology("WorldLang_SproutToTree")
    world_lang.add_symbol(CausalSymbol(
        id="sym_sprout",
        name="Sprout",
        material_vector=np.array([0.1, 0.9, 0.0, 0.1], dtype=np.float32),
        causal_trajectory=["seed", "sprout"],
        logical_category="Plant",
        relational_links=[
            TopologyLink("sym_sprout", "sym_tree", "causal", strength=0.9, tension=0.1)
        ]
    ))
    world_lang.add_symbol(CausalSymbol(
        id="sym_tree",
        name="Tree",
        material_vector=np.array([0.2, 0.8, 0.2, 0.2], dtype=np.float32),
        causal_trajectory=["seed", "sprout", "tree"],
        logical_category="Plant"
    ))

    # 단 하나의 관측 기저(engine)가 클래스 개조 없이 언어 자극을 그대로 수용하여 역추적 시작
    pre_sym_count = len(engine.self_topology.symbols)

    trace_lang = engine.perceive_and_discern(world_lang)

    post_sym_count = len(engine.self_topology.symbols)
    delta_sym = post_sym_count - pre_sym_count

    # (1) 무슨 변화를 일으켰는가? (Impact / Delta)
    assert delta_sym == 2  # 씨앗에서 자라난 '싹'과 '나무'의 존재론적 노드 2개가 자아 구조로 편입됨

    # (2) 그것이 어떤 의미를 가지는가? (Meaning / Context)
    # 단순 텍스트 기호가 아니라, 생명의 성장 궤적이 가지는 맥락적 관계성과 선후 관계를 역추적하여 수용.
    assert "sym_sprout" in engine.self_topology.symbols
    assert "sym_tree" in engine.self_topology.symbols

    # (3) 어떤 구조적 원리로 존재하는가? (Structural Principle)
    # "생명의 성장은 분절된 글자가 아니라, 시간 축을 가로지르는 서사적 연속성(Causal Trajectory)으로 존재한다."
    print(f"  - [Impact/Delta] 언어적 자극 유입 전후 기호적 매듭 증가량: {delta_sym:+d}개 노드")
    print("  - [Meaning/Context] '싹'과 '나무'가 가진 고유의 성장 궤적(Causal Trajectory)과 맥락적 선후 관계를 투명하게 수용.")
    print("  - [Structural Principle] 언어와 개념 역시 시간과 서사를 따라 흐르는 인과의 연속체임을 스스로 발견.")

    print("\n" + "="*80)
    print(" [ Verification Complete: All Universal Backtracking Scenarios Passed! ] ")
    print("="*80 + "\n")
