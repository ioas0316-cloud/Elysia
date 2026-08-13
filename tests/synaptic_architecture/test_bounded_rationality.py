import numpy as np
import pytest
from synaptic_architecture.cognitive_engine import (
    ElysiaCognitiveEngine,
    GOBLIN_PROFILE,
    DRAGON_PROFILE,
    CognitiveBiasProfile
)

def test_goblin_vs_dragon_profile_spec():
    """
    고블린과 드래곤의 인지 규격(슬롯, 기억 수명, 예측 깊이, 편향)이
    서로 다르게 완벽하게 파라미터화되어 있는지 확인합니다.
    """
    assert GOBLIN_PROFILE.attention_slots == 1
    assert GOBLIN_PROFILE.lookahead_depth == 1
    assert GOBLIN_PROFILE.impulsivity == 0.95
    assert GOBLIN_PROFILE.risk_sensitivity == 0.9

    assert DRAGON_PROFILE.attention_slots == 10
    assert DRAGON_PROFILE.lookahead_depth == 6
    assert DRAGON_PROFILE.ego_pride == 0.95
    assert DRAGON_PROFILE.impulsivity == 0.05

def test_attention_bottleneck_goblin_eviction_and_deflection():
    """
    고블린 프로필 하에서 주의력 병목(Attention Bottleneck)이 100% 인과적으로
    작동하는지(주사위 없이 가중치 기반 축출 및 무시) 검증합니다.
    """
    engine = ElysiaCognitiveEngine(resolution=128, profile=GOBLIN_PROFILE)

    stimulus_healer = np.uint64(0xAAAAAAA000000001)
    stimulus_tank = np.uint64(0xBBBBBBB000000002)
    stimulus_noise = np.uint64(0xCCCCCCC000000003)

    # 1. 고블린은 슬롯이 1개이므로 최초 자극(Healer)을 수용하면 가득 참
    status_1 = engine.update_attention_and_bottleneck(stimulus_healer, "Healer", base_intensity=1.0)
    assert status_1 == "ATTENTION_ACCEPTED"
    assert len(engine.attention_registry) == 1
    assert stimulus_healer in engine.attention_registry

    # Healer의 가중치 계산
    weight_healer = engine.calculate_attention_weight(stimulus_healer, "Healer", 1.0)

    # 2. 아주 강한 도발 자극(Taunt) 주입 -> Healer보다 가중치가 크므로 Healer를 뇌에서 "축출"하고 Tank를 수용
    status_2 = engine.update_attention_and_bottleneck(stimulus_tank, "Taunt", base_intensity=5.0)
    assert status_2 == "ATTENTION_EVICTION"
    assert len(engine.attention_registry) == 1
    assert stimulus_tank in engine.attention_registry
    assert stimulus_healer not in engine.attention_registry  # 완벽한 망각 및 인과적 축출!

    # 3. 아주 약하고 무의미한 소음 자극(Noise) 주입 -> 현재 Tank 가중치보다 낮으므로 인과적으로 무시(Deflected)
    status_3 = engine.update_attention_and_bottleneck(stimulus_noise, "Noise", base_intensity=0.1)
    assert status_3 == "ATTENTION_DEFLECTED"
    assert len(engine.attention_registry) == 1
    assert stimulus_tank in engine.attention_registry
    assert stimulus_noise not in engine.attention_registry

def test_100_percent_determinism_argmax():
    """
    WFC collapse 가 어떠한 확률적 무작위 요소도 없이 100% 결정론적(Deterministic argmax)으로
    동작함을 입증하기 위해, 동일한 조건에서 여러 번 수렴시켰을 때 완전히 동일한 결과를 도출하는지 테스트합니다.
    """
    engine = ElysiaCognitiveEngine(resolution=128, profile=GOBLIN_PROFILE)
    dna_healer = engine.build_fractal_dna("Healer_DNA", np.uint64(0x1111111111111111))
    dna_tank = engine.build_fractal_dna("Tank_DNA", np.uint64(0x2222222222222222))

    stimulus = np.uint64(0x1111111100000000)

    results = []
    for _ in range(10):
        # 복사를 통해 매 시도마다 필드에 영구 흔적이 쌓이지 않도록 엔진 상태를 복제/동일하게 유지
        temp_engine = ElysiaCognitiveEngine(resolution=128, profile=GOBLIN_PROFILE)
        temp_dna_h = temp_engine.build_fractal_dna("Healer_DNA", np.uint64(0x1111111111111111))
        temp_dna_t = temp_engine.build_fractal_dna("Tank_DNA", np.uint64(0x2222222222222222))

        res = temp_engine.solve_wfc_collapse(stimulus, [temp_dna_h, temp_dna_t])
        results.append(res)

    # 10번의 결과가 모두 완벽하게 한치 오차도 없이 동일한지 비교
    winner_categories = [r["collapsed_dna"]["category"] for r in results]
    scores = [r["resonance_score"] for r in results]
    positions = [tuple(r["collapse_position"]) for r in results]

    # 중복을 제거한 세트의 길이가 1이어야 함 (즉, 모든 값이 동일)
    assert len(set(winner_categories)) == 1
    assert len(set(scores)) == 1
    assert len(set(positions)) == 1

def test_lookahead_depth_stress_contraction():
    """
    고블린과 드래곤의 예측 탐색 깊이 차이를 검증하고,
    긴장(Tension)이 높을 때 탐색 범위가 인과적으로 수축(Stress Contraction)하는지 테스트합니다.
    """
    engine_dragon = ElysiaCognitiveEngine(resolution=128, profile=DRAGON_PROFILE)
    engine_goblin = ElysiaCognitiveEngine(resolution=128, profile=GOBLIN_PROFILE)

    dna_h = engine_dragon.build_fractal_dna("Healer", np.uint64(0x9999999999999999))
    dna_t = engine_dragon.build_fractal_dna("Tank", np.uint64(0x8888888888888888))

    # 1. 드래곤은 평온한 상태에서 깊은 예측 탐색 루프(Lookahead 6)를 가집니다.
    # 메타 이력을 확인하여 드래곤의 6단계 예측 탐색을 확인합니다.
    engine_dragon.solve_wfc_collapse(np.uint64(0x9999999900000000), [dna_h, dna_t])
    meta_dragon = engine_dragon.get_meta_reflection()
    equilibrium_events_dragon = [m for m in meta_dragon if m["action"] == "RESONANCE_EQUILIBRIUM"]
    assert len(equilibrium_events_dragon) > 0
    # 드래곤은 깊이가 6이므로 축소되지 않았을 때 lookahead 단계가 5 혹은 6이어야 함 (미세 긴장 수축 반영)
    desc = equilibrium_events_dragon[-1]["description"]
    assert "Lookahead 단계: 6/6" in desc or "Lookahead 단계: 5/6" in desc

    # 2. 고블린은 애초에 탐색 깊이가 1단계이므로 수축과 무관하게 1단계로만 동작합니다.
    dna_g1 = engine_goblin.build_fractal_dna("Healer", np.uint64(0x9999999999999999))
    dna_g2 = engine_goblin.build_fractal_dna("Tank", np.uint64(0x8888888888888888))
    engine_goblin.solve_wfc_collapse(np.uint64(0x9999999900000000), [dna_g1, dna_g2])
    meta_goblin = engine_goblin.get_meta_reflection()
    equilibrium_events_goblin = [m for m in meta_goblin if m["action"] == "RESONANCE_EQUILIBRIUM"]
    assert len(equilibrium_events_goblin) > 0
    assert "Lookahead 단계: 1" in equilibrium_events_goblin[-1]["description"]

def test_fish_like_streamlining_propulsion():
    """
    물고기 유영 메타포:
    정합성이 높은 자극이 유입될 때, 저항/마찰을 추진력(Propulsion)으로 치환하여
    로컬 여백(Yeobaek)과 전도력(Conductance)을 정비하는지 확인합니다.
    """
    engine = ElysiaCognitiveEngine(resolution=128, profile=DRAGON_PROFILE)

    # 0.5rad 관점 회전을 준 상태
    engine.set_perspective("Strategic Calm", 0.5)

    # 1. 관점과 정합성이 높은 자극 wave 생성
    # user_header_vector를 [cos(0.5), sin(0.5), 0.5] 에 가깝게 수동 세팅하여 완벽 매칭 유도
    header = np.array([np.cos(0.5), np.sin(0.5), 0.5], dtype=np.float32)
    header /= np.linalg.norm(header)

    dna = engine.build_fractal_dna("Ally", np.uint64(0x5555555555555555))
    pos = dna["cell_position"]

    # 기존 여백과 전도율 기록
    old_margin = engine.field.coordination_margin[pos[0], pos[1]]
    old_conductance = engine.field.conductance[pos[0], pos[1]]

    # WFC 붕괴 수행
    engine.solve_wfc_collapse(
        stimulus_wave=np.uint64(0x5555555555555555),
        candidate_dnas=[dna],
        user_header_vector=header
    )

    # 메타 로그 확인 및 추진력에 의한 필드 가소성 변화 검증
    meta = engine.get_meta_reflection()
    propulsion_events = [m for m in meta if m["action"] == "STREAMLINED_PROPULSION"]

    assert len(propulsion_events) > 0
    assert "물고기 유영 활성화" in propulsion_events[0]["description"]

    # 여백과 전도율이 증가했는지 검증 (유선형 추진력이 필드에 반영됨)
    new_margin = engine.field.coordination_margin[pos[0], pos[1]]
    new_conductance = engine.field.conductance[pos[0], pos[1]]

    assert new_margin > old_margin
    assert new_conductance > old_conductance
