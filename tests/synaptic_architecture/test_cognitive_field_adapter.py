import numpy as np
import pytest
from synaptic_architecture.cognitive_field_adapter import (
    CharacterStats,
    FieldParameters,
    CognitiveFieldAdapter,
    JobAttentionMask,
    ClassAdvancementPhaseTransition,
    CommanderAuraField
)
from synaptic_architecture.inverse_inference_engine import InverseInferenceEngine, EpistemicFact
from synaptic_architecture.cognitive_engine import ElysiaCognitiveEngine

def test_cognitive_field_adapter_precision():
    """
    고블린, 베테랑 인간 기사, 고대 드래곤의 스탯을 변환한 후 소수점 셋째 자리까지 기댓값과 일치하는지 정밀 검증합니다.
    """
    adapter = CognitiveFieldAdapter()

    # 1. 고블린 보병
    goblin_stats = CharacterStats(str_=8, dex=14, int_=5, con=8, wis=4, honor=0.0, infamy=10.0)
    goblin_params = adapter.transform(goblin_stats)

    assert goblin_params.attention_slots == 1
    assert goblin_params.lookahead_depth == 1
    # 수식: 1.0 * (1.0 + 0.02*4) * (1.0 + 0.005*8) = 1.0 * 1.08 * 1.04 = 1.1232 -> 반올림 1.123
    assert goblin_params.field_sigma == pytest.approx(1.123, abs=1e-3)
    # 반응 가속도 수식 정밀 계산: 1.0 + (0.04 * 14) * sqrt(1 + 5/25) = 1.0 + 0.56 * 1.095445 = 1.613
    assert goblin_params.volition_acceleration == pytest.approx(1.613, abs=1e-3)
    assert goblin_params.attractor_mass == pytest.approx(1.560, abs=1e-3)
    # 사회적 중력장: 0.02 * 0 - 0.01 * 10 = -0.1
    assert goblin_params.social_gravity == pytest.approx(-0.100, abs=1e-3)

    # 2. 베테랑 인간 기사
    knight_stats = CharacterStats(str_=35, dex=25, int_=20, con=40, wis=35, honor=50.0, infamy=5.0)
    knight_params = adapter.transform(knight_stats)

    assert knight_params.attention_slots == 3
    assert knight_params.lookahead_depth == 2
    # 수식: 1.0 * (1.0 + 0.02*35) * (1.0 + 0.005*40) = 1.0 * 1.70 * 1.20 = 2.040
    assert knight_params.field_sigma == pytest.approx(2.040, abs=1e-3)
    assert knight_params.volition_acceleration == pytest.approx(2.342, abs=1e-3)
    assert knight_params.attractor_mass == pytest.approx(3.550, abs=1e-3)
    # 사회적 중력장: 0.02 * 50 - 0.01 * 5 = 1.0 - 0.05 = 0.95
    assert knight_params.social_gravity == pytest.approx(0.950, abs=1e-3)

    # 3. 고대 드래곤
    dragon_stats = CharacterStats(str_=90, dex=20, int_=85, con=95, wis=90, honor=200.0, infamy=150.0)
    dragon_params = adapter.transform(dragon_stats)

    # 수식 정밀 연산값: int(1 + math.log2(1 + 85/8) + 90/30) = int(1 + 3.539 + 3.0) = 7
    assert dragon_params.attention_slots == 7
    # 수식 정밀 연산값: int(1 + 85/12 * (0.8 + 0.2 * 90/50)) = int(1 + 7.0833 * 1.16) = int(1 + 8.216) = 9
    assert dragon_params.lookahead_depth == 9
    # 수식: 1.0 * (1.0 + 0.02*90) * (1.0 + 0.005*95) = 1.0 * 2.80 * 1.475 = 4.130
    assert dragon_params.field_sigma == pytest.approx(4.130, abs=1e-3)
    assert dragon_params.volition_acceleration == pytest.approx(2.678, abs=1e-3)
    assert dragon_params.attractor_mass == pytest.approx(7.400, abs=1e-3)
    # 사회적 중력장: 0.02 * 200 - 0.01 * 150 = 4.0 - 1.5 = 2.5
    assert dragon_params.social_gravity == pytest.approx(2.500, abs=1e-3)

def test_job_attention_mask_bias():
    """
    탱커, 힐러, 암살자 직업 마스크가 인지 가치 필터 점수를 편향시키는지 검증합니다.
    """
    tank_mask = JobAttentionMask.create_tanker()
    healer_mask = JobAttentionMask.create_healer()
    assassin_mask = JobAttentionMask.create_assassin()

    # 힐러는 아군(Ally)에 극대화된 가중치(4.0)를 주는 반면 암살자는 낮게(0.2) 줌
    assert healer_mask.apply("Ally", 10.0) == 40.0
    assert assassin_mask.apply("Ally", 10.0) == 2.0

    # 탱커는 위협(Threat)에 높은 가중치(3.0)를 가짐
    assert tank_mask.apply("Threat", 10.0) == 30.0

def test_class_advancement_phase_transition():
    """
    성기사 및 그림자 군주 전직 시 위상적 전이가 일어나 새로운 보편우상 어트랙터가 이식되는지 검증합니다.
    """
    engine = ElysiaCognitiveEngine(resolution=128)
    transition = ClassAdvancementPhaseTransition("Novice")

    # Novice 상태에서는 "Sacred"나 "Void" 어트랙터가 존재하지 않음
    assert "Sacred" not in engine.field.attractors
    assert "Void" not in engine.field.attractors

    # 1. 성기사(Paladin) 전직
    report_paladin = transition.trigger_transition(engine, "Paladin")
    assert "Sacred" in engine.field.attractors
    assert "Sacred" in report_paladin
    assert engine.field.attractors["Sacred"]["mass"] == 60.0

    # 2. 그림자 군주(ShadowLord) 전직
    report_shadow = transition.trigger_transition(engine, "ShadowLord")
    assert "Void" in engine.field.attractors
    assert "Void" in report_shadow

def test_commander_aura_field_synchronization():
    """
    지휘관이 광역 의지장을 투사하여 부하들의 인지 슬롯을 100% 인과적으로 동기화하는지 검증합니다.
    """
    # 5명의 부하 병사 AI 생성 (슬롯이 꽉 찬 상태 시뮬레이션)
    soldiers = []
    for i in range(5):
        s = ElysiaCognitiveEngine(resolution=64)
        # 슬롯을 임의로 채움
        s.update_attention_and_bottleneck(np.uint64(i + 1), "General", base_intensity=1.0)
        soldiers.append(s)

    commander_aura = CommanderAuraField("commander_hero", aura_intensity=50.0)
    # 지휘관 명령 투사
    command_wave = np.uint64(0xABCDEFABCDEF)
    sync_count = commander_aura.project_will(command_wave, soldiers)

    # 지휘관의 권위 가중치(50.0)가 매우 높기 때문에, 모든 부하가 기존 인지 슬롯을 강제 축출하고 명령 수용해야 함
    assert sync_count == 5
    for soldier in soldiers:
        assert command_wave in soldier.attention_registry
        assert soldier.attention_registry[command_wave]["category"] == "CommanderCommand"

def test_inverse_inference_action_to_epistemic_fact():
    """
    물리적 행동(WFC Collapse 결과) 및 비외력 맥락을 역관측하여
    "영웅(Hero)/악당(Villain)"에 대한 고차원 의미론적 팩트를 역추론하고 인그램으로 각인하는 피드백 루프를 검증합니다.
    """
    engine = ElysiaCognitiveEngine(resolution=128)
    inverse_engine = InverseInferenceEngine(memory_controller=engine.memory_controller)

    # 초기 캐릭터 스탯 및 평판
    stats = CharacterStats(str_=10, dex=10, int_=10, con=10, wis=10, honor=0.0, infamy=0.0)

    # 1. 영웅(HERO) 행동 탐지: 경비병의 경례(Guard_Salute) + 외력 없음(external_force: 0.0)
    context_royal = {"location_context": "Royal_Gate", "external_force": 0.0}
    fact_hero = inverse_engine.observe_and_infer("player_hero", "Guard_Salute", context_royal, stats)

    assert fact_hero is not None
    assert fact_hero.fact_type == "HERO"
    assert stats.honor == 25.0  # 자율적 명예 보상 증가

    # Causal memory engram이 올바르게 각인되었는지 검증
    engrams = [info for info in engine.memory_controller.index.values() if info.get("cause_id") == "InverseInferenceEngine"]
    assert len(engrams) > 0
    assert engrams[-1]["data_blob"]["type"] == "EPISTEMIC_INVERSE_INFERENCE"
    assert engrams[-1]["data_blob"]["fact_type"] == "HERO"

    # 2. 악당(VILLAIN) 행동 탐지: 시민들의 사방 도망침(Citizens_Flee)
    context_square = {"location_context": "Central_Square", "external_force": 0.0}
    fact_villain = inverse_engine.observe_and_infer("player_hero", "Citizens_Flee", context_square, stats)

    assert fact_villain is not None
    assert fact_villain.fact_type == "VILLAIN"
    assert stats.infamy == 30.0  # 자율적 악명 증가

    # 악명에 의한 사회적 중력장 파동 재생성 및 다음 인지 변환에 피드백 반영
    adapter = CognitiveFieldAdapter()
    params = adapter.transform(stats)
    # 명예 25, 악명 30 => social_gravity = 0.02 * 25 - 0.01 * 30 = 0.5 - 0.3 = 0.2
    assert params.social_gravity == pytest.approx(0.2, abs=1e-3)
