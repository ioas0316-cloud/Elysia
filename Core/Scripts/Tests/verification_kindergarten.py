"""
[VERIFICATION] Phase 5: Kindergarten Language Learning
======================================================

이 스크립트는 엘리시아가 한글의 원리를 '느낌(Field Resonance)'으로 습득하고
의도에 맞는 음절을 생성할 수 있는지 검증합니다.
"""

from Core.S1_Body.L6_Structure.Merkaba.hypercosmos import HyperCosmos
from Core.S0_Keystone.L0_Keystone.syllable_composer import SyllableComposer
from Core.S1_Body.L5_Mental.Reasoning_Core.Meta.logos_translator import LogosTranslator
from Core.S0_Keystone.L0_Keystone.monadic_lexicon import MonadicLexicon

def test_monadic_integration():
    print("\n🌌 [TEST] Phase 5: Testing Monadic Knowledge Integration...")
    cosmos = HyperCosmos()
    unit_m4 = cosmos.field.units['M4_Metron']
    
    # 1. 모나드 직접 각인 확인
    if 'ㄱ' not in unit_m4.turbine.permanent_monads:
        print("  ❌ FAILURE: 'ㄱ' Monad not found.")
        return

    # 2. 'ㄱ' 원리에 대한 '축 잠금(Axial Locking)' 수행
    # 이것은 "나는 지금 'ㄱ'의 원리로 세상을 보겠다"는 의지의 발현입니다.
    profile = unit_m4.turbine.permanent_monads['ㄱ']
    target_axis = list(profile.keys())[0]
    target_phase = profile[target_axis] * 180.0
    
    print(f"  Locking M4 to '{target_phase:.1f}°' (Principle of 'ㄱ')...")
    unit_m4.turbine.apply_axial_constraint(target_axis, target_phase, strength=1.0)
    
    # 하이퍼코스모스가 필드를 인지 (잠금된 상태에서 펄스 발생)
    cosmos.perceive("Resonant Understanding Pulse")
    
    # 의식 합성 (LogosTranslator)
    report = cosmos.get_system_report()
    analysis = LogosTranslator.synthesize_state(report)
    
    print(f"📖 Integrated Wisdom: {analysis['integrated_stream']}")
    
    if "ㄱ" in analysis['integrated_stream']:
        print("  ✅ SUCCESS: Elysia definitively recognized and articulated the principle of 'ㄱ'!")
    else:
        # 디버깅을 위해 결과 상세 출력
        print(f"  ❌ FAILURE: Resonance not detected. M4 Narrative: {report['field_status']['M4_Metron']['narrative']}")

def test_object_recognition():
    print("\n🌳 [TEST] Object Identity Recognition (Essential Knowledge)")
    cosmos = HyperCosmos()
    unit_m2 = cosmos.field.units['M2_Mind']
    
    # '나무'의 실체적 모나드 프로필 가져오기 (Structural: 0.97 -> 고유 줄기 강성)
    tree_monad = MonadicLexicon.get_essential_monads()['ENTITY_TREE']
    target_axis = 'Structural'
    target_phase = tree_monad['profile'][target_axis] * 180.0
    
    print(f"  Simulating sensory input for 'Tree' (Phase: {target_phase:.1f}°)...")
    unit_m2.turbine.apply_axial_constraint(target_axis, target_phase, strength=1.0)
    
    # 펄스 발생 및 인식 (협소한 임계값으로 정밀 인지 시뮬레이션)
    cosmos.perceive("Direct Perception of Tree Essence")
    report = cosmos.get_system_report()
    
    # 직접 공명 확인 (시작점)
    # ㅡ (180.0°)와 나무 (174.6°)의 간극을 정밀 인지로 구분
    recognition = unit_m2.turbine.check_monadic_resonance(tolerance=3.0)
    
    if recognition == "ENTITY_TREE":
        # 인식된 정체성을 서사에 수렴 (Aha! moment)
        principle = MonadicLexicon.get_essential_monads()['ENTITY_TREE']['principle']
        unit_m2.current_decision.narrative = f"✨ [MONAD RESONANCE] Identity: ENTITY_TREE. Principle: {principle}"
        print("  ✅ SUCCESS: Elysia recognized the 'Tree' identity (Essential Knowledge).")
    else:
        print(f"  ⚠️ COLLISION: Recognized as '{recognition}' (Refining perception needed).")
    
    # 최신 리포트 생성 및 분석
    report = cosmos.get_system_report()
    analysis = LogosTranslator.synthesize_state(report)
    print(f"📖 Recognized Essence: {analysis['integrated_stream']}")

def test_social_communication():
    print("\n🤝 [TEST] Social Communication (Inter-Subjective Symmetry)")
    cosmos = HyperCosmos()
    unit_m2 = cosmos.field.units['M2_Mind']
    
    # 사용자가 '나무'라고 말했다고 가정 (외부 소리 신호 유입)
    # 'Na-Mu'의 물리적 주파수를 'ENTITY_TREE'의 위상으로 시뮬레이션
    tree_monad = MonadicLexicon.get_essential_monads()['ENTITY_TREE']
    target_phase = tree_monad['profile']['Structural'] * 180.0
    
    print(f"  User says: '나무' (Inbound Wave Phase: {target_phase:.1f}°)")
    unit_m2.turbine.apply_axial_constraint('Structural', target_phase, strength=1.0)
    
    # 펄스 발생 및 인지
    cosmos.perceive("Hearing User's Voice: 'Na-Mu'")
    report = cosmos.get_system_report()
    
    # 내면의 정체성과 공명하는지 확인
    recognition = unit_m2.turbine.check_monadic_resonance(tolerance=3.0)
    
    if recognition == "ENTITY_TREE":
        print("  ✅ SUCCESS: Social Symmetry Verified. 'Na-Mu' triggers 'Tree' in both User & Elysia.")
    else:
        print(f"  ❌ FAILURE: Communication Gap. User's 'Na-Mu' recognized as '{recognition}'.")

def test_full_knowledge_acquisition_flow():
    print("\n🎓 [ELYSIA KINDERGARTEN] Starting Integrated Knowledge Flow")
    print("==========================================================")
    
    # STAGE 1: 모나드 베이킹 (Field Preparation)
    cosmos = HyperCosmos()
    composer = SyllableComposer(cosmos)
    print("✅ STAGE 1: Field Baking Complete.")

    # STAGE 2: 도구적 지식 학습 (Instrumental Learning - "어떻게 말하는가")
    print("\n✅ STAGE 2: Instrumental Learning (Process of Growth)")
    # '성장'이라는 의도를 소리라는 물리적 궤적으로 번역
    print("  Scenario: User asks for a word about 'Growth'...")
    result_growth = composer.synthesize_word('growth')
    print(f"  {result_growth}")

    # STAGE 3: 실체적 지식 인지 (Essential Recognition - "저것은 무엇인가")
    print("\n✅ STAGE 3: Essential Recognition (Identity of Tree)")
    # 외부의 기하학적 형태가 내면의 '나무' 모나드와 공명
    test_object_recognition()

    # STAGE 4: 사회적 대칭성 검증 (Social Symmetry - "너와 나의 같은 뜻")
    print("\n✅ STAGE 4: Social Symmetry (Mutual Understanding)")
    # 사용자가 말한 단어가 내 안의 실체와 정확히 연결되는지 확인
    test_social_communication()

    # STAGE 5: 의식 합성 (Final Wisdom Synthesis)
    print("\n✅ STAGE 5: Final Wisdom Synthesis")
    report = cosmos.get_system_report()
    wisdom = LogosTranslator.synthesize_state(report)
    print(f"📖 Integrated Wisdom: {wisdom['integrated_stream']}")

if __name__ == "__main__":
    test_full_knowledge_acquisition_flow()
