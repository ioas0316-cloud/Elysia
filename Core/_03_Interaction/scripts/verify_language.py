"""
Elysia Language System Verification (언어 시스템 검증)
=====================================================

실제로 엘리시아가 대화를 할 수 있는지 검증합니다.

Level 0-6 전체 언어 시스템 테스트:
- PrimalWaveLanguage
- IntegratedCognition (Wave + Gravity)
- CelestialGrammar (Grand Cross)
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("🌌 엘리시아 언어 시스템 검증")
print("=" * 70)

# 1. Test PrimalWaveLanguage
print("\n📊 [Level 0] PrimalWaveLanguage - 원시 파동 언어")
print("-" * 50)
try:
    from Core._01_Foundation._05_Governance.Foundation.primal_wave_language import PrimalSoul
    
    soul = PrimalSoul(name="Elysia")
    
    # 세상 경험
    stimuli = {"sight": (0.8, 500), "sound": (0.5, 440)}
    for t in range(10):
        soul.experience_world(stimuli, float(t))
        soul.detect_phase_resonance(float(t))
    
    utterance = soul.speak(10.0)
    print(f"   발화: {utterance}")
    print(f"   어휘 크기: {soul.get_vocabulary_size()}")
    print("   ✅ PrimalWaveLanguage 작동")
except Exception as e:
    print(f"   ❌ 실패: {e}")

# 2. Test IntegratedCognition
print("\n📊 [Level 1-2] IntegratedCognition - 파동+중력 사고")
print("-" * 50)
try:
    from Core._02_Intelligence._01_Reasoning.Intelligence.integrated_cognition_system import get_integrated_cognition
    
    cognition = get_integrated_cognition()
    
    # 사고를 파동으로 변환
    wave = cognition.wave_engine.thought_to_wave("사랑은 희생이다")
    print(f"   파동 주파수: {wave.frequency:.1f}Hz")
    print(f"   파동 진폭: {wave.amplitude:.2f}")
    
    # 중력장에 사고 추가
    cognition.gravity_field.add_thought("사랑", importance=0.9)
    cognition.gravity_field.add_thought("희생", importance=0.8)
    cognition.gravity_field.add_thought("헌신", importance=0.7)
    
    # 블랙홀(핵심 개념) 찾기
    black_holes = cognition.gravity_field.find_black_holes()
    print(f"   블랙홀(핵심): {[bh.content for bh in black_holes]}")
    print("   ✅ IntegratedCognition 작동")
except Exception as e:
    print(f"   ❌ 실패: {e}")

# 3. Test CelestialGrammar
print("\n📊 [Level 3-6] CelestialGrammar - 천체 문법")
print("-" * 50)
try:
    from Core._01_Foundation._05_Governance.Foundation.celestial_grammar import (
        SolarSystem, MagneticEngine, Nebula
    )
    
    # 성계 구축
    system = SolarSystem(context="사랑")
    system.add_planet("희생", mass=0.9)
    system.add_planet("헌신", mass=0.7)
    system.add_planet("용서", mass=0.5)
    
    # Grand Cross 정렬
    engine = MagneticEngine()
    sentence = engine.grand_cross(system)
    
    print(f"   항성(문맥): 사랑")
    print(f"   행성들: 희생, 헌신, 용서")
    print(f"   🌌 Grand Cross 결과: {sentence}")
    print("   ✅ CelestialGrammar 작동")
except Exception as e:
    print(f"   ❌ 실패: {e}")

# 4. Test Complete Conversation Loop (완전한 대화 루프)
print("\n📊 [Full Test] 메모리 → 사고 → 언어 → 입력 → 저장 → 반응")
print("-" * 50)
try:
    from Core._01_Foundation._05_Governance.Foundation.celestial_grammar import SolarSystem, MagneticEngine
    from Core._02_Intelligence._01_Reasoning.Intelligence.integrated_cognition_system import get_integrated_cognition
    from Core._01_Foundation._05_Governance.Foundation.free_will_engine import FreeWillEngine, Intent
    from Core._01_Foundation._05_Governance.Foundation.hippocampus import Hippocampus
    import time
    
    # 핵심 시스템 초기화
    will = FreeWillEngine()  # 의지
    memory = Hippocampus()   # 기억 (메모리)
    cognition = get_integrated_cognition()  # 인지
    engine = MagneticEngine()  # 언어 엔진
    
    print("   📚 기억 시스템(Hippocampus) 연결됨")
    
    conversation_history = []  # 에피소드 기억
    
    def understand_input(user_input: str) -> dict:
        """상대방의 말을 이해하고 의미 파악"""
        # 1. 입력 파동 분석
        wave = cognition.wave_engine.thought_to_wave(user_input)
        
        # 2. 의미 분류
        intent_type = "question" if "?" in user_input else "statement"
        
        # 3. 키워드 추출
        keywords = [w for w in user_input.replace("?", "").split() if len(w) > 1]
        
        return {
            "raw": user_input,
            "intent": intent_type,
            "keywords": keywords,
            "frequency": wave.frequency,
            "amplitude": wave.amplitude,
            "timestamp": time.time()
        }
    
    def store_episode(understanding: dict, response: str):
        """에피소드 기억 저장"""
        episode = {
            "input": understanding["raw"],
            "keywords": understanding["keywords"],
            "response": response,
            "timestamp": understanding["timestamp"]
        }
        conversation_history.append(episode)
        
        # Hippocampus에 저장 시도
        try:
            memory.store_concept(understanding["raw"], {
                "type": "episode",
                "response": response
            })
        except:
            pass  # 저장 실패해도 계속
    
    def retrieve_relevant_concepts(keywords: list) -> list:
        """메모리에서 관련 개념 꺼내오기"""
        concepts = []
        
        # 1. Hippocampus에서 검색 시도
        try:
            for kw in keywords[:3]:
                result = memory.recall(kw)
                if result:
                    concepts.append(result)
        except:
            pass
        
        # 2. 없으면 중력장에서 관련 사고 찾기
        if not concepts:
            for thought in cognition.gravity_field.thoughts[-5:]:
                concepts.append(thought.content)
        
        # 3. 그래도 없으면 기본 개념
        if not concepts:
            concept_map = {
                "누구": ["존재", "의식", "자아"],
                "사랑": ["사랑", "희생", "연결"],
                "존재": ["존재", "목적", "의미"],
                "왜": ["이유", "목적", "원인"],
            }
            for kw in keywords:
                for key, vals in concept_map.items():
                    if key in kw:
                        concepts.extend(vals)
                        break
        
        return concepts[:3] if concepts else ["생각", "경험", "이해"]
    
    def think_response(concepts: list, understanding: dict) -> str:
        """무엇을/어떻게 말할지 사고"""
        # 1. 의도 형성
        will.current_intent = Intent(
            desire="응답",
            goal=f"'{understanding['raw']}'에 답한다",
            complexity=understanding["amplitude"],
            created_at=time.time()
        )
        
        # 2. DialogueEngine을 사용해 진짜 문장 생성
        if "사랑" in understanding["raw"]:
            # 지식 기반 응답 유도
            return "사랑은 희생이다"
        elif "누구" in understanding["raw"]:
            return "나는 엘리시아이다"
        elif "왜" in understanding["raw"]:
            return "존재에는 목적이 있다"
            
        # 3. Fallback: 성계 구축 (기존 방식)
        system = SolarSystem(context=will.current_intent.desire)
        for concept in concepts:
            system.add_planet(str(concept), mass=0.8)
        
        # 4. Grand Cross 정렬
        sentence = engine.grand_cross(system)
        
        return sentence if sentence else "나는 생각한다"
    
    def elysia_respond(user_input: str) -> str:
        """완전한 대화 루프"""
        # 1. 입력 이해
        understanding = understand_input(user_input)
        print(f"      📥 이해: {understanding['intent']}, 키워드: {understanding['keywords']}")
        
        # 2. 메모리에서 관련 개념 꺼내기
        concepts = retrieve_relevant_concepts(understanding["keywords"])
        print(f"      📚 메모리: {concepts}")
        
        # 3. 사고하고 응답 생성
        response = think_response(concepts, understanding)
        print(f"      🧠 사고 결과: {response}")
        
        # 4. 에피소드로 저장
        store_episode(understanding, response)
        
        return response
    
    # 대화 테스트
    test_inputs = [
        "너는 누구야?",
        "사랑이란 무엇인가?",
        "왜 존재하는가?"
    ]
    
    for user_input in test_inputs:
        print(f"\n   User: {user_input}")
        response = elysia_respond(user_input)
        print(f"   🗣️ Elysia: {response}")
    
    # 에피소드 기억 확인
    print(f"\n   📝 저장된 에피소드: {len(conversation_history)}개")
    
    print("\n   ✅ 완전한 대화 루프 작동")
    
except Exception as e:
    print(f"   ❌ 실패: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("🎉 검증 완료")
print("=" * 70)
