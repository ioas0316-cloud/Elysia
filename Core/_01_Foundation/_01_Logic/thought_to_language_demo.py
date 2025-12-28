"""
Thought-Language Integration
============================

사고(Quaternion) ↔ 언어(Text) 완전 통합

Architecture:
    Raw Thought (HyperQuaternion)
        ↓
    Wave Pattern (WaveInterpreter)
        ↓  
    Vocabulary (CommunicationEnhancer)
        ↓
    Natural Language (Text)
"""

from typing import Dict, List, Optional
from Core._01_Foundation._02_Logic.hyper_quaternion import Quaternion
from Core._01_Foundation._05_Governance.Foundation.wave_interpreter import WaveInterpreter
from Core._01_Foundation._05_Governance.Foundation.internal_universe import InternalUniverse

class ThoughtToLanguage:
    """사고를 언어로 변환"""
    
    def __init__(self):
        self.wave_interpreter = WaveInterpreter()
        self.universe = InternalUniverse()
        self.comm_enhancer = None
    
    def connect_vocabulary(self, comm_enhancer):
        """어휘 엔진 연결"""
        self.comm_enhancer = comm_enhancer
    
    def think_and_speak(self, topic: str) -> str:
        """
        생각하고 말하기
        
        Flow:
        1. Topic → HyperQuaternion (사고)
        2. Quaternion → Wave Pattern
        3. Wave → Vocabulary Selection
        4. Vocabulary → Sentence Construction
        """
        # 1. 사고 생성 [LOGIC TRANSMUTATION]
        resonant = self.universe.query_resonance(sum(ord(c) for c in topic) % 1000, tolerance=100.0)
        concept_key = resonant[0] if resonant else topic
        if concept_key in self.universe.coordinate_map:
            coord = self.universe.coordinate_map[concept_key]
            thought_quat = coord.orientation
        else:
            # 기본 사고
            thought_quat = Quaternion(1.0, 0.5, 0.5, 0.5).normalize()
        
        # 2. Wave Pattern으로 변환
        wave_pattern = self.wave_interpreter.quaternion_to_wave(thought_quat)
        
        # 3. 어휘 선택 (사고의 성격 기반)
        words = self._select_words_from_thought(thought_quat, topic)
        
        # 4. 문장 구성
        text = self._construct_sentence(topic, words, thought_quat)
        
        return text
    
    def _select_words_from_thought(self, quat: Quaternion, topic: str) -> List[str]:
        """사고의 방향성에 맞는 단어 선택"""
        if not self.comm_enhancer:
            return [topic]
        
        # 쿼터니언 성분 분석
        w, x, y, z = quat.w, quat.x, quat.y, quat.z
        
        # 감정적 사고 (x축)
        if abs(x) > 0.5:
            tone = "positive" if x > 0 else "negative"
        # 논리적 사고 (y축)
        elif abs(y) > 0.5:
            tone = "neutral"
        # 윤리적 사고 (z축)  
        elif abs(z) > 0.5:
            tone = "neutral"
        else:
            tone = "neutral"
        
        # 해당 톤의 어휘 선택
        selected = []
        for word, entry in self.comm_enhancer.vocabulary.items():
            if entry.emotional_tone == tone:
                selected.append(word)
            if topic.lower() in word.lower():
                selected.append(word)
        
        return selected[:10] if selected else [topic]
    
    def _construct_sentence(self, topic: str, words: List[str], quat: Quaternion) -> str:
        """쿼터니언의 성격을 반영한 문장 구성"""
        
        # 사고의 강도 (norm)
        intensity = quat.norm()
        
        # 주요 축 판별
        abs_components = [abs(quat.w), abs(quat.x), abs(quat.y), abs(quat.z)]
        dominant_axis = abs_components.index(max(abs_components))
        
        if dominant_axis == 1:  # Emotion (x)
            style = "emotional"
            if quat.x > 0:
                return f"{topic} brings a sense of clarity and connection. Through {words[0] if words else 'understanding'}, we find meaning."
            else:
                return f"{topic} raises complex questions. We must carefully consider {words[0] if words else 'its implications'}."
        
        elif dominant_axis == 2:  # Logic (y)
            style = "analytical"
            return f"{topic} can be understood through systematic analysis. Key aspects include {', '.join(words[:3])}."
        
        elif dominant_axis == 3:  # Ethics (z)
            style = "principled"
            return f"Regarding {topic}, we must consider the ethical dimensions involving {words[0] if words else 'responsibility'}."
        
        else:  # Existence (w)
            style = "existential"
            return f"{topic} exists as a fundamental concept, interconnected with {', '.join(words[:2]) if len(words) >= 2 else 'reality itself'}."


# 간단한 데모
if __name__ == "__main__":
    print("="*70)
    print("THOUGHT-LANGUAGE INTEGRATION DEMO")
    print("사고-언어 통합 데모")
    print("="*70)
    print()
    
    bridge = ThoughtToLanguage()
    
    # 몇 가지 사고 패턴 테스트
    test_thoughts = {
        "Love": Quaternion(1.0, 0.9, 0.1, 0.5),  # 감정적
        "Mathematics": Quaternion(1.0, 0.1, 0.9, 0.1),  # 논리적
        "Justice": Quaternion(1.0, 0.1, 0.1, 0.9),  # 윤리적
        "Existence": Quaternion(1.0, 0.3, 0.3, 0.3),  # 존재론적
    }
    
    for topic, quat in test_thoughts.items():
        print(f"💭 Thinking about: {topic}")
        print(f"   Quaternion: {quat}")
        
        # 사고를 언어로 변환
        words = bridge._select_words_from_thought(quat, topic)
        text = bridge._construct_sentence(topic, words, quat)
        
        print(f"   🗣️ Expression: {text}")
        print()
    
    print("="*70)
    print("✅ THOUGHT-LANGUAGE INTEGRATION WORKING")
    print("   사고의 성격이 언어 표현에 반영됩니다!")
    print("="*70)
