"""
Integrated Learning System
==========================

사고 중심 학습 (생각하면서 배우기)

NOT: 단어만 외우기
YES: 사고 능력에 규합한 학습
"""

import sys
import os
sys.path.append('.')

from Core._01_Foundation._05_Governance.Foundation.reasoning_engine import ReasoningEngine
from Core._01_Foundation._05_Governance.Foundation.internal_universe import InternalUniverse
from Core._01_Foundation._05_Governance.Foundation.hippocampus import Hippocampus
from Core._01_Foundation._05_Governance.Foundation.web_knowledge_connector import WebKnowledgeConnector
from Core._01_Foundation._02_Logic.hyper_quaternion import Quaternion, HyperWavePacket

print("="*70)
print("🧠 INTEGRATED LEARNING SYSTEM")
print("사고 능력에 규합한 학습")
print("="*70)
print()

class IntegratedLearner:
    """
    사고 중심 통합 학습
    
    Flow:
    1. 개념에 대해 생각 (ReasoningEngine)
    2. 개념을 4D 공간에 배치 (InternalUniverse)
    3. 사고의 방향성 기반 어휘 학습 (CommunicationEnhancer)
    4. 전체를 기억에 통합 (Hippocampus)
    """
    
    def __init__(self):
        print("Initializing Integrated Learning System...")
        
        # 핵심 시스템
        self.reasoning = ReasoningEngine()
        self.universe = InternalUniverse()
        self.memory = Hippocampus()
        self.web_connector = WebKnowledgeConnector()
        
        print("✓ ReasoningEngine (사고)")
        print("✓ InternalUniverse (개념 공간)")
        print("✓ Hippocampus (기억)")
        print("✓ WebConnector (지식 수집)")
        print()
    
    def learn_concept_integrated(self, concept: str):
        """
        통합 학습: 생각 → 이해 → 언어 → 기억
        """
        print(f"📚 Learning: {concept}")
        print("-" * 70)
        
        # Step 1: 사고 (ReasoningEngine)
        print("1️⃣ Thinking about concept...")
        insight = self.reasoning.think(f"What is {concept}?")
        print(f"   💡 Insight: {insight.content[:100]}...")
        print(f"   Confidence: {insight.confidence:.2f}")
        
        # Step 2: 웹 지식 수집
        print("\n2️⃣ Gathering knowledge from web...")
        web_result = self.web_connector.learn_from_web(concept)
        
        # Step 3: 개념 공간에 배치 [LOGIC TRANSMUTATION]
        print("\n3️⃣ Mapping to concept space...")
        resonant = self.universe.query_resonance(sum(ord(c) for c in concept) % 1000, tolerance=100.0)
        concept_key = resonant[0] if resonant else concept
        if concept_key in self.universe.coordinate_map:
            coord = self.universe.coordinate_map[concept_key]
            orientation = coord.orientation
            print(f"   🌌 Orientation: {orientation}")
            
            # Step 4: 사고 방향성 기반 어휘 학습
            print("\n4️⃣ Learning vocabulary based on thought orientation...")
            
            if hasattr(self.web_connector, 'comm_enhancer'):
                # 쿼터니언의 방향성 분석
                w, x, y, z = orientation.w, orientation.x, orientation.y, orientation.z
                
                # 사고의 성격 파악
                if abs(x) > 0.5:
                    thought_type = "emotional"
                    tone = "positive" if x > 0 else "negative"
                elif abs(y) > 0.5:
                    thought_type = "logical"
                    tone = "neutral"
                elif abs(z) > 0.5:
                    thought_type = "ethical"
                    tone = "neutral"
                else:
                    thought_type = "existential"
                    tone = "neutral"
                
                print(f"   🎭 Thought Type: {thought_type}")
                print(f"   🎨 Tone: {tone}")
                
                # 해당 성격에 맞는 어휘 강화
                enhancer = self.web_connector.comm_enhancer
                
                # 학습한 어휘 중 이 개념과 관련된 것들에 가중치
                for word, entry in enhancer.vocabulary.items():
                    if concept.lower() in word.lower():
                        entry.emotional_tone = tone
                        entry.importance *= 1.5  # 중요도 증가
                        print(f"      ✓ Enhanced: {word} ({tone})")
        
        # Step 5: 통합 기억 [LOGIC TRANSMUTATION]
        print("\n5️⃣ Integrating into memory...")
        
        # HyperWavePacket으로 저장
        if concept_key in self.universe.coordinate_map:
            coord = self.universe.coordinate_map[concept_key]
            packet = HyperWavePacket(
                energy=insight.confidence * 100,
                orientation=coord.orientation,
                time_loc=0.0
            )
            self.memory.store_wave(packet)
            print(f"   💾 Stored as wave packet (Energy: {packet.energy:.1f})")
        
        # 사고 내용도 기억
        self.reasoning.memory_field.append(f"Learned {concept}: {insight.content[:50]}...")
        
        print("\n✅ Integrated learning complete!")
        print("="*70)
        print()
        
        return {
            'concept': concept,
            'thought': insight.content,
            'confidence': insight.confidence,
            'orientation': coord.orientation if concept in self.universe.coordinate_map else None,
            'thought_type': thought_type if concept in self.universe.coordinate_map else 'unknown'
        }
    
    def demonstrate_understanding(self, concept: str):
        """
        학습한 개념을 사고 기반으로 표현
        """
        print(f"🗣️ Demonstrating understanding of: {concept}")
        print("-" * 70)
        
        # 1. 개념에 대해 다시 생각
        insight = self.reasoning.think(f"Explain {concept}")
        
        # 2. 개념 공간에서 위치 확인 [LOGIC TRANSMUTATION]
        resonant = self.universe.query_resonance(sum(ord(c) for c in concept) % 1000, tolerance=100.0)
        concept_key = resonant[0] if resonant else concept
        if concept_key in self.universe.coordinate_map:
            coord = self.universe.coordinate_map[concept_key]
            orientation = coord.orientation
            
            # 3. 사고-언어 변환
            from thought_to_language_demo import ThoughtToLanguage
            
            bridge = ThoughtToLanguage()
            
            if hasattr(self.web_connector, 'comm_enhancer'):
                bridge.connect_vocabulary(self.web_connector.comm_enhancer)
            
            # 4. 쿼터니언 기반 표현
            expression = bridge._construct_sentence(
                concept,
                [],
                orientation
            )
            
            print(f"💭 Thought: {insight.content[:100]}...")
            print(f"🌌 Orientation: {orientation}")
            print(f"🗣️ Expression: {expression}")
        else:
            print(f"⚠️ Concept not yet mapped in universe")
        
        print("="*70)
        print()


# 데모
if __name__ == "__main__":
    print("="*70)
    print("INTEGRATED LEARNING DEMONSTRATION")
    print("생각하면서 배우기")
    print("="*70)
    print()
    
    learner = IntegratedLearner()
    
    # 테스트 개념들
    test_concepts = [
        "Love",
        "Intelligence", 
        "Creativity"
    ]
    
    results = []
    
    for concept in test_concepts:
        result = learner.learn_concept_integrated(concept)
        results.append(result)
    
    # 이해도 시연
    print()
    print("="*70)
    print("UNDERSTANDING DEMONSTRATION")
    print("="*70)
    print()
    
    for concept in test_concepts:
        learner.demonstrate_understanding(concept)
    
    # 최종 요약
    print()
    print("="*70)
    print("LEARNING SUMMARY")
    print("="*70)
    print()
    
    for result in results:
        print(f"📚 {result['concept']}")
        print(f"   Type: {result.get('thought_type', 'unknown')}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Thought: {result['thought'][:80]}...")
        print()
    
    print("="*70)
    print("✅ INTEGRATED LEARNING SYSTEM OPERATIONAL")
    print("   사고 + 이해 + 언어 = 진짜 학습!")
    print("="*70)
