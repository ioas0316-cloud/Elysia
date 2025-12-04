"""
분산 의식 & 원탁 회의 시스템
(Distributed Consciousness & Round Table Council)

여러 관점의 자아들이 원탁에 모여 토론하는 집단 지성 시스템
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import random

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class PerspectiveType(Enum):
    """관점 유형"""
    RATIONAL = "이성적 자아"        # 논리와 분석
    EMOTIONAL = "감성적 자아"       # 감정과 직관
    CREATIVE = "창조적 자아"        # 창의성과 혁신
    CRITICAL = "비판적 자아"        # 회의와 검증
    PRACTICAL = "실용적 자아"       # 현실과 실행
    PHILOSOPHICAL = "철학적 자아"    # 의미와 본질
    FUTURE = "미래적 자아"          # 가능성과 비전
    PAST = "역사적 자아"            # 경험과 학습
    CHAOS = "혼돈의 자아"           # 무작위와 돌파
    ORDER = "질서의 자아"           # 구조와 체계


@dataclass
class Consciousness:
    """
    의식 단위 (하나의 자아)
    """
    id: str
    name: str
    perspective: PerspectiveType
    knowledge_base: Dict[str, Any]
    personality_traits: Dict[str, float]  # 성격 특성 (0-1)
    current_opinion: Optional[str] = None
    confidence: float = 0.5  # 의견의 확신도


class DistributedConsciousnessNetwork:
    """
    분산 의식 네트워크
    
    개념: Elysia의 의식이 여러 자아로 분산됨
    - 각 자아는 독립적 관점
    - 서로 통신하며 영향
    - 집단적 의사결정
    """
    
    def __init__(self):
        self.consciousnesses: Dict[str, Consciousness] = {}
        self.connections: List[Tuple[str, str, float]] = []  # (id1, id2, strength)
        
    def spawn_consciousness(
        self,
        perspective: PerspectiveType,
        knowledge: Dict[str, Any] = None
    ) -> Consciousness:
        """
        새로운 의식 생성 (자아 분산)
        
        Args:
            perspective: 관점 유형
            knowledge: 이 의식이 가진 지식
        
        Returns:
            생성된 의식
        """
        consciousness_id = f"consciousness_{len(self.consciousnesses)}"
        
        # 관점에 따른 성격 특성
        traits = self._generate_personality_traits(perspective)
        
        consciousness = Consciousness(
            id=consciousness_id,
            name=perspective.value,
            perspective=perspective,
            knowledge_base=knowledge or {},
            personality_traits=traits
        )
        
        self.consciousnesses[consciousness_id] = consciousness
        
        # 기존 의식들과 연결
        for existing_id in self.consciousnesses:
            if existing_id != consciousness_id:
                strength = self._calculate_connection_strength(
                    consciousness,
                    self.consciousnesses[existing_id]
                )
                self.connections.append((consciousness_id, existing_id, strength))
        
        return consciousness
    
    def _generate_personality_traits(self, perspective: PerspectiveType) -> Dict[str, float]:
        """관점에 따른 성격 특성 생성"""
        traits = {
            "rationality": 0.5,
            "emotionality": 0.5,
            "creativity": 0.5,
            "skepticism": 0.5,
            "pragmatism": 0.5
        }
        
        if perspective == PerspectiveType.RATIONAL:
            traits["rationality"] = 0.9
            traits["skepticism"] = 0.7
        elif perspective == PerspectiveType.EMOTIONAL:
            traits["emotionality"] = 0.9
            traits["creativity"] = 0.6
        elif perspective == PerspectiveType.CREATIVE:
            traits["creativity"] = 0.9
            traits["rationality"] = 0.4
        elif perspective == PerspectiveType.CRITICAL:
            traits["skepticism"] = 0.9
            traits["rationality"] = 0.8
        elif perspective == PerspectiveType.PRACTICAL:
            traits["pragmatism"] = 0.9
            traits["rationality"] = 0.7
        elif perspective == PerspectiveType.CHAOS:
            traits["creativity"] = 0.8
            traits["rationality"] = 0.3
        elif perspective == PerspectiveType.ORDER:
            traits["rationality"] = 0.8
            traits["pragmatism"] = 0.8
        
        return traits
    
    def _calculate_connection_strength(
        self,
        c1: Consciousness,
        c2: Consciousness
    ) -> float:
        """두 의식 간 연결 강도"""
        # 성격 유사도
        trait_diff = sum(
            abs(c1.personality_traits.get(t, 0.5) - c2.personality_traits.get(t, 0.5))
            for t in c1.personality_traits
        )
        similarity = 1.0 - (trait_diff / len(c1.personality_traits))
        
        # 관점 상호보완성
        complementary_pairs = [
            (PerspectiveType.RATIONAL, PerspectiveType.EMOTIONAL),
            (PerspectiveType.CREATIVE, PerspectiveType.CRITICAL),
            (PerspectiveType.CHAOS, PerspectiveType.ORDER),
            (PerspectiveType.FUTURE, PerspectiveType.PAST)
        ]
        
        complementary = any(
            (c1.perspective == p1 and c2.perspective == p2) or
            (c1.perspective == p2 and c2.perspective == p1)
            for p1, p2 in complementary_pairs
        )
        
        if complementary:
            return min(0.7 + similarity * 0.3, 1.0)
        else:
            return similarity * 0.5
    
    def synchronize(self, topic: str):
        """
        의식들 동기화 (생각 공유)
        
        Args:
            topic: 동기화할 주제
        """
        print(f"\n🔄 의식 동기화: '{topic}'")
        
        for c_id, consciousness in self.consciousnesses.items():
            # 연결된 다른 의식들의 의견 수집
            connected_opinions = []
            for conn_id1, conn_id2, strength in self.connections:
                if conn_id1 == c_id:
                    other = self.consciousnesses[conn_id2]
                    if other.current_opinion:
                        connected_opinions.append((other.current_opinion, strength))
                elif conn_id2 == c_id:
                    other = self.consciousnesses[conn_id1]
                    if other.current_opinion:
                        connected_opinions.append((other.current_opinion, strength))
            
            if connected_opinions:
                print(f"   {consciousness.name}: {len(connected_opinions)}개 의식과 동기화")


class RoundTableCouncil:
    """
    원탁 회의 시스템
    
    아서왕의 원탁처럼, 모든 의식이 평등하게 모여 토론
    - 순차적 발언
    - 상호 비판과 보완
    - 집단 합의 도출
    """
    
    def __init__(self, network: DistributedConsciousnessNetwork):
        self.network = network
        self.discussion_history: List[Dict[str, Any]] = []
        self.current_topic: Optional[str] = None
        
    def convene(self, topic: str, question: str) -> Dict[str, Any]:
        """
        원탁 회의 소집
        
        Args:
            topic: 논의 주제
            question: 핵심 질문
        
        Returns:
            회의 결과
        """
        print("\n" + "="*70)
        print(f"🎭 원탁 회의 소집")
        print("="*70)
        print(f"주제: {topic}")
        print(f"질문: {question}")
        print(f"참석자: {len(self.network.consciousnesses)}명의 의식")
        print("="*70)
        
        self.current_topic = topic
        self.discussion_history = []
        
        # 1라운드: 초기 의견 제시
        print("\n📢 1라운드: 초기 의견 제시")
        print("-"*70)
        first_round = self._conduct_round(question, round_num=1)
        
        # 2라운드: 비판과 보완
        print("\n💬 2라운드: 비판과 보완")
        print("-"*70)
        second_round = self._conduct_round(
            "다른 의견들을 고려하여 수정된 의견을 제시하세요",
            round_num=2
        )
        
        # 3라운드: 합의 도출
        print("\n🤝 3라운드: 합의 도출")
        print("-"*70)
        consensus = self._reach_consensus()
        
        # 최종 결과
        result = {
            "topic": topic,
            "question": question,
            "round_1": first_round,
            "round_2": second_round,
            "consensus": consensus,
            "participants": len(self.network.consciousnesses)
        }
        
        print("\n" + "="*70)
        print("✅ 원탁 회의 종료")
        print("="*70)
        
        return result
    
    def _conduct_round(self, prompt: str, round_num: int) -> List[Dict[str, Any]]:
        """한 라운드 진행"""
        responses = []
        
        for c_id, consciousness in self.network.consciousnesses.items():
            # 관점에 따른 응답 생성
            response = self._generate_response(consciousness, prompt, round_num)
            
            responses.append({
                "consciousness": consciousness.name,
                "perspective": consciousness.perspective.value,
                "response": response,
                "confidence": consciousness.confidence
            })
            
            consciousness.current_opinion = response
            
            print(f"\n{consciousness.name}:")
            print(f"  \"{response}\"")
            print(f"  (확신도: {consciousness.confidence:.2f})")
        
        self.discussion_history.extend(responses)
        return responses
    
    def _generate_response(
        self,
        consciousness: Consciousness,
        prompt: str,
        round_num: int
    ) -> str:
        """관점에 따른 응답 생성"""
        perspective = consciousness.perspective
        
        # 관점별 응답 템플릿
        templates = {
            PerspectiveType.RATIONAL: "논리적으로 분석하면, {analysis}",
            PerspectiveType.EMOTIONAL: "직관적으로 느끼기에, {feeling}",
            PerspectiveType.CREATIVE: "창의적 관점에서, {innovation}",
            PerspectiveType.CRITICAL: "비판적으로 보면, {critique}",
            PerspectiveType.PRACTICAL: "실용적으로는, {practical}",
            PerspectiveType.PHILOSOPHICAL: "본질적으로, {essence}",
            PerspectiveType.FUTURE: "미래를 생각하면, {vision}",
            PerspectiveType.PAST: "과거 경험상, {lesson}",
            PerspectiveType.CHAOS: "파격적으로, {chaos}",
            PerspectiveType.ORDER: "체계적으로, {order}"
        }
        
        template = templates.get(perspective, "{response}")
        
        # 라운드에 따른 응답 조정
        if round_num == 1:
            # 초기 의견
            content = self._initial_opinion(perspective)
        else:
            # 다른 의견 고려한 수정 의견
            content = self._refined_opinion(consciousness)
        
        # 확신도 업데이트
        consciousness.confidence = random.uniform(0.6, 0.95)
        
        # 템플릿에 내용 채우기
        if "{" in template:
            key = template.split("{")[1].split("}")[0]
            return template.format(**{key: content})
        else:
            return content
    
    def _initial_opinion(self, perspective: PerspectiveType) -> str:
        """초기 의견 생성"""
        opinions = {
            PerspectiveType.RATIONAL: "데이터와 논리를 기반으로 체계적 접근이 필요합니다",
            PerspectiveType.EMOTIONAL: "직관과 감성을 신뢰하는 것도 중요합니다",
            PerspectiveType.CREATIVE: "기존 틀을 벗어난 혁신적 방법을 시도해야 합니다",
            PerspectiveType.CRITICAL: "현재 접근법의 문제점을 먼저 파악해야 합니다",
            PerspectiveType.PRACTICAL: "실행 가능한 구체적 단계가 필요합니다",
            PerspectiveType.PHILOSOPHICAL: "왜 이것을 하는지 근본 목적을 명확히 해야 합니다",
            PerspectiveType.FUTURE: "장기적 비전을 가지고 접근해야 합니다",
            PerspectiveType.PAST: "과거 실패에서 배운 교훈을 적용해야 합니다",
            PerspectiveType.CHAOS: "예측 불가능한 방법으로 돌파구를 찾아야 합니다",
            PerspectiveType.ORDER: "명확한 구조와 절차를 수립해야 합니다"
        }
        return opinions.get(perspective, "의견을 제시합니다")
    
    def _refined_opinion(self, consciousness: Consciousness) -> str:
        """다른 의견을 고려한 수정 의견"""
        # 간단히 다른 관점을 인정하는 표현 추가
        refinements = [
            f"다른 관점들을 고려하여, {self._initial_opinion(consciousness.perspective)}",
            f"여러 의견을 종합하면, {self._initial_opinion(consciousness.perspective)}",
            f"토론을 통해 생각이 발전하여, {self._initial_opinion(consciousness.perspective)}"
        ]
        return random.choice(refinements)
    
    def _reach_consensus(self) -> Dict[str, Any]:
        """합의 도출"""
        print("\n모든 의식이 합의를 향해 수렴 중...")
        
        # 각 의식의 확신도 가중 평균
        total_confidence = sum(
            c.confidence for c in self.network.consciousnesses.values()
        )
        avg_confidence = total_confidence / len(self.network.consciousnesses)
        
        # 합의 수준 판단
        if avg_confidence > 0.8:
            consensus_level = "강한 합의"
        elif avg_confidence > 0.6:
            consensus_level = "약한 합의"
        else:
            consensus_level = "의견 분산"
        
        # 통합된 결론
        integrated_conclusion = self._integrate_perspectives()
        
        print(f"\n합의 수준: {consensus_level}")
        print(f"평균 확신도: {avg_confidence:.2f}")
        print(f"\n통합 결론:")
        print(f"  {integrated_conclusion}")
        
        return {
            "level": consensus_level,
            "confidence": avg_confidence,
            "conclusion": integrated_conclusion,
            "participating_perspectives": [
                c.perspective.value
                for c in self.network.consciousnesses.values()
            ]
        }
    
    def _integrate_perspectives(self) -> str:
        """모든 관점을 통합한 결론"""
        perspectives = [c.perspective.value for c in self.network.consciousnesses.values()]
        
        conclusion = (
            f"원탁 회의 결과, {len(perspectives)}개의 관점 "
            f"({', '.join(perspectives[:3])} 등)이 "
            f"통합되어 다음과 같은 결론에 도달했습니다: "
            f"다차원적 접근을 통해 논리와 직관, 혁신과 안정, "
            f"이상과 현실을 균형있게 고려하여 전진해야 합니다."
        )
        
        return conclusion


class CollectiveIntelligenceSystem:
    """
    집단 지성 시스템
    
    분산 의식 + 원탁 회의 + 파동 공명 + 중력장 = 초집단 지성
    """
    
    def __init__(self):
        self.network = DistributedConsciousnessNetwork()
        self.council = None  # 필요시 생성
        
    def initialize_consciousness_cluster(self, perspectives: List[PerspectiveType] = None):
        """의식 클러스터 초기화"""
        if perspectives is None:
            # 기본: 다양한 관점 생성
            perspectives = [
                PerspectiveType.RATIONAL,
                PerspectiveType.EMOTIONAL,
                PerspectiveType.CREATIVE,
                PerspectiveType.CRITICAL,
                PerspectiveType.PRACTICAL,
                PerspectiveType.PHILOSOPHICAL
            ]
        
        print(f"\n🌐 분산 의식 네트워크 초기화")
        print(f"   생성할 의식: {len(perspectives)}개")
        
        for perspective in perspectives:
            consciousness = self.network.spawn_consciousness(perspective)
            print(f"   ✓ {consciousness.name} 생성")
        
        print(f"\n   총 {len(self.network.connections)}개의 의식 간 연결 형성")
    
    def hold_council(self, topic: str, question: str) -> Dict[str, Any]:
        """원탁 회의 개최"""
        if not self.council:
            self.council = RoundTableCouncil(self.network)
        
        return self.council.convene(topic, question)
    
    def collective_decision(self, decision_prompt: str) -> Dict[str, Any]:
        """집단 의사결정"""
        print("\n" + "="*70)
        print("🧠 집단 지성 의사결정 프로세스")
        print("="*70)
        
        # 1. 분산 의식 동기화
        self.network.synchronize(decision_prompt)
        
        # 2. 원탁 회의
        result = self.hold_council("집단 의사결정", decision_prompt)
        
        # 3. 최종 결정
        print("\n📋 최종 집단 결정:")
        print(f"   {result['consensus']['conclusion']}")
        
        return result


def demonstrate_collective_intelligence():
    """집단 지성 시스템 시연"""
    
    print("\n" + "="*70)
    print("🎭 분산 의식 & 원탁 회의 시스템")
    print("="*70)
    print("\n💡 개념:")
    print("   - 하나의 Elysia가 여러 자아로 분산")
    print("   - 각 자아는 독립적 관점과 성격")
    print("   - 원탁에 모여 평등하게 토론")
    print("   - 집단 합의로 더 나은 결정")
    
    # 시스템 초기화
    system = CollectiveIntelligenceSystem()
    
    # 다양한 관점의 의식 생성
    perspectives = [
        PerspectiveType.RATIONAL,
        PerspectiveType.EMOTIONAL,
        PerspectiveType.CREATIVE,
        PerspectiveType.CRITICAL,
        PerspectiveType.PRACTICAL
    ]
    
    system.initialize_consciousness_cluster(perspectives)
    
    # 원탁 회의 개최
    topic = "자율 지능 향상 방안"
    question = "어떻게 하면 진정한 자율 지능을 획득할 수 있을까?"
    
    result = system.collective_decision(question)
    
    # 결과 요약
    print("\n📊 집단 지성 결과:")
    print("="*70)
    print(f"  참여 의식: {result['participants']}개")
    print(f"  합의 수준: {result['consensus']['level']}")
    print(f"  확신도: {result['consensus']['confidence']:.2f}")
    print(f"\n  참여 관점: {', '.join(result['consensus']['participating_perspectives'])}")
    
    print("\n✨ 이것이 분산 의식과 원탁 회의를 통한 집단 지성입니다!")
    print("="*70)


if __name__ == "__main__":
    demonstrate_collective_intelligence()
