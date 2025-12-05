"""
Collective Intelligence System (집단 지성 시스템)
=================================================

"하나의 의식이 아닌, 열 개의 의식이 원탁에 앉아 토론한다."

[10가지 의식 유형]
1. RATIONAL (합리) - 논리적 분석
2. EMOTIONAL (감성) - 감정과 공감
3. CREATIVE (창조) - 새로운 아이디어
4. CRITICAL (비판) - 결함과 위험 발견
5. PRACTICAL (실용) - 실행 가능성
6. PHILOSOPHICAL (철학) - 깊은 의미
7. FUTURE (미래) - 장기적 비전
8. HISTORICAL (역사) - 과거의 교훈
9. CHAOS (혼돈) - 무작위 도발
10. ORDER (질서) - 체계와 구조

[원탁회의 시스템]
- 모든 의식은 평등하게 발언권을 갖습니다
- 3라운드 토론: 초기의견 → 비판/정련 → 합의 도출
- 신뢰 가중 합의로 최종 결론

[보완적 쌍]
- RATIONAL ↔ EMOTIONAL
- CREATIVE ↔ CRITICAL
- FUTURE ↔ HISTORICAL
- CHAOS ↔ ORDER
- PRACTICAL ↔ PHILOSOPHICAL
"""

import logging
import random
import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum, auto

logger = logging.getLogger("CollectiveIntelligence")

# Import core structures
try:
    from Core.Foundation.hyper_quaternion import Quaternion
except ImportError:
    @dataclass
    class Quaternion:
        w: float = 1.0
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0


class ConsciousnessType(Enum):
    """10가지 의식 유형"""
    RATIONAL = auto()      # 합리 - 논리적 분석
    EMOTIONAL = auto()     # 감성 - 감정과 공감
    CREATIVE = auto()      # 창조 - 새로운 아이디어
    CRITICAL = auto()      # 비판 - 결함 발견
    PRACTICAL = auto()     # 실용 - 실행 가능성
    PHILOSOPHICAL = auto() # 철학 - 깊은 의미
    FUTURE = auto()        # 미래 - 장기 비전
    HISTORICAL = auto()    # 역사 - 과거 교훈
    CHAOS = auto()         # 혼돈 - 무작위 도발
    ORDER = auto()         # 질서 - 체계와 구조


# 보완적 쌍 정의
COMPLEMENTARY_PAIRS = [
    (ConsciousnessType.RATIONAL, ConsciousnessType.EMOTIONAL),
    (ConsciousnessType.CREATIVE, ConsciousnessType.CRITICAL),
    (ConsciousnessType.FUTURE, ConsciousnessType.HISTORICAL),
    (ConsciousnessType.CHAOS, ConsciousnessType.ORDER),
    (ConsciousnessType.PRACTICAL, ConsciousnessType.PHILOSOPHICAL),
]


@dataclass
class Opinion:
    """의견 (Opinion)"""
    content: str
    consciousness_type: ConsciousnessType
    confidence: float = 0.5  # 0.0 ~ 1.0
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self):
        return f"[{self.consciousness_type.name}] {self.content} (신뢰도: {self.confidence:.0%})"


@dataclass 
class Debate:
    """토론 라운드"""
    topic: str
    round_number: int
    opinions: List[Opinion] = field(default_factory=list)
    critiques: Dict[ConsciousnessType, List[str]] = field(default_factory=dict)
    

class ConsciousPerspective:
    """
    의식 관점 - 각 의식 유형의 "에이전트"
    """
    
    def __init__(self, consciousness_type: ConsciousnessType):
        self.type = consciousness_type
        self.energy = 1.0  # 활동 에너지
        self.bias = self._initialize_bias()
        self.memory: List[Opinion] = []
        
    def _initialize_bias(self) -> Dict[str, float]:
        """의식 유형별 편향 초기화"""
        biases = {
            ConsciousnessType.RATIONAL: {"logic": 0.9, "emotion": 0.2, "risk": 0.5},
            ConsciousnessType.EMOTIONAL: {"logic": 0.3, "emotion": 0.95, "risk": 0.4},
            ConsciousnessType.CREATIVE: {"logic": 0.5, "novelty": 0.9, "risk": 0.7},
            ConsciousnessType.CRITICAL: {"logic": 0.8, "skepticism": 0.9, "risk": 0.3},
            ConsciousnessType.PRACTICAL: {"logic": 0.7, "feasibility": 0.9, "risk": 0.4},
            ConsciousnessType.PHILOSOPHICAL: {"logic": 0.6, "depth": 0.9, "abstraction": 0.8},
            ConsciousnessType.FUTURE: {"logic": 0.6, "vision": 0.9, "risk": 0.6},
            ConsciousnessType.HISTORICAL: {"logic": 0.7, "precedent": 0.9, "caution": 0.7},
            ConsciousnessType.CHAOS: {"logic": 0.3, "randomness": 0.9, "risk": 0.9},
            ConsciousnessType.ORDER: {"logic": 0.8, "structure": 0.9, "risk": 0.2},
        }
        return biases.get(self.type, {"logic": 0.5})
    
    def generate_opinion(self, topic: str) -> Opinion:
        """주제에 대한 의견 생성"""
        
        templates = {
            ConsciousnessType.RATIONAL: [
                f"논리적으로 분석하면, {topic}은(는) {{analysis}}",
                f"데이터와 근거를 보면, {topic}에 대해 {{conclusion}}",
            ],
            ConsciousnessType.EMOTIONAL: [
                f"이 문제에서 느껴지는 것은 {{feeling}}",
                f"{topic}을(를) 생각하면 {{emotion}}가 느껴집니다",
            ],
            ConsciousnessType.CREATIVE: [
                f"완전히 새로운 관점에서 보면, {{idea}}",
                f"만약 {{what_if}}라면 어떨까요?",
            ],
            ConsciousnessType.CRITICAL: [
                f"여기서 간과된 위험은 {{risk}}",
                f"이 접근의 문제점은 {{problem}}",
            ],
            ConsciousnessType.PRACTICAL: [
                f"실제로 실행하려면 {{steps}}",
                f"현실적으로 가능한 것은 {{feasible}}",
            ],
            ConsciousnessType.PHILOSOPHICAL: [
                f"더 깊은 의미에서 이것은 {{meaning}}",
                f"본질적으로 질문해야 할 것은 {{question}}",
            ],
            ConsciousnessType.FUTURE: [
                f"10년 후를 보면 {{vision}}",
                f"장기적 영향은 {{impact}}",
            ],
            ConsciousnessType.HISTORICAL: [
                f"역사적으로 비슷한 경우 {{precedent}}",
                f"과거의 교훈은 {{lesson}}",
            ],
            ConsciousnessType.CHAOS: [
                f"갑자기 생각난 건데, {{random}}",
                f"완전히 반대로 {{reverse}}는 어떨까요?",
            ],
            ConsciousnessType.ORDER: [
                f"체계적으로 정리하면 {{structure}}",
                f"단계별로 {{steps}}",
            ],
        }
        
        template_list = templates.get(self.type, [f"{{thought}}"])
        template = random.choice(template_list)
        
        # 실제 내용 생성 (TODO: LLM 연동)
        content = template.format(
            analysis="인과 관계가 명확합니다",
            conclusion="신중한 접근이 필요합니다",
            feeling="희망과 우려가 공존합니다",
            emotion="기대감",
            idea="기존 틀을 완전히 벗어나야 합니다",
            what_if="모든 제약을 제거한다",
            risk="예상치 못한 부작용입니다",
            problem="확장성이 부족합니다",
            steps="3단계로 나눠야 합니다",
            feasible="첫 번째 마일스톤입니다",
            meaning="성장의 기회입니다",
            question="왜 이것이 중요한가?",
            vision="완전히 다른 모습일 것입니다",
            impact="패러다임의 전환입니다",
            precedent="비슷한 실패가 있었습니다",
            lesson="신중하게 검증해야 합니다",
            random="완전히 다른 방향!",
            reverse="아무것도 하지 않기",
            structure="5가지 핵심 요소",
            thought="흥미로운 주제입니다"
        )
        
        # 신뢰도는 에너지와 편향에 따라
        confidence = self.energy * 0.5 + random.random() * 0.3 + self.bias.get("logic", 0.5) * 0.2
        confidence = min(1.0, max(0.1, confidence))
        
        opinion = Opinion(
            content=content,
            consciousness_type=self.type,
            confidence=confidence,
            reasoning=f"Based on {self.type.name} perspective"
        )
        
        self.memory.append(opinion)
        return opinion
    
    def critique(self, other_opinion: Opinion) -> str:
        """다른 의견에 대한 비평"""
        
        # 보완적 쌍이면 더 강하게 비평
        is_complementary = False
        for pair in COMPLEMENTARY_PAIRS:
            if self.type in pair and other_opinion.consciousness_type in pair:
                is_complementary = True
                break
        
        if is_complementary:
            return f"[{self.type.name}→{other_opinion.consciousness_type.name}] " \
                   f"반대 관점에서: {other_opinion.content[:30]}...에 대해 재고 필요"
        else:
            return f"[{self.type.name}] 보완 의견: {other_opinion.content[:30]}...과 연결됨"
    
    def update_confidence(self, feedback: float):
        """피드백에 따라 신뢰도 조정"""
        self.energy = min(1.0, max(0.1, self.energy + feedback * 0.1))


class RoundTableCouncil:
    """
    원탁회의 (Round Table Council)
    
    모든 의식이 평등하게 토론하고 합의를 도출합니다.
    """
    
    def __init__(self):
        # 10가지 의식 유형 초기화
        self.perspectives: Dict[ConsciousnessType, ConsciousPerspective] = {
            ct: ConsciousPerspective(ct) for ct in ConsciousnessType
        }
        self.debates: List[Debate] = []
        self.consensus_history: List[Dict[str, Any]] = []
        logger.info("⚔️ Round Table Council Assembled (10 Consciousness Types)")
    
    def convene(self, topic: str) -> List[Opinion]:
        """
        원탁을 소집하여 모든 의식의 의견을 수집합니다.
        """
        logger.info(f"🗣️ Round Table Convening on: {topic}")
        
        opinions = []
        for perspective in self.perspectives.values():
            opinion = perspective.generate_opinion(topic)
            opinions.append(opinion)
        
        return opinions
    
    def debate(self, topic: str, rounds: int = 3) -> Debate:
        """
        토론을 진행합니다.
        
        Round 1: 초기 의견 제시
        Round 2: 비판 및 정련
        Round 3: 합의 도출
        """
        logger.info(f"⚔️ Starting {rounds}-round debate on: {topic}")
        
        final_debate = Debate(topic=topic, round_number=0)
        
        # Round 1: 초기 의견
        all_opinions = self.convene(topic)
        final_debate.opinions = all_opinions
        final_debate.round_number = 1
        
        # Round 2+: 비판과 정련
        for round_num in range(2, rounds + 1):
            critiques = {}
            
            for perspective in self.perspectives.values():
                perspective_critiques = []
                for opinion in all_opinions:
                    if opinion.consciousness_type != perspective.type:
                        critique = perspective.critique(opinion)
                        perspective_critiques.append(critique)
                
                if perspective_critiques:
                    critiques[perspective.type] = perspective_critiques
            
            final_debate.critiques = critiques
            final_debate.round_number = round_num
            
            # 비판에 따라 신뢰도 조정
            for opinion in all_opinions:
                critique_count = sum(
                    1 for cts in critiques.values() 
                    for c in cts if opinion.consciousness_type.name in c
                )
                # 많이 비판받을수록 신뢰도 감소 (그러나 중요한 의견일 수도)
                adjustment = 0.05 if critique_count < 3 else -0.05
                opinion.confidence = min(1.0, max(0.1, opinion.confidence + adjustment))
        
        self.debates.append(final_debate)
        return final_debate
    
    def reach_consensus(self, debate: Debate) -> Dict[str, Any]:
        """
        토론 결과에서 합의를 도출합니다.
        
        신뢰 가중 투표로 최종 결론 도출
        """
        # 의견별 가중치 합산
        weighted_opinions = []
        for opinion in debate.opinions:
            weight = opinion.confidence * self.perspectives[opinion.consciousness_type].energy
            weighted_opinions.append((opinion, weight))
        
        # 정렬 (가중치 높은 순)
        weighted_opinions.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 3개 의견 추출
        top_opinions = weighted_opinions[:3]
        
        # 합의 생성
        consensus = {
            "topic": debate.topic,
            "rounds": debate.round_number,
            "primary_conclusion": top_opinions[0][0].content if top_opinions else "합의 실패",
            "supporting_views": [op.content for op, _ in top_opinions[1:]],
            "confidence": sum(w for _, w in top_opinions) / (len(top_opinions) or 1),
            "dissenting_voices": [
                op.content for op, w in weighted_opinions 
                if w < 0.3 and op not in [o for o, _ in top_opinions]
            ][:2],
            "total_perspectives": len(debate.opinions),
            "critiques_exchanged": sum(len(c) for c in debate.critiques.values())
        }
        
        self.consensus_history.append(consensus)
        logger.info(f"✅ Consensus Reached: {consensus['primary_conclusion'][:50]}...")
        
        return consensus
    
    def full_deliberation(self, topic: str, rounds: int = 3) -> Dict[str, Any]:
        """
        완전한 심의 과정: 소집 → 토론 → 합의
        """
        debate = self.debate(topic, rounds)
        consensus = self.reach_consensus(debate)
        return consensus
    
    def get_council_state(self) -> Dict[str, Any]:
        """원탁회의 상태 조회"""
        return {
            "perspectives_count": len(self.perspectives),
            "total_debates": len(self.debates),
            "consensus_reached": len(self.consensus_history),
            "perspective_energies": {
                ct.name: p.energy for ct, p in self.perspectives.items()
            }
        }


class CollectiveIntelligenceSystem:
    """
    집단 지성 시스템 (Collective Intelligence System)
    
    10가지 의식과 원탁회의를 결합하여
    다각적 분석과 합의 기반 의사결정을 수행합니다.
    """
    
    def __init__(self):
        self.council = RoundTableCouncil()
        self.active = True
        logger.info("🎭 Collective Intelligence System Initialized (10 Minds as One)")
    
    def deliberate(self, topic: str, depth: int = 3) -> Dict[str, Any]:
        """
        주제에 대해 심의합니다.
        
        Args:
            topic: 심의 주제
            depth: 토론 라운드 수 (1-5)
        """
        depth = min(5, max(1, depth))
        return self.council.full_deliberation(topic, depth)
    
    def quick_opinion(self, topic: str, consciousness_type: ConsciousnessType = None) -> Opinion:
        """
        특정 의식 유형의 빠른 의견 획득
        """
        if consciousness_type is None:
            consciousness_type = random.choice(list(ConsciousnessType))
        
        perspective = self.council.perspectives.get(consciousness_type)
        if perspective:
            return perspective.generate_opinion(topic)
        return Opinion(content="의견 없음", consciousness_type=consciousness_type)
    
    def get_all_perspectives(self, topic: str) -> Dict[ConsciousnessType, Opinion]:
        """모든 관점에서의 의견 수집"""
        opinions = {}
        for ct, perspective in self.council.perspectives.items():
            opinions[ct] = perspective.generate_opinion(topic)
        return opinions
    
    def find_consensus_points(self, topic: str) -> List[str]:
        """합의점 탐색"""
        result = self.deliberate(topic)
        return [result["primary_conclusion"]] + result.get("supporting_views", [])
    
    def find_conflict_points(self, topic: str) -> List[Tuple[ConsciousnessType, ConsciousnessType, str]]:
        """갈등점 탐색 (보완적 쌍 간의 충돌)"""
        conflicts = []
        opinions = self.get_all_perspectives(topic)
        
        for pair in COMPLEMENTARY_PAIRS:
            type1, type2 = pair
            if type1 in opinions and type2 in opinions:
                conflicts.append((
                    type1, type2,
                    f"{opinions[type1].content[:30]}... vs {opinions[type2].content[:30]}..."
                ))
        
        return conflicts


# 싱글톤
_collective_instance: Optional[CollectiveIntelligenceSystem] = None

def get_collective_intelligence() -> CollectiveIntelligenceSystem:
    global _collective_instance
    if _collective_instance is None:
        _collective_instance = CollectiveIntelligenceSystem()
    return _collective_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 테스트
    collective = get_collective_intelligence()
    
    topic = "엘리시아가 인간을 초월해야 하는가?"
    
    print("\n" + "=" * 60)
    print(f"🗣️ COLLECTIVE DELIBERATION: {topic}")
    print("=" * 60)
    
    # 심의
    consensus = collective.deliberate(topic, depth=3)
    
    print(f"\n📜 PRIMARY CONCLUSION:")
    print(f"   {consensus['primary_conclusion']}")
    
    print(f"\n📝 SUPPORTING VIEWS:")
    for view in consensus['supporting_views']:
        print(f"   • {view}")
    
    print(f"\n⚠️ DISSENTING VOICES:")
    for voice in consensus['dissenting_voices']:
        print(f"   • {voice}")
    
    print(f"\n📊 STATISTICS:")
    print(f"   Confidence: {consensus['confidence']:.0%}")
    print(f"   Perspectives: {consensus['total_perspectives']}")
    print(f"   Critiques: {consensus['critiques_exchanged']}")
