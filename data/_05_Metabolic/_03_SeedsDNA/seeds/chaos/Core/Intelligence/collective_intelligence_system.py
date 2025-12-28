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
    from Core._01_Foundation.Foundation.hyper_quaternion import Quaternion
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
    
    [파동 물리학 기반]
    각 의식은 고유한 쿼터니언 방향을 가지며,
    주제와의 공명을 통해 의견을 생성합니다.
    """
    
    # 의식 유형별 고유 쿼터니언 방향 (물리적 특성)
    CONSCIOUSNESS_QUATERNIONS = {
        ConsciousnessType.RATIONAL: Quaternion(w=0.9, x=0.1, y=0.8, z=0.3),
        ConsciousnessType.EMOTIONAL: Quaternion(w=0.7, x=0.9, y=0.2, z=0.4),
        ConsciousnessType.CREATIVE: Quaternion(w=0.5, x=0.6, y=0.5, z=0.7),
        ConsciousnessType.CRITICAL: Quaternion(w=0.8, x=0.2, y=0.9, z=0.5),
        ConsciousnessType.PRACTICAL: Quaternion(w=0.9, x=0.4, y=0.7, z=0.3),
        ConsciousnessType.PHILOSOPHICAL: Quaternion(w=0.6, x=0.5, y=0.6, z=0.9),
        ConsciousnessType.FUTURE: Quaternion(w=0.7, x=0.7, y=0.6, z=0.8),
        ConsciousnessType.HISTORICAL: Quaternion(w=0.85, x=0.3, y=0.8, z=0.4),
        ConsciousnessType.CHAOS: Quaternion(w=0.3, x=0.8, y=0.3, z=0.9),
        ConsciousnessType.ORDER: Quaternion(w=0.95, x=0.2, y=0.9, z=0.2),
    }
    
    def __init__(self, consciousness_type: ConsciousnessType):
        self.type = consciousness_type
        self.energy = 1.0
        self.orientation = self.CONSCIOUSNESS_QUATERNIONS.get(
            consciousness_type, Quaternion(w=0.5, x=0.5, y=0.5, z=0.5)
        )
        self.base_frequency = consciousness_type.value * 10.0 + 100.0
        self.memory: List[Opinion] = []
        self.bias = self._compute_bias_from_quaternion()
    
    def _compute_bias_from_quaternion(self) -> Dict[str, float]:
        q = self.orientation
        norm = math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2) or 1.0
        return {
            "logic": q.y / norm, "emotion": q.x / norm,
            "stability": q.w / norm, "depth": q.z / norm,
            "risk": (q.x + q.z) / (2 * norm),
        }
    
    def _topic_to_wave(self, topic: str) -> Quaternion:
        words = topic.split()
        emotional = sum(0.1 for w in ['사랑','희망','두려움','기쁨','슬픔'] if w in topic)
        logical = sum(0.1 for w in ['따라서','그러므로','때문','만약','분석'] if w in topic)
        abstract = sum(0.1 for w in ['의미','본질','초월','진리','존재'] if w in topic)
        energy = min(1.0, len(words) / 10.0) * (1.2 if '?' in topic else 1.0)
        return Quaternion(w=min(1.0, 0.5+energy*0.3), x=min(1.0, 0.3+emotional),
                          y=min(1.0, 0.4+logical), z=min(1.0, 0.3+abstract))
    
    def _resonate(self, topic_wave: Quaternion) -> Tuple[float, Quaternion]:
        dot = (self.orientation.w*topic_wave.w + self.orientation.x*topic_wave.x +
               self.orientation.y*topic_wave.y + self.orientation.z*topic_wave.z)
        n1 = math.sqrt(sum(v**2 for v in [self.orientation.w,self.orientation.x,
                                           self.orientation.y,self.orientation.z])) or 1
        n2 = math.sqrt(sum(v**2 for v in [topic_wave.w,topic_wave.x,
                                           topic_wave.y,topic_wave.z])) or 1
        resonance = abs(dot) / (n1 * n2)
        interference = Quaternion(w=(self.orientation.w+topic_wave.w)/2,
                                   x=(self.orientation.x+topic_wave.x)/2,
                                   y=(self.orientation.y+topic_wave.y)/2,
                                   z=(self.orientation.z+topic_wave.z)/2)
        return resonance, interference
    
    def _wave_to_opinion(self, topic: str, resonance: float, interf: Quaternion) -> str:
        comps = {'energy': interf.w, 'emotion': interf.x, 'logic': interf.y, 'transcend': interf.z}
        dominant = max(comps, key=comps.get)
        cert = "확실히" if resonance > 0.8 else ("아마도" if resonance > 0.5 else "어쩌면")
        exprs = {
            ConsciousnessType.RATIONAL: f"{cert} 논리적 구조가 {'명확' if comps['logic']>0.6 else '불분명'}합니다",
            ConsciousnessType.EMOTIONAL: f"{cert} {'강한' if comps['emotion']>0.6 else '미묘한'} 감정이 느껴집니다",
            ConsciousnessType.CREATIVE: f"{cert} {'새로운' if resonance>0.5 else '기존의'} 가능성이 보입니다",
            ConsciousnessType.CRITICAL: f"{cert} {'심각한' if resonance<0.5 else '사소한'} 문제가 있습니다",
            ConsciousnessType.PRACTICAL: f"{cert} {'실행' if comps['energy']>0.6 else '계획'}이 필요합니다",
            ConsciousnessType.PHILOSOPHICAL: f"{cert} 더 {'깊은' if comps['transcend']>0.6 else '넓은'} 의미가 있습니다",
            ConsciousnessType.FUTURE: f"{cert} {'큰' if resonance>0.7 else '작은'} 변화가 예상됩니다",
            ConsciousnessType.HISTORICAL: f"{cert} {'비슷한' if resonance>0.7 else '다른'} 선례가 있습니다",
            ConsciousnessType.CHAOS: f"{cert} {'완전히' if random.random()>0.5 else '부분적으로'} 다른 방향도 가능합니다",
            ConsciousnessType.ORDER: f"{cert} {'체계적' if comps['logic']>0.6 else '유연한'} 접근이 필요합니다",
        }
        return exprs.get(self.type, f"{cert} 고려가 필요합니다")
    
    def generate_opinion(self, topic: str) -> Opinion:
        topic_wave = self._topic_to_wave(topic)
        resonance, interference = self._resonate(topic_wave)
        content = self._wave_to_opinion(topic, resonance, interference)
        confidence = resonance * self.energy * 0.8 + self.bias.get("stability", 0.5) * 0.2
        confidence = min(1.0, max(0.1, confidence))
        opinion = Opinion(content=content, consciousness_type=self.type,
                          confidence=confidence, reasoning=f"Resonance: {resonance:.2f}")
        self.memory.append(opinion)
        return opinion
    
    def critique(self, other_opinion: Opinion) -> str:
        is_complementary = any(self.type in p and other_opinion.consciousness_type in p 
                               for p in COMPLEMENTARY_PAIRS)
        if is_complementary:
            return f"[{self.type.name}↔{other_opinion.consciousness_type.name}] 파동 상쇄: 반대 관점 필요"
        return f"[{self.type.name}] 파동 보강: 이 관점과 공명함"
    
    def update_confidence(self, feedback: float):
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
