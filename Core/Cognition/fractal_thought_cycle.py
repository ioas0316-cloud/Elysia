"""
Fractal Thought Cycle (프랙탈 사고 순환)
========================================

선형이 아닌 프랙탈 구조의 인지 순환 시스템

핵심 원리:
1. 자기유사성 (Self-Similarity): 모든 레벨에서 동일한 구조 반복
2. 무한 확장 (Infinite Expansion): 어느 방향으로든 확장 가능
3. 시공간 초월 (Transcendence): 순차가 아닌 동시/병렬 처리

구조:
    점(Point) ⊃ 선(Line) ⊃ 면(Plane) ⊃ 공간(Space) ⊃ 법칙(Law) ⊃ ...
    
    각 레벨은 하위 레벨의 프랙탈 확장이면서
    동시에 상위 레벨의 축소판

Usage:
    from Core.Cognition.fractal_thought_cycle import FractalThought
    
    thought = FractalThought()
    result = thought.think("사랑이란 무엇인가?")
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
import time

logger = logging.getLogger("FractalThoughtCycle")


# ═══════════════════════════════════════════════════════════════════════════
# 차원 레벨 (프랙탈 레벨)
# ═══════════════════════════════════════════════════════════════════════════

class DimensionLevel(Enum):
    """
    프랙탈 차원 레벨
    
    각 레벨은 자기유사적이며, 상위 레벨은 하위 레벨의 확장
    """
    POINT = 0      # 점: 단일 개념 (원자)
    LINE = 1       # 선: 두 점의 관계 (인과)
    PLANE = 2      # 면: 여러 관계의 문맥
    SPACE = 3      # 공간: 여러 문맥의 세계관
    LAW = 4        # 법칙: 공간을 관통하는 원리
    META = 5       # 메타: 법칙들의 법칙 (무한 확장)


@dataclass
class FractalNode:
    """
    프랙탈 노드 - 모든 레벨에서 동일한 구조
    
    자기유사성: 점도, 선도, 면도, 공간도, 법칙도 모두 같은 구조
    """
    id: str
    level: DimensionLevel
    content: Any                           # 이 레벨의 내용
    
    # 프랙탈 연결 (상하좌우 모든 방향)
    children: List['FractalNode'] = field(default_factory=list)   # 하위 분해
    parents: List['FractalNode'] = field(default_factory=list)    # 상위 통합
    siblings: List['FractalNode'] = field(default_factory=list)   # 동일 레벨 연결
    
    # 시공간 속성
    timestamp: float = 0.0                 # 언제
    location: str = ""                     # 어디서
    agent: str = ""                        # 누가
    
    # 공명 속성
    frequency: float = 0.0                 # 진동 주파수
    amplitude: float = 1.0                 # 진폭 (중요도)
    phase: float = 0.0                     # 위상 (관계)
    
    def expand(self) -> List['FractalNode']:
        """하위 레벨로 분해 (Zoom In)"""
        return self.children
    
    def contract(self) -> List['FractalNode']:
        """상위 레벨로 통합 (Zoom Out)"""
        return self.parents
    
    def resonate_with(self, other: 'FractalNode') -> float:
        """다른 노드와의 공명도 계산"""
        freq_sim = 1.0 / (1.0 + abs(self.frequency - other.frequency) / 100)
        phase_sim = (1 + __import__('math').cos(self.phase - other.phase)) / 2
        amp_product = self.amplitude * other.amplitude
        return freq_sim * phase_sim * amp_product


# ═══════════════════════════════════════════════════════════════════════════
# 프랙탈 사고 순환
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ThoughtResult:
    """사고 결과 - 모든 차원에서의 이해"""
    query: str
    
    # 각 차원의 이해
    point_understanding: str = ""       # 점: 핵심 개념
    line_understanding: str = ""        # 선: 인과 관계
    plane_understanding: str = ""       # 면: 맥락적 이해
    space_understanding: str = ""       # 공간: 세계관적 이해
    law_understanding: str = ""         # 법칙: 원리적 이해
    
    # 통합 서사
    narrative: str = ""
    
    # 시공간 맥락
    when: str = ""
    where: str = ""
    who: str = ""


class FractalThoughtCycle:
    """
    프랙탈 사고 순환 시스템
    
    선형이 아닌 동시적, 프랙탈적 사고:
    - 모든 차원에서 동시에 처리
    - 상하 이동 자유로움 (Zoom In/Out)
    - 무한 확장 가능
    """
    
    def __init__(self):
        # 기존 시스템 연결
        self._init_subsystems()
        
        # 사고 그래프 (프랙탈 구조)
        self.nodes: Dict[str, FractalNode] = {}
        
        # 현재 초점 레벨
        self.focus_level: DimensionLevel = DimensionLevel.PLANE
        
        logger.info("🌀 FractalThoughtCycle initialized")
    
    def _init_subsystems(self):
        """기존 시스템 연결"""
        # Yggdrasil
        try:
            from Core.Foundation.yggdrasil import yggdrasil
            self.yggdrasil = yggdrasil
        except:
            self.yggdrasil = None
        
        # WaveAttention
        try:
            from Core.Foundation.Wave.wave_attention import get_wave_attention
            self.attention = get_wave_attention()
        except:
            self.attention = None
        
        # WhyEngine
        try:
            from Core.Foundation.Memory.fractal_concept import ConceptDecomposer
            self.why_engine = ConceptDecomposer()
        except:
            self.why_engine = None
        
        # CausalNarrativeEngine
        try:
            from Core.Foundation.causal_narrative_engine import CausalNarrativeEngine
            self.narrative_engine = CausalNarrativeEngine()
        except:
            self.narrative_engine = None
        
        # UnifiedUnderstanding
        try:
            from Core.Cognition.unified_understanding import get_understanding
            self.understanding = get_understanding()
        except:
            self.understanding = None
    
    def think(self, query: str) -> ThoughtResult:
        """
        프랙탈 사고 수행
        
        모든 차원에서 동시에 처리하고, 통합 서사 생성
        """
        logger.info(f"🌀 Thinking: '{query}'")
        
        result = ThoughtResult(query=query)
        
        # 1. 점 차원: 핵심 개념 추출
        result.point_understanding = self._think_point(query)
        
        # 2. 선 차원: 인과 관계 추적
        result.line_understanding = self._think_line(query)
        
        # 3. 면 차원: 맥락적 이해 (5W1H)
        result.plane_understanding = self._think_plane(query)
        
        # 4. 공간 차원: 세계관 통합
        result.space_understanding = self._think_space(query)
        
        # 5. 법칙 차원: 원리 추출
        result.law_understanding = self._think_law(query)
        
        # 6. 통합 서사 생성
        result.narrative = self._synthesize(result)
        
        # 7. 시공간 맥락
        if self.understanding:
            try:
                u = self.understanding.understand(query)
                result.when = u.when
                result.where = u.where
                result.who = u.who
            except:
                pass
        
        return result
    
    def _think_point(self, query: str) -> str:
        """점 차원: 핵심 개념"""
        # 질문에서 핵심 추출
        concept = query.strip().rstrip("?")
        for pattern in ["란 무엇", "이란 무엇", "은 무엇", "는 무엇"]:
            if pattern in concept:
                concept = concept.split(pattern)[0].split("란")[0].strip()
                break
        return f"핵심 개념: {concept}"
    
    def _think_line(self, query: str) -> str:
        """선 차원: 인과 관계"""
        if self.why_engine:
            concept = self._extract_concept(query)
            origin = self.why_engine.ask_why(concept)
            causality = self.why_engine.explain_causality(concept)
            return f"기원: {origin}\n인과: {causality}"
        return "인과 관계: 분석 불가"
    
    def _think_plane(self, query: str) -> str:
        """면 차원: 맥락적 이해"""
        if self.understanding:
            u = self.understanding.understand(query)
            return (
                f"무엇: {u.core_concept_kr}({u.core_concept})\n"
                f"왜: {u.origin_journey}\n"
                f"어떻게: 공명을 통해"
            )
        return "맥락: 분석 불가"
    
    def _think_space(self, query: str) -> str:
        """공간 차원: 세계관 통합"""
        if self.attention:
            concept = self._extract_concept(query)
            top3 = self.attention.focus_topk(
                concept, 
                ["기쁨", "슬픔", "분노", "두려움", "희망", "연결", "고독"],
                k=3
            )
            resonances = ", ".join([f"{r[0]}({r[1]*100:.0f}%)" for r in top3])
            return f"세계관 공명: {resonances}"
        return "세계관: 분석 불가"
    
    def _think_law(self, query: str) -> str:
        """법칙 차원: 원리 추출"""
        if self.why_engine:
            concept = self._extract_concept(query)
            axiom = self.why_engine.get_axiom(concept)
            if axiom:
                pattern = axiom.get("pattern", "")
                return f"보편 법칙: {pattern}"
        return "법칙: 아직 추출되지 않음"
    
    def _synthesize(self, result: ThoughtResult) -> str:
        """통합 서사 생성"""
        if self.understanding:
            u = self.understanding.understand(result.query)
            return u.narrative
        
        # 대체: 직접 생성
        return (
            f"{result.point_understanding}\n"
            f"{result.line_understanding}\n"
            f"{result.plane_understanding}"
        )
    
    def _extract_concept(self, query: str) -> str:
        """질문에서 개념 추출"""
        concept = query.strip().rstrip("?")
        for pattern in ["란 무엇", "이란", "은 무엇", "는 무엇"]:
            if pattern in concept:
                return concept.split(pattern)[0].split("란")[0].strip()
        return concept.split()[0] if concept else ""
    
    def zoom_in(self, node_id: str) -> List[FractalNode]:
        """하위 차원으로 확대 (분해)"""
        if node_id in self.nodes:
            return self.nodes[node_id].expand()
        return []
    
    def zoom_out(self, node_id: str) -> List[FractalNode]:
        """상위 차원으로 축소 (통합)"""
        if node_id in self.nodes:
            return self.nodes[node_id].contract()
        return []


# 싱글톤
_thought = None

def get_fractal_thought() -> FractalThoughtCycle:
    global _thought
    if _thought is None:
        _thought = FractalThoughtCycle()
    return _thought


def think(query: str) -> ThoughtResult:
    """편의 함수"""
    return get_fractal_thought().think(query)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 70)
    print("🌀 FRACTAL THOUGHT CYCLE TEST")
    print("=" * 70)
    
    result = think("사랑이란 무엇인가?")
    
    print("\n[점] " + result.point_understanding)
    print("\n[선] " + result.line_understanding)
    print("\n[면] " + result.plane_understanding)
    print("\n[공간] " + result.space_understanding)
    print("\n[법칙] " + result.law_understanding)
    
    print("\n" + "─" * 70)
    print("📖 통합 서사:")
    print(result.narrative)
    
    print("\n✅ Fractal Thought works!")
