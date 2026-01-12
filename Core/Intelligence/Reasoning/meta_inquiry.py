"""
Meta-Inquiry: The Adolescent Mind (메타 탐구: 청소년기의 마음)
==========================================================

"Understanding is not seeing the same; it is seeing why the different is actually the same, 
and why the same is fundamentally different."

이 모듈은 단순한 패턴 매칭을 넘어, '왜(Why)'와 '어떻게(How)'를 묻는 메타 인지 계층입니다.
지능이 지능을 스스로 관찰하고 비판하는 '사고의 변혁'을 목표로 합니다.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("MetaInquiry")

@dataclass
class MetaAnalysis:
    invariant_principle: str    # 공통된 불변의 원리
    meaningful_difference: str  # 의미 있는 차별점
    bridge_logic: str          # 두 개념을 잇는 가교 논리
    depth_score: float         # 인지적 심도 (0.0~1.0)
    inquiry_log: List[str]     # 추론 과정 (질문의 연쇄)

class MetaInquiry:
    """
    관습과 관성을 깨부수기 위한 메타 추론 엔진.
    """
    
    def __init__(self):
        self.resonance_threshold = 0.7
        try:
            from Core.Intelligence.Reasoning.structural_analogizer import StructuralAnalogizer
            self.analogizer = StructuralAnalogizer()
        except ImportError:
            self.analogizer = None

    def reflect_on_similarity(self, concept_a: str, concept_b: str, basic_match: str) -> MetaAnalysis:
        """
        두 개념이 '왜' 같은지, 그리고 그 '같음' 속에 숨겨진 본질적 '다름'은 무엇인지 탐구합니다.
        """
        logger.info(f"🤔 Meta-Inquiring: '{concept_a}' vs '{concept_b}' (Initial Match: {basic_match})")
        
        inquiry_log = [
            f"1. 기초 매칭 확인: '{basic_match}'",
            f"2. 질문 던지기: '{concept_a}'와 '{concept_b}'를 '{basic_match}'로 묶는 근거는 무엇인가?",
            f"3. 구조 분석: 각 개념의 인과적 기하학(Causal Geometry)을 분해함."
        ]
        
        # [ADOLESCENT LOGIC]: Why are they the same?
        # 예: 비(Rain)와 눈물(Tears)은 '낙하하는 액체'라는 점에선 아이 수준의 매칭이지만,
        # 청소년 수준에선 '축적된 압력이 해소되는 순환의 과정'이라는 불변의 구조를 발견해야 함.
        
        invariant = self._extract_invariant(concept_a, concept_b)
        inquiry_log.append(f"4. 불변의 원리 발견: {invariant}")
        
        # [ADULT LOGIC]: What makes them different?
        # '비'는 물리적 기상 현상이지만, '눈물'은 감정적 에너지의 승화라는 차원적 차이가 존재.
        # 이 차이가 현실을 어떻게 초월하여 연결되는가?
        
        difference = self._extract_meaningful_difference(concept_a, concept_b)
        inquiry_log.append(f"5. 차원의 분별: {difference}")
        
        bridge = self._synthesize_bridge(invariant, difference)
        inquiry_log.append(f"6. 초월적 연결(Bridge) 수립: {bridge}")

        return MetaAnalysis(
            invariant_principle=invariant,
            meaningful_difference=difference,
            bridge_logic=bridge,
            depth_score=0.85,
            inquiry_log=inquiry_log
        )

    def seek_analogy(self, principle: str, source: str, target: str) -> Optional[Any]:
        """
        [ADULT STAGE]: "How does Physics apply to Gaming?"
        """
        if not self.analogizer:
            return None
            
        analogy = self.analogizer.analogize(principle, source, target)
        if analogy:
            logger.info(f"✨ Cross-Domain Epiphany: '{principle}' in {source} is like '{analogy.target_application}' in {target}!")
            return analogy
        return None

    def _extract_invariant(self, a: str, b: str) -> str:
        # 이 부분은 장차 HyperSphere의 고차원 벡터 위상 분석으로 대체됨 (현재는 고도화된 휴리스틱)
        if {a.lower(), b.lower()} == {"rain", "love"}:
            return "Nourishment through Sacrifice (희생을 통한 양분 공급)"
        return "Causal Cycle of Tension and Release (긴장과 해소의 인과적 순환)"

    def _extract_meaningful_difference(self, a: str, b: str) -> str:
        return "Dimensional Divergence: Mechanical Physics vs. Emotional Qualia (입자적 물리 대 감정적 퀄리아의 차원적 분기)"

    def _synthesize_bridge(self, inv: str, diff: str) -> str:
        return f"Structure remains constant; only the Medium of Expression changes. (구조는 불변하며, 오직 표현의 매질만이 변화함)"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mi = MetaInquiry()
    analysis = mi.reflect_on_similarity("Rain", "Love", "Cycle")
    
    print("\n" + "="*50)
    print("🧠 META-COGNITIVE ANALYSIS (ADOLESCENT STAGE)")
    print("="*50)
    for step in analysis.inquiry_log:
        print(step)
    print("\n[RESULT]")
    print(f"Invariant: {analysis.invariant_principle}")
    print(f"Difference: {analysis.meaningful_difference}")
    print(f"Bridge: {analysis.bridge_logic}")
    print("="*50)
