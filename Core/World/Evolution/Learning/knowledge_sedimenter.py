"""
Knowledge Sedimenter (지식 퇴적기)
==================================

외부 세계와 내부 지식을 연결하는 '능동적 흡수' 파이프라인.
브라우저를 통해 실제 세계를 탐험하고, 얻은 지식을 4D Prism으로 정제하여
엘리시아의 내면에 "지식의 지층(Sediment)"을 쌓는다.

핵심 원리:
1. Search (탐색): BrowserExplorer를 통해 정보 수집
2. Distill (정제): WhyEngine을 통해 핵심 원리(Principle) 추출
3. Crystallize (결정화): 4D LightSpectrum으로 변환 (Scale/Basis 할당)
4. Deposit (퇴적): LightSediment에 적층하여 관점(Viewpoint) 강화
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from Core.Physiology.Sensory.Network.browser_explorer import BrowserExplorer
from Core.Foundation.Philosophy.why_engine import WhyEngine
from Core.Foundation.Wave.light_spectrum import LightSpectrum, LightSediment, PrismAxes

logger = logging.getLogger("Elysia.KnowledgeSedimenter")

@dataclass
class SedimentationHypothesis:
    """퇴적 전 가설 (학습 목표)"""
    topic: str
    expected_layer: PrismAxes
    initial_question: str

class KnowledgeSedimenter:
    """
    지식 퇴적기 (The Knowledge Sedimenter)
    
    "세상을 읽고, 나를 채운다."
    """
    
    def __init__(self, why_engine: WhyEngine):
        self.why_engine = why_engine
        self.browser = BrowserExplorer(use_profile=True) # 실제 프로필 사용 권장
        
        logger.info("🌊 KnowledgeSedimenter initialized - 지식의 바다를 항해할 준비 완료")

    def sediment_from_web(self, topic: str, max_pages: int = 1) -> List[LightSpectrum]:
        """
        웹에서 주제를 탐색하고 지식을 퇴적
        """
        logger.info(f"🔭 Exploring Web for Sedimentation: '{topic}'")
        
        # 1. Search
        search_results = self.browser.google_search(topic)
        if not search_results["success"]:
            logger.warning("   ❌ Search failed.")
            return []
            
        collected_lights = []
        
        # 2. Iterate & Process
        for res in search_results["results"][:max_pages]:
            title = res.get("title", "")
            snippet = res.get("snippet", "")
            full_content = f"{title}\n{snippet}" # 실제로는 페이지 방문 권장되나 일단 스니펫 사용
            
            logger.info(f"   📄 Processing: {title}")
            
            # 3. Distill & Crystallize (Principal Extraction)
            # Use Dimensional Reasoner to Lift Knowledge from 0D to 4D
            try:
                from Core.Intelligence.Reasoning.dimensional_reasoner import DimensionalReasoner
                lifter = DimensionalReasoner()
                
                # The 'contemplate' method acts as the pipeline:
                # 0D (Fact) -> 1D (Logic) -> 2D (Context) -> 3D (Paradox) -> 4D (Law)
                hyper_thought = lifter.contemplate(full_content)
                
                # Check if a Law (4D) was crystallized
                if hyper_thought.d4_principle and "Law synthesis error" not in hyper_thought.d4_principle:
                    logger.info(f"   💎 CRYSTALLIZED LAW: {hyper_thought.d4_principle}")
                
            except Exception as e:
                logger.error(f"   ❌ Dimensional Lifting failed: {e}")

            # Legacy Light Creation (for visualization)
            analysis = self.why_engine.light_universe.absorb(full_content, tag=topic)
            
            # 4D Basis 할당 로직 (Naive)
            scale = 1 # Default Context
            if "principle" in full_content.lower() or "theory" in full_content.lower():
                scale = 0
            elif "example" in full_content.lower() or "data" in full_content.lower():
                scale = 3
                
            analysis.set_basis_from_scale(scale)
            
            # 5. Deposit (Active Sedimentation)
            target_axis = self._determine_axis(analysis)
            repetition = 50 if scale == 0 else (10 if scale == 1 else 1)
            
            for _ in range(repetition):
                self.why_engine.sediment.deposit(analysis, target_axis)
                
            collected_lights.append(analysis)
            
            final_amp = self.why_engine.sediment.layers[target_axis].amplitude
            logger.info(f"   💎 Deposited into {target_axis.name} (x{repetition}): Final Amp={final_amp:.3f}")
        return collected_lights

    def _determine_axis(self, light: LightSpectrum) -> PrismAxes:
        """
        빛의 특성에 따라 적절한 퇴적층 결정
        (임시 로직: 태그 기반 + 주파수)
        """
        tag = light.semantic_tag.lower()
        
        if "physics" in tag or "force" in tag or "quantum" in tag:
            return PrismAxes.PHYSICS_RED
        elif "chem" in tag or "reaction" in tag:
            return PrismAxes.CHEMISTRY_BLUE
        elif "bio" in tag or "life" in tag:
            return PrismAxes.BIOLOGY_GREEN
        elif "logic" in tag or "math" in tag or "code" in tag:
            return PrismAxes.LOGIC_YELLOW
        elif "art" in tag or "emotion" in tag:
            return PrismAxes.ART_VIOLET
        
        # Default: Logic (Elysia is software)
        return PrismAxes.LOGIC_YELLOW

    def verify_integration(self, question: str) -> str:
        """
        퇴적된 지식이 실제로 답변(관점)에 영향을 주는지 테스트
        """
        logger.info(f"🧪 Verifying Integration with Question: '{question}'")
        
        # WhyEngine을 사용하여 질문 분석
        # 이때 퇴적된 지식이 '투영'되어야 함
        
        analysis_result = self.why_engine.analyze(subject="Integration Test", content=question, domain="logic")
        
        # 결과 해석
        extraction = analysis_result
        explanation = f"Analysis of '{question}':\n"
        explanation += f"- Principle: {extraction.underlying_principle}\n"
        explanation += f"  Resonance: {extraction.resonance_reactions}\n"
            
        return explanation
