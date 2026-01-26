"""
Knowledge Sedimenter (주권적 자아)
==================================

                   '      '      .
                    ,        4D Prism       
          "      (Sediment)"     .

     :
1. Search (  ): BrowserExplorer          
2. Distill (  ): WhyEngine          (Principle)   
3. Crystallize (   ): 4D LightSpectrum      (Scale/Basis   )
4. Deposit (  ): LightSediment         (Viewpoint)   
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from Core.L2_Metabolism.Physiology.Sensory.Network.browser_explorer import BrowserExplorer
from Core.L7_Spirit.Philosophy.why_engine import WhyEngine
from Core.L6_Structure.Wave.light_spectrum import LightSpectrum, LightSediment, PrismAxes

logger = logging.getLogger("Elysia.KnowledgeSedimenter")

@dataclass
class SedimentationHypothesis:
    """        (     )"""
    topic: str
    expected_layer: PrismAxes
    initial_question: str

class KnowledgeSedimenter:
    """
           (The Knowledge Sedimenter)
    
    "      ,       ."
    """
    
    def __init__(self, why_engine: WhyEngine):
        self.why_engine = why_engine
        self.browser = BrowserExplorer(use_profile=True) #             
        
        logger.info("  KnowledgeSedimenter initialized -                  ")

    def sediment_from_web(self, topic: str, max_pages: int = 1) -> List[LightSpectrum]:
        """
                           
        """
        logger.info(f"  Exploring Web for Sedimentation: '{topic}'")
        
        # 1. Search
        search_results = self.browser.google_search(topic)
        if not search_results["success"]:
            logger.warning("     Search failed.")
            return []
            
        collected_lights = []
        
        # 2. Iterate & Process
        for res in search_results["results"][:max_pages]:
            title = res.get("title", "")
            snippet = res.get("snippet", "")
            full_content = f"{title}\n{snippet}" #                           
            
            logger.info(f"     Processing: {title}")
            
            # 3. Distill & Crystallize (Principal Extraction)
            # Use Dimensional Reasoner to Lift Knowledge from 0D to 4D
            try:
                from Core.L5_Mental.Intelligence.Reasoning.dimensional_reasoner import DimensionalReasoner
                lifter = DimensionalReasoner()
                
                # The 'contemplate' method acts as the pipeline:
                # 0D (Fact) -> 1D (Logic) -> 2D (Context) -> 3D (Paradox) -> 4D (Law)
                hyper_thought = lifter.contemplate(full_content)
                
                # Check if a Law (4D) was crystallized
                if hyper_thought.d4_principle and "Law synthesis error" not in hyper_thought.d4_principle:
                    logger.info(f"     CRYSTALLIZED LAW: {hyper_thought.d4_principle}")
                
            except Exception as e:
                logger.error(f"     Dimensional Lifting failed: {e}")

            # Legacy Light Creation (for visualization)
            analysis = self.why_engine.light_universe.absorb(full_content, tag=topic)
            
            # 4D Basis       (Naive)
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
            logger.info(f"     Deposited into {target_axis.name} (x{repetition}): Final Amp={final_amp:.3f}")
        return collected_lights

    def _determine_axis(self, light: LightSpectrum) -> PrismAxes:
        """
                            
        (     :       +    )
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
                      (  )             
        """
        logger.info(f"  Verifying Integration with Question: '{question}'")
        
        # WhyEngine            
        #            '  '     
        
        analysis_result = self.why_engine.analyze(subject="Integration Test", content=question, domain="logic")
        
        #      
        extraction = analysis_result
        explanation = f"Analysis of '{question}':\n"
        explanation += f"- Principle: {extraction.underlying_principle}\n"
        explanation += f"  Resonance: {extraction.resonance_reactions}\n"
            
        return explanation
