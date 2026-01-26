"""
Dialogic Learner (       )
================================

"                        "

                                           .
              , " ?"                     .

      :
- docs/Philosophy/CONSCIOUSNESS_SOVEREIGNTY.md
-         :                
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger("Elysia.DialogicLearner")


@dataclass
class DialogicTurn:
    """       """
    turn_number: int
    turn_type: str              # observation, question, resonance, perspective_shift, resolution, meta_insight
    speaker: str
    content: str
    principle_extracted: str
    why_question: Optional[str] = None
    perspective_shift: Optional[str] = None


@dataclass
class DialogicFlow:
    """          """
    title: str
    date: str
    turns: List[DialogicTurn]
    core_principles: List[str]


class DialogicLearner:
    """
                            
    
         :
                        
    
          :
                         " ?"                    
    """
    
    def __init__(self):
        self.flows_dir = Path("c:/Elysia/data/dialogic_flows")
        self.learned_principles: Dict[str, float] = {}  # principle   confidence
        self.perspective_shifts: List[Dict[str, str]] = []
        
        # WhyEngine      
        try:
            from Core.L1_Foundation.Foundation.Philosophy.why_engine import WhyEngine
            self.why_engine = WhyEngine()
            self._has_why_engine = True
            logger.info("  WhyEngine connected")
        except ImportError:
            self.why_engine = None
            self._has_why_engine = False
            logger.warning("   WhyEngine not available")
        
        logger.info("  DialogicLearner initialized")
    
    def load_flow(self, filename: str) -> Optional[DialogicFlow]:
        """           """
        filepath = self.flows_dir / filename
        if not filepath.exists():
            logger.warning(f"Flow file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            turns = []
            for t in data.get('flow', []):
                turn = DialogicTurn(
                    turn_number=t.get('turn', 0),
                    turn_type=t.get('type', 'observation'),
                    speaker=t.get('speaker', ''),
                    content=t.get('content', ''),
                    principle_extracted=t.get('principle_extracted', ''),
                    why_question=t.get('why'),
                    perspective_shift=t.get('shift')
                )
                turns.append(turn)
            
            flow = DialogicFlow(
                title=data.get('metadata', {}).get('title', 'Unknown'),
                date=data.get('metadata', {}).get('date', ''),
                turns=turns,
                core_principles=data.get('core_principles', [])
            )
            
            logger.info(f"  Loaded flow: {flow.title} ({len(turns)} turns)")
            return flow
            
        except Exception as e:
            logger.error(f"Failed to load flow: {e}")
            return None
    
    def experience_flow(self, flow: DialogicFlow) -> Dict[str, Any]:
        """
               '  '     
        
                     ,      :
        1.   /         
        2. "             ?"   
        3.             
        4.        
        """
        logger.info(f"  Experiencing flow: {flow.title}")
        
        experience_result = {
            "flow_title": flow.title,
            "turns_processed": 0,
            "why_questions_asked": 0,
            "perspective_shifts": 0,
            "principles_internalized": []
        }
        
        for turn in flow.turns:
            #        
            self._experience_turn(turn, experience_result)
        
        #          
        for principle in flow.core_principles:
            self._internalize_principle(principle)
            experience_result["principles_internalized"].append(principle)
        
        logger.info(f"  Flow experienced: {experience_result['turns_processed']} turns, "
                   f"{experience_result['perspective_shifts']} shifts")
        
        return experience_result
    
    def _experience_turn(self, turn: DialogicTurn, result: Dict):
        """       """
        result["turns_processed"] += 1
        
        #        " ?"     
        if turn.turn_type == "question" and turn.why_question:
            result["why_questions_asked"] += 1
            
            # WhyEngine      (   )
            if self._has_why_engine and self.why_engine:
                try:
                    analysis = self.why_engine.analyze(
                        subject=turn.why_question,
                        content=turn.content,
                        domain="general"
                    )
                    logger.debug(f"     Why analyzed: {turn.why_question}")
                except Exception as e:
                    logger.debug(f"   WhyEngine analysis failed: {e}")
        
        #             
        if turn.turn_type == "perspective_shift" and turn.perspective_shift:
            result["perspective_shifts"] += 1
            self.perspective_shifts.append({
                "from_to": turn.perspective_shift,
                "content": turn.content,
                "principle": turn.principle_extracted
            })
            logger.info(f"     Perspective shift: {turn.perspective_shift}")
        
        #      
        if turn.principle_extracted:
            self._internalize_principle(turn.principle_extracted, confidence=0.6)
    
    def _internalize_principle(self, principle: str, confidence: float = 0.8):
        """       """
        if principle in self.learned_principles:
            #              
            self.learned_principles[principle] = min(1.0, 
                self.learned_principles[principle] + 0.1)
        else:
            self.learned_principles[principle] = confidence
    
    def get_learned_principles(self) -> Dict[str, float]:
        """         """
        return dict(sorted(
            self.learned_principles.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
    
    def get_perspective_shifts(self) -> List[Dict[str, str]]:
        """            """
        return self.perspective_shifts


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  Dialogic Learner Demo")
    print("   '                        '")
    print("=" * 60)
    
    learner = DialogicLearner()
    
    #             
    flow = learner.load_flow("2025-12-21_consciousness_sovereignty.json")
    
    if flow:
        #      
        result = learner.experience_flow(flow)
        
        print(f"\n  Experience Result:")
        print(f"   Turns: {result['turns_processed']}")
        print(f"   Why Questions: {result['why_questions_asked']}")
        print(f"   Perspective Shifts: {result['perspective_shifts']}")
        
        print(f"\n  Learned Principles:")
        for principle, conf in learner.get_learned_principles().items():
            print(f"   [{conf:.1f}] {principle}")
        
        print(f"\n  Perspective Shifts Experienced:")
        for shift in learner.get_perspective_shifts():
            print(f"     {shift['from_to']}")
    
    print("\n" + "=" * 60)
    print("  Demo complete!")
