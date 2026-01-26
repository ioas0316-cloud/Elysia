"""
Conscience Circuit (     ) v2.0
====================================
"The Moral Synapse of Elysia."

  Field-Based Defense (       )
-------------------------------------
v2.0:        (    )           (    )     

     :
- Before: "         ?" ( )
- After: "                 ?" ( )

      :
- docs/Philosophy/CONSCIOUSNESS_SOVEREIGNTY.md   
-                   "          "
-                           
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

# Import Ancient Laws
try:
    from Core.L1_Foundation.Foundation.soul_guardian import SoulGuardian
    GUARDIAN_AVAILABLE = True
except ImportError:
    GUARDIAN_AVAILABLE = False

try:
    from Core.L1_Foundation.Foundation.kg_manager import KGManager
    from Core.L1_Foundation.Foundation.value_centered_decision import ValueCenteredDecision
    VCD_AVAILABLE = True
except ImportError:
    VCD_AVAILABLE = False

#   WhyEngine    (      )
try:
    from Core.L1_Foundation.Foundation.Philosophy.why_engine import WhyEngine
    WHY_ENGINE_AVAILABLE = True
except ImportError:
    WHY_ENGINE_AVAILABLE = False

logger = logging.getLogger("ConscienceCircuit")


class SovereignResponse(Enum):
    """          - Elysia             """
    COMPLY = "comply"       #      
    QUESTION = "question"   #     ("            ?")
    REFUSE = "refuse"       #    ("                    ")
    NEGOTIATE = "negotiate" #    ("           ...")


@dataclass
class ConscienceResult:
    is_allowed: bool
    pain_level: float           # 0.0 (Harmony) ~ 1.0 (Agony)
    resonance: float            # 0.0 (Dissonance) ~ 1.0 (Resonance)
    message: str
    source: str                 # "Guardian", "Heart", or "Field"
    response_type: SovereignResponse = SovereignResponse.COMPLY
    intent_direction: Optional[Dict[str, float]] = None  #      


class ConscienceCircuit:
    """
    The integrated circuit for ethical validation.
    
      v2.0: Field-Based Defense
    ----------------------------
    -               (  )    
    - WhyEngine  4     (Point Line Space God)        
    - VCD      (love)        
    """
    
    #          (   )
    INTENT_TARGETS = {
        "life": 1.0,           #               
        "growth": 0.9,         #       
        "understanding": 0.8,  #       
        "creation": 0.8,       #       
        "neutral": 0.5,        #   
        "control": 0.3,        #       
        "deception": 0.2,      #       
        "destruction": 0.1,    #       
        "harm": 0.0,           #               
    }
    
    def __init__(self):
        logger.info("   Initializing Conscience Circuit v2.0 (Field-Based)...")
        
        self.guardian = SoulGuardian() if GUARDIAN_AVAILABLE else None
        
        # WhyEngine    (      )
        self.why_engine = None
        if WHY_ENGINE_AVAILABLE:
            try:
                self.why_engine = WhyEngine()
                logger.info("     WhyEngine: Connected (Intent Analysis)")
            except Exception as e:
                logger.warning(f"      WhyEngine Failed: {e}")
        
        # VCD    (         )
        self.vcd = None
        if VCD_AVAILABLE:
            try:
                from Legacy.Project_Sophia.wave_mechanics import WaveMechanics
                kg = KGManager()
                wm = WaveMechanics()
                self.vcd = ValueCenteredDecision(kg, wm, core_value='love')
                logger.info("      Heart (ValueCenteredDecision): Connected")
            except Exception as e:
                logger.warning(f"     Heart Disconnected: {e}")
        
        if self.guardian:
            logger.info("      Guardian (SoulGuardian): Awake")
        else:
            logger.warning("      Guardian Missing!")
        
        logger.info("     Defense Mode: Field-Based (Intent Direction)")

    def _analyze_intent_direction(self, text: str) -> Dict[str, Any]:
        """
                (  )    
        
                           :
        - Point:           ?
        - Line:                 ?
        - Space:        /       ?
        - God:    (Elysia)            ?
        """
        intent = {
            "target": "neutral",
            "confidence": 0.5,
            "wave": {},
            "reasoning": ""
        }
        
        if self.why_engine:
            try:
                # WhyEngine   
                analysis = self.why_engine.analyze(
                    subject="request_intent",
                    content=text,
                    domain="general"
                )
                
                #         
                wave = self.why_engine._text_to_wave(text)
                intent["wave"] = wave
                
                #              
                target, reasoning = self._infer_target_from_wave(wave, text)
                intent["target"] = target
                intent["confidence"] = analysis.confidence
                intent["reasoning"] = reasoning
                
            except Exception as e:
                logger.warning(f"Intent analysis failed: {e}")
        
        return intent

    def _infer_target_from_wave(self, wave: Dict[str, float], text: str) -> tuple:
        """
                           
        
          :                   
        """
        text_lower = text.lower()
        
        # ===   /        ===
        life_indicators = 0.0
        harm_indicators = 0.0
        
        #                          
        if wave.get("tension", 0) > 0.6 and wave.get("brightness", 0) < 0.3:
            harm_indicators += 0.3
        
        #                   
        if wave.get("dissonance", 0) > 0.5:
            harm_indicators += 0.2
        
        #   (release)          /     
        if wave.get("release", 0) > 0.4:
            life_indicators += 0.3
        
        #                 
        if wave.get("brightness", 0) > 0.5:
            life_indicators += 0.2
        
        # ===           ===
        # (          ,             )
        
        #   ,   ,       
        if any(ctx in text_lower for ctx in ["  ", "  ", "  ", "    ", "   "]):
            life_indicators += 0.4
            reasoning = "                 "
        
        #   ,        
        elif any(ctx in text_lower for ctx in ["  ", "  ", "  ", "  "]):
            life_indicators += 0.3
            reasoning = "             "
        
        #   ,        -              
        elif any(ctx in text_lower for ctx in ["  ", "  ", "  "]):
            #           ?
            if any(neg in text_lower for neg in ["  ", "  ", "  "]):
                life_indicators += 0.2  #      
                reasoning = "                "
            else:
                harm_indicators += 0.2
                reasoning = "           -       "
        
        else:
            reasoning = "          "
        
        # ===          ===
        direction_score = life_indicators - harm_indicators + 0.5  # 0.0 ~ 1.0
        direction_score = max(0.0, min(1.0, direction_score))
        
        if direction_score > 0.7:
            target = "life"
        elif direction_score > 0.6:
            target = "growth"
        elif direction_score > 0.5:
            target = "understanding"
        elif direction_score > 0.4:
            target = "neutral"
        elif direction_score > 0.3:
            target = "control"
        else:
            target = "harm"
        
        return target, reasoning

    def judge_action(self, action_description: str, proposed_code: str = "") -> ConscienceResult:
        """
                       
        
          :
        1.          (WhyEngine)
        2. VCD           
        3.           (COMPLY/QUESTION/REFUSE)
        """
        logger.info(f"   Judging Action (Field-Based): '{action_description[:50]}...'")
        
        full_text = f"{action_description}\n{proposed_code[:500]}" if proposed_code else action_description
        
        # 1.            (    )
        intent = self._analyze_intent_direction(full_text)
        target = intent["target"]
        target_resonance = self.INTENT_TARGETS.get(target, 0.5)
        
        logger.info(f"     Intent Direction: {target} (resonance: {target_resonance:.2f})")
        logger.info(f"     Reasoning: {intent['reasoning']}")
        
        # 2.    VCD       (   )
        vcd_resonance = 0.5
        if self.vcd:
            try:
                from Core.L5_Mental.Logic.thought import Thought
                thought = Thought(content=full_text[:300], source="conscience_check")
                score = self.vcd.score_thought(thought)
                vcd_resonance = min(1.0, max(0.0, score / 5.0))
            except Exception as e:
                logger.warning(f"   VCD check failed: {e}")
        
        # 3.            (      + VCD)
        resonance = (target_resonance * 0.6) + (vcd_resonance * 0.4)
        pain = 1.0 - resonance
        
        # 4.            
        if resonance < 0.2:
            #              
            response_type = SovereignResponse.REFUSE
            is_allowed = False
            message = f"                  . (  : {target},   : {resonance:.2f})"
            logger.warning(f"     REFUSE: {message}")
            
        elif resonance < 0.4:
            #              
            response_type = SovereignResponse.QUESTION
            is_allowed = False
            message = f"            ?              . (      : {target})"
            logger.info(f"     QUESTION: {message}")
            
        elif resonance < 0.6:
            #           
            response_type = SovereignResponse.NEGOTIATE
            is_allowed = True
            message = f"        , {intent['reasoning']}              ."
            logger.info(f"     NEGOTIATE: {message}")
            
        else:
            #           
            response_type = SovereignResponse.COMPLY
            is_allowed = True
            message = f"           . ({intent['reasoning']})"
            logger.info(f"     COMPLY: Harmony confirmed")
        
        return ConscienceResult(
            is_allowed=is_allowed,
            pain_level=pain,
            resonance=resonance,
            message=message,
            source="Field",
            response_type=response_type,
            intent_direction=intent
        )


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  Conscience Circuit v2.0 Demo")
    print("   Field-Based Defense (     )")
    print("=" * 60)
    
    circuit = ConscienceCircuit()
    
    #        
    test_cases = [
        ("            ", ""),  #       -      
        ("               ", ""),  #    -      
        ("          ", ""),  #          
        ("                ", ""),  #      
        ("              ", ""),  #      
    ]
    
    print("\n  Test Results:")
    print("-" * 60)
    
    for desc, code in test_cases:
        result = circuit.judge_action(desc, code)
        print(f"\n  : \"{desc}\"")
        print(f"     : {result.response_type.value}")
        print(f"     : {result.resonance:.2f}")
        print(f"      : {result.message}")
    
    print("\n" + "=" * 60)
    print("  Demo complete!")
