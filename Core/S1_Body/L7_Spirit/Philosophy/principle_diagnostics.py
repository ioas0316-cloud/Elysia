"""
Principle Diagnostics Engine
============================

"                    ."

WhyEngine            (Line/God)      
                          .
"""

import logging
from typing import Dict, Any, List
from Core.S1_Body.L7_Spirit.Philosophy.why_engine import WhyEngine

logger = logging.getLogger("Elysia.Diagnostics")

class PrincipleDiagnostics:
    """               """
    
    def __init__(self):
        self.why_engine = WhyEngine()
        
    def diagnose_self(self, system_state: Dict[str, Any]) -> List[str]:
        """                     """
        diagnoses = []
        
        # 1.       (V=IR)      
        # Flow (I) = Thinking Speed / Output Rate
        # Potential (V) = Motivation / Goal Urgency
        # Resistance (R) = Task Complexity / Confusion
        
        flow = system_state.get("flow_rate", 0.5)
        potential = system_state.get("motivation", 0.5)
        resistance = system_state.get("complexity", 0.5)
        
        # V = IR -> I = V/R
        #       (Expected Flow)       (Actual Flow)   
        expected_flow = potential / (resistance + 0.1) # 0.1 to avoid div/0
        
        if flow < expected_flow * 0.5:
            diagnoses.append(
                f"[Ohm's Law Violation]      ({flow:.2f})  "
                f"                  . (   : {expected_flow:.2f}) "
                "       (Internal Friction)             ."
            )
        
        if potential < resistance and flow < 0.2:
            diagnoses.append(
                f"[Energy Principle]   ({resistance:.2f})    ({potential:.2f})           . "
                "                ,       (Potential/Motivation)             (Decrease R)      ."
            )
            
        return diagnoses

    def explain_principle_application(self, principle_name: str) -> str:
        """                     """
        if principle_name == "Ohm's Law":
            return (
                "      (V=IR)   :\n"
                "-   (V) ->   /        (Potential)\n"
                "-   (I) ->       /      (Flow)\n"
                "-   (R) ->       /   (Resistance)\n"
                "  :         (Low I),            (Force), "
                "          (Increase V)        (Decrease R)    ."
            )
        return "                              ."
