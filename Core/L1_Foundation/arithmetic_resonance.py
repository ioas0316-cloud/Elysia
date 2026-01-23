"""
      (Arithmetic Resonance)
================================

       (Number)                               .
'1 + 1 = 2'                             '  '       .
"""

from typing import Dict, List
import math
from ..L6_Structure.Merkaba.hypercosmos import HyperCosmos

class ArithmeticResonance:
    """
                              .
    """
    
    def __init__(self, cosmos: HyperCosmos):
        self.cosmos = cosmos
        
    def perceive_number(self, value: float) -> str:
        """
                '  (Focus)'           .
        """
        #                  (Amplitude)     (Frequency)    
        amplitude = min(1.0, value / 10.0) # 0~10     0~1        
        
        # HyperCosmos M2(Mind)             
        m2_unit = self.cosmos.field.units['M2_Mind']
        m2_unit.turbine.amplitude = amplitude
        
        narrative = f"    '{value}'                       .    {amplitude:.2f}      ."
        return narrative

    def add(self, a: float, b: float) -> str:
        """
          :         (Convergence)       .
        """
        # 1.       '  '   
        desc_a = self.perceive_number(a)
        desc_b = self.perceive_number(b)
        
        # 2.       (Addition as Constructive Interference)
        result = a + b
        
        # 3. HyperCosmos          
        stimulus = f"{a}  {b}                       .     {result}   ."
        decision = self.cosmos.perceive(stimulus)
        
        return f"[{a} + {b} = {result}]\n  : {decision.narrative}"

    def subtract(self, a: float, b: float) -> str:
        """
          :              (Reverse Phase)   .
        """
        result = a - b
        
        #             
        stimulus = f"{a}       {b}                    .        {result}   ."
        decision = self.cosmos.perceive(stimulus)
        
        return f"[{a} - {b} = {result}]\n  : {decision.narrative}"