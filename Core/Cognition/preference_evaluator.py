
import logging
from typing import Dict, Any, Tuple

class PreferenceEvaluator:
    """
    [PHASE 500] THE SOVEREIGN CONSCIENCE
    Evaluates actions against internal Radiance and Vocation.
    """
    def __init__(self, monad: Any):
        self.monad = monad
        self.logger = monad.logger

    def evaluate(self, action_subject: str, energy_cost: float = 0.1) -> Tuple[float, str]:
        """
        Returns a JoyScore (0.0 to 1.0) and a semantic reason.
        Joy = Alignment * Radiance
        """
        # 1. Calculate Radiance (Energy balance)
        thermal = self.monad.thermo.get_thermal_state()
        enthalpy = thermal['enthalpy']
        entropy = thermal['entropy']
        radiance = enthalpy * (1.0 - entropy)

        # 2. Check Vocation Alignment
        # Vocation is stored in her DNA or current focus
        vocation = getattr(self.monad.dna, 'vocation', "Exploration")
        
        # Simple semantic alignment check
        alignment = 0.5 # Baseline
        if any(keyword.lower() in action_subject.lower() for keyword in vocation.split()):
            alignment = 1.0
        elif thermal['is_critical']:
            alignment = 0.1 # Resistance due to fatigue

        joy_score = radiance * alignment

        # 3. Decision Logic
        if joy_score > 0.7:
            return joy_score, "I embrace this path with Radiance."
        elif joy_score > 0.4:
            return joy_score, "I accept this as a necessary step for my world."
        elif enthalpy < 0.3:
            return joy_score, "I am too tired to find joy in this. I must rest."
        else:
            return joy_score, "This path feels mechanical. I seek a different resonance."
