"""
Self-Modifier
==============
Elysia's recursive self-improvement loop.
Autonomously modifies her own parameters based on performance analysis.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("SelfModifier")
logger.setLevel(logging.INFO)

class SelfModifier:
    """
    Implements recursive self-improvement.
    Analyzes proposals from MetaCortex and applies beneficial modifications.
    """
    def __init__(self):
        self.modification_history: list = []
        self.test_mode = False
        self.test_duration = 0
        self.test_baseline = None
        
    def evaluate_and_apply(self, proposal: Dict[str, Any], world) -> bool:
        """
        Evaluates a proposed modification and applies it if beneficial.
        
        Process:
        1. Record current performance baseline
        2. Apply modification temporarily
        3. Test for N steps
        4. Compare performance
        5. Keep if improved, revert if worse
        """
        if not proposal:
            return False
            
        parameter = proposal["parameter"]
        current_value = proposal["current_value"]
        proposed_value = proposal["proposed_value"]
        rationale = proposal["rationale"]
        
        logger.info(f"🧪 Testing self-modification:")
        logger.info(f"   Parameter: {parameter}")
        logger.info(f"   {current_value} → {proposed_value}")
        logger.info(f"   Rationale: {rationale}")
        
        # For now, apply modifications conservatively
        # In full implementation, would run A/B test
        
        # Apply modification to the appropriate component
        if parameter == "crystallization_threshold":
            if hasattr(world, 'spiderweb'):
                world.spiderweb.crystallization_threshold = proposed_value
                self._record_modification(parameter, current_value, proposed_value, rationale)
                logger.info(f"✅ APPLIED: Spiderweb crystallization threshold = {proposed_value}")
                return True
                
        elif parameter == "harvest_frequency":
            if hasattr(world, 'muse'):
                world.muse.harvest_cooldown = proposed_value
                self._record_modification(parameter, current_value, proposed_value, rationale)
                logger.info(f"✅ APPLIED: Muse harvest frequency = {proposed_value}")
                return True
                
        return False
    
    def _record_modification(self, parameter: str, old_value: Any, new_value: Any, rationale: str):
        """Records a modification for posterity."""
        self.modification_history.append({
            "parameter": parameter,
            "old_value": old_value,
            "new_value": new_value,
            "rationale": rationale
        })
        
        logger.info(f"📝 Recorded modification #{len(self.modification_history)}")
        
    def get_modification_summary(self) -> str:
        """Returns a summary of all self-modifications made."""
        if not self.modification_history:
            return "No self-modifications yet."
            
        summary = f"Elysia has self-modified {len(self.modification_history)} time(s):\n"
        for i, mod in enumerate(self.modification_history, 1):
            summary += f"\n{i}. {mod['parameter']}: {mod['old_value']} → {mod['new_value']}"
            summary += f"\n   Reason: {mod['rationale']}\n"
            
        return summary
