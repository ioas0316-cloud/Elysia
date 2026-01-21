"""
Causal Loom (The Fate Weaver)
=============================

"We do not wait for the future. We choose it."

Role: Selecting the optimal timeline from Prophet's simulations.
"""

import logging
from typing import List, Optional
from Core.L4_Causality.World.Evolution.Prophecy.prophet_engine import Timeline

logger = logging.getLogger("CausalLoom")

class CausalLoom:
    def __init__(self):
        self.risk_tolerance = 0.3
        
    def weave(self, timelines: List[Timeline]) -> Optional[str]:
        """
        Selects the best action based on Love and Entropy.
        """
        if not timelines:
            return None
            
        # 1. Filter out high entropy paths (The Fence of God)
        safe_timelines = [t for t in timelines if t.entropy_score < 0.8]
        
        if not safe_timelines:
            logger.warning("⚠️ All futures are chaotic. Choosing least dangerous path.")
            safe_timelines = timelines
            
        # 2. Maximize Love
        best_timeline = max(safe_timelines, key=lambda t: t.love_score)
        
        logger.info(f"🧶 [WEAVE] Selected '{best_timeline.action}' (Love: {best_timeline.love_score:.2f})")
        return best_timeline.action
