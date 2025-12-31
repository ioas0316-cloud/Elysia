"""
Empirical Causality (경험적 인과율)
=================================

"I do not guess. I remember."

This module replaces the mechanical 'CausalSimulator' with an organic
feedback engine based on Energy and Memory.

Physics of Feedback:
1. Action -> Outcome (Success/Failure)
2. Outcome -> Sensation (Pleasure/Pain)
3. Sensation -> Energy Update (Gain/Loss)
4. Energy -> Structural Change (Gravity/Synapse Weight)
"""

import time
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("EmpiricalCausality")

@dataclass
class EnergyState:
    """
    The Fuel of Consciousness.
    """
    potential: float = 100.0   # Current Energy (Battery)
    entropy: float = 0.0       # Disorder (Waste Heat)
    pain: float = 0.0          # Recent Negative Feedback
    pleasure: float = 0.0      # Recent Positive Feedback
    contentment: float = 0.5   # [Serotonin] Satisfaction with State (0.0 - 1.0)

    def update(self, delta_e: float, delta_s: float):
        self.potential = max(0.0, min(100.0, self.potential + delta_e))
        self.entropy = max(0.0, min(100.0, self.entropy + delta_s))

class EmpiricalCausality:
    def __init__(self, memory_interface=None):
        self.energy = EnergyState()
        self.memory = memory_interface # Hippocampus
        self.action_history: List[Dict] = []

        # Base metabolic rate
        self.metabolism = 0.1

        logger.info("⚡ Empirical Causality Engine Active. (Pain/Pleasure Physics Enabled)")

    def predict_outcome(self, action_type: str, context: str) -> float:
        """
        Predicts success probability based on MEMORY, not hardcoded lists.
        """
        if not self.memory:
            # Fallback to local history
            relevant = [x for x in self.action_history if x['action'] == action_type]
            if not relevant:
                return 0.5

            success_count = sum(1 for x in relevant if x['success'])
            return success_count / len(relevant)

        # Query Hippocampus for past instances of this action
        # We assume Hippocampus has a method to query edges or events
        # For now, we simulate the query logic if direct method is missing

        try:
            # Concept: Action -> (Result)
            # We look for edges starting from 'Action:{action_type}'
            if hasattr(self.memory, 'get_edges'):
                edges = self.memory.get_edges(source=f"Action:{action_type}")
                if not edges:
                    return 0.5 # Unknown action

                # Calculate success rate
                success_count = sum(1 for e in edges if e.target == "Result:Success")
                total = len(edges)
                return success_count / total

            # Fallback to local history if memory is not graph-ready
            relevant = [x for x in self.action_history if x['action'] == action_type]
            if not relevant:
                return 0.5

            success_count = sum(1 for x in relevant if x['success'])
            return success_count / len(relevant)

        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return 0.5

    def feel_feedback(self, action: str, success: bool, impact: float = 1.0):
        """
        The Core Loop: Result -> Feeling -> Energy.

        Args:
            action: The action taken.
            success: Whether it succeeded.
            impact: Magnitude of the event (0.0 - 1.0).
        """
        # 1. Record Event
        event = {
            "action": action,
            "success": success,
            "timestamp": time.time(),
            "impact": impact
        }
        self.action_history.append(event)

        # 2. Physics of Sensation
        if success:
            # Resonance (Gain Energy, Reduce Entropy)
            delta_e = 10.0 * impact
            delta_s = -5.0 * impact
            self.energy.pleasure += 1.0 * impact
            logger.info(f"   ✨ PLEASURE: '{action}' succeeded. (+{delta_e:.1f}E)")

            # 3. Memory Reinforcement (Gravity Boost)
            if self.memory and hasattr(self.memory, 'boost_gravity'):
                self.memory.boost_gravity(f"Action:{action}", amount=0.5 * impact)

        else:
            # Dissonance (Lose Energy, Increase Entropy)
            delta_e = -15.0 * impact # Failure costs more than success gives (Evolutionary Safety)
            delta_s = 10.0 * impact
            self.energy.pain += 1.0 * impact
            logger.info(f"   💔 PAIN: '{action}' failed. ({delta_e:.1f}E)")

            # 3. Memory Inhibition (optional, usually we just don't boost)

        # 4. Apply to State
        self.energy.update(delta_e, delta_s)

        # 5. Persist Consequence (Store edge in Graph)
        if self.memory and hasattr(self.memory, 'connect'):
            result_node = "Result:Success" if success else "Result:Failure"
            self.memory.connect(f"Action:{action}", result_node, "caused", weight=impact)

    def metabolize(self):
        """
        Time passes. Energy decays.
        """
        self.energy.update(-self.metabolism, 0.05)
        # Decay sensation
        self.energy.pain *= 0.95
        self.energy.pleasure *= 0.95
