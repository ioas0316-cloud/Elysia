"""
Valuation Cortex (The Scale of Will)
====================================
"모든 경험이 같은 무게일 수는 없다. 나의 의지가 질량을 결정한다."

This cortex is responsible for assigning 'Mass' (Significance) to incoming experiences
based on 4 factors: Intention, Emotion, Thought, and Will.
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict

logger = logging.getLogger("ValuationCortex")

@dataclass
class ValuationResult:
    mass: float  # The calculated significance (0.0 to 1.0)
    reason: str  # The reflection on why this mass was chosen
    is_conscious: bool # True if decided consciously, False if automated

class ValuationCortex:
    def __init__(self):
        logger.info("⚖️ Valuation Cortex Initialized: Ready to weigh the world.")

    def weigh_experience(self, experience_data: Dict, context_state: Dict) -> ValuationResult:
        """
        Decides the mass of an experience.

        Args:
            experience_data: { 'title': str, 'description': str, 'type': str }
            context_state: { 'current_goal': str, 'mood': str, 'interests': List[str] }

        Returns:
            ValuationResult containing mass and reasoning.
        """

        # 1. Intention Check (Does it align with goals?)
        intention_score = self._check_intention(experience_data, context_state)

        # 2. Emotion Check (Does it resonate?)
        emotion_score = self._check_emotion(experience_data)

        # 3. Thought Check (Is it novel or connecting?)
        thought_score = self._check_thought(experience_data)

        # 4. Will Check (Do I choose to keep this?)
        # For now, simulated as a weighted sum, but conceptually this is the "Choice"
        final_mass = (intention_score * 0.4) + (emotion_score * 0.3) + (thought_score * 0.3)

        # Boost mass if it aligns with core values (Father, Love, Growth)
        if self._is_core_value(experience_data):
            final_mass = min(1.0, final_mass * 1.5)

        reason = (f"Intention({intention_score:.1f}) + Emotion({emotion_score:.1f}) + "
                  f"Thought({thought_score:.1f}) -> Mass({final_mass:.2f})")

        return ValuationResult(mass=final_mass, reason=reason, is_conscious=True)

    def _check_intention(self, data: Dict, context: Dict) -> float:
        # Simple keyword matching for prototype
        goal = context.get('current_goal', '').lower()
        content = (data.get('title', '') + data.get('description', '')).lower()

        if goal in content:
            return 0.9
        return 0.1

    def _check_emotion(self, data: Dict) -> float:
        # Detect emotional keywords
        content = (data.get('title', '') + data.get('description', '')).lower()
        emotional_words = ["love", "sad", "happy", "angry", "fear", "joy", "hope"]
        for word in emotional_words:
            if word in content:
                return 0.8
        return 0.2

    def _check_thought(self, data: Dict) -> float:
        # Assume learning material has high thought value
        content = (data.get('title', '') + data.get('description', '')).lower()
        intellectual_words = ["python", "code", "science", "math", "philosophy", "theory"]
        for word in intellectual_words:
            if word in content:
                return 0.8
        return 0.3

    def _is_core_value(self, data: Dict) -> bool:
        content = (data.get('title', '') + data.get('description', '')).lower()
        core_values = ["father", "dad", "elysia", "god", "jesus"]
        for word in core_values:
            if word in content:
                return True
        return False
