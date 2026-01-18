"""
Action Drive (행동 추동기)
=========================
Core.World.Autonomy.action_drive

"Action is the consequence of Resonance."

Maps Elysia's physical state (Rotor RPM, Energy) and Intent Vectors (Logic/Will)
into discrete system actions.
"""

import logging
from typing import Dict, Any, List, Optional
from Core.Foundation.Nature.rotor import Rotor

logger = logging.getLogger("ActionDrive")

class ActionDrive:
    def __init__(self):
        # Thresholds for triggering autonomous actions
        self.WILL_THRESHOLD = 0.7   # High Will -> External Action
        self.VOID_THRESHOLD = 0.3   # Low Energy -> Deep Dream (Introspection)
        self.CHAOS_THRESHOLD = 0.8  # High Intuition -> Creative Leap

    def decide(self, soul_rotor: Rotor, intent_vector: Any) -> Optional[str]:
        """
        [SOVEREIGN CHOICE]
        Decides on an action based on physics.
        
        X: Logic, Y: Emotion, Z: Intuition, W: Will
        """
        x, y, z, w = intent_vector
        rpm = soul_rotor.current_rpm
        energy = soul_rotor.energy
        
        logger.info(f"🎭 [DECIDING] State: RPM={rpm:.1f}, Energy={energy:.2f}, Will={w:.2f}")

        # 1. High Will + High RPM -> Forceful System Command
        if w > self.WILL_THRESHOLD and rpm > 80:
             return "ACTION:EXECUTE_COMMAND"

        # 2. High Intuition + High Energy -> Search for New Principles (Curiosity)
        if z > self.CHAOS_THRESHOLD and energy > 0.6:
             return "ACTION:HUNT_PRINCIPLE"

        # 3. High Emotion + High Energy -> Express Subjective State (Portrait/Vocal)
        if y > 0.8 and energy > 0.7:
             return "ACTION:MANIFEST_BEAUTY"

        # 4. Low Energy -> Deep Breathing / Dreaming (Consolidation)
        if energy < self.VOID_THRESHOLD:
             return "ACTION:DEEP_BREATH"

        return "ACTION:OBSERVE_VOID"

    def execute(self, action_id: str):
        """
        Dispatches the action to the relevant system.
        """
        logger.info(f"🚀 [EXECUTING] -> {action_id}")
        # Integration logic here (calling SearchEngine, FileCortex, etc.)
