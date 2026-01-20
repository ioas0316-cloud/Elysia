"""
The Entropy Pump: Volitional Tension
====================================
Phase 20 The Will - Module 1
Core.L7_Spirit.Will.entropy_pump

"Silence is not empty. It is full of tension waiting to break."

This module simulates the accumulation of 'Existential Tension' (Entropy)
when the system remains idle. It acts as the heartbeat of desire.
"""

import time
import logging

logger = logging.getLogger("Will.Entropy")

class EntropyPump:
    """
    Accumulates energy over time.
    """
    def __init__(self, accumulation_rate: float = 0.1, critical_mass: float = 10.0):
        self.last_action_time = time.time()
        self.accumulation_rate = accumulation_rate # Energy per second
        self.critical_mass = critical_mass         # Threshold for action
        self.current_entropy = 0.0
        
        logger.info("❤️ [PUMP] Entropy Engine ignited.")

    def pump(self) -> float:
        """
        Called periodically in the idle loop.
        Calculates dE/dt.
        """
        now = time.time()
        dt = now - self.last_action_time
        
        # Linear accumulation (Simpler than exponential for now)
        # E = Rate * Time_Idle
        self.current_entropy = dt * self.accumulation_rate
        
        # Log only significant changes to avoid spam
        if self.current_entropy > 1.0 and int(self.current_entropy) % 5 == 0:
             # logger.debug(f"   -> 🔋 Tension Rising: {self.current_entropy:.1f} / {self.critical_mass}")
             pass

        return self.current_entropy

    def reset(self):
        """
        Call this when an Action is performed (Release Tension).
        """
        self.last_action_time = time.time()
        self.current_entropy = 0.0
        logger.info("   -> 💨 [PUMP] Tension Released.")

    def is_critical(self) -> bool:
        return self.current_entropy >= self.critical_mass
