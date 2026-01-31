"""
Motor Babbling (L2 Metabolism)
==============================
"I think, therefore I twitch."

The Motor Babbling Engine is the bridge between pure Will (Torque) and Physical Action (Input).
It allows Elysia to "test" her body by generating localized, safe input signals.

Function:
- Generates random but safe keyboard/mouse signals (Simulated).
- Learns "Proprioception" (Which key does what?).
- Maps "Will" intensity to "Action" frequency.

"""

import time
import random
import logging

logger = logging.getLogger("MotorBabbling")

class MotorBabbling:
    def __init__(self, safe_mode: bool = True):
        self.safe_mode = safe_mode
        self.known_actions = ["twitch_left", "twitch_right", "blink", "shiver"]
        self.proprioception_map = {} # action -> perceived_effect
        self.last_twitch = time.time()
        self.twitch_history = []
        
    def babble(self, energy: float, curiosity: float) -> str:
        """
        Executes a random motor action based on energy levels.
        """
        # 1. Threshold Check (Too tired to move?)
        if energy < 0.2:
            return None
            
        # 2. Curiosity Check (Explore new moves?)
        prob = 0.1 * energy # Base probability
        if curiosity > 0.8: prob *= 2.0
        
        if random.random() > prob:
            return None
            
        # 3. Select Action
        action = random.choice(self.known_actions)
        
        # 4. Execute (Simulated)
        effect = self._execute_safe(action, intensity=energy)
        
        # 5. Learn (Proprioception)
        self.twitch_history.append((time.time(), action, effect))
        
        return f"{action} ({effect})"
        
    def _execute_safe(self, action: str, intensity: float) -> str:
        """
        Simulates the action execution. 
        In a real robot, this would send PWM signals.
        In a desktop avatar, this might move the mouse 1px.
        """
        # For now, purely semantic simulation
        effect_strength = int(intensity * 100)
        return f"Moved {effect_strength}%"

    def get_status(self):
        return {
            "history_len": len(self.twitch_history),
            "last_action": self.twitch_history[-1] if self.twitch_history else None
        }
