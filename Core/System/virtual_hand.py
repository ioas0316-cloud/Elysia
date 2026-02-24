"""
[BODY] Virtual Hand: Motor Output Simulator
==========================================
Location: Scripts/System/Body/virtual_hand.py

Role:
- Simulates a Physical Keyboard (Output Device).
- Accepts 'Motor Impulses' (0-255).
- Updates a 'Virtual Screen' (Text Buffer).
- Safe Sandbox for Motor Babbling.
"""

import time
import random

class VirtualKeyboard:
    def __init__(self):
        self.screen_buffer = ""
        # Hidden wiring: The 'Physical' truth of the simulator
        # The AI doesn't know this map initially. It must learn it.
        self._hardware_map = {
            # Standard ASCII (Simplified)
            65: 'A', 66: 'B', 67: 'C', 68: 'D', 69: 'E',
            70: 'F', 71: 'G', 72: 'H', 73: 'I', 74: 'J',
            75: 'K', 76: 'L', 77: 'M', 78: 'N', 79: 'O',
            80: 'P', 81: 'Q', 82: 'R', 83: 'S', 84: 'T',
            85: 'U', 86: 'V', 87: 'W', 88: 'X', 89: 'Y', 90: 'Z',
            32: ' ' # Space
        }
        
    def receive_impulse(self, signal_code: int):
        """
        The Muscle Actuator.
        signal_code: The motor command (0-255) sent by the brain.
        """
        # Simulate physical mechanic
        char = self._hardware_map.get(signal_code, '?') # ? = Miss/Noise
        
        if char != '?':
            self.screen_buffer += char
        else:
            pass # Did nothing (Weak signal)
            
    def get_visual_feedback(self) -> str:
        """
        The Eye sees the screen.
        """
        return self.screen_buffer

    def clear_screen(self):
        self.screen_buffer = ""

class MotorCortex:
    def __init__(self, hand: VirtualKeyboard):
        self.hand = hand
        # Synaptic Weights: Mapping Intention (Char) -> Motor Code (Int)
        # Initially Empty (Tabula Rasa)
        self.synapses = {} 
        
    def babble(self) -> int:
        """
        Random Motor Noise (Exploration).
        Returns the signal code sent.
        """
        code = random.randint(32, 90) # Constraint search space for demo speed
        self.hand.receive_impulse(code)
        return code
        
    def intend_and_act(self, target_char: str) -> bool:
        """
        Exploitation: Use learned synapse to type.
        """
        if target_char in self.synapses:
            code = self.synapses[target_char]
            self.hand.receive_impulse(code)
            return True # Confident Action
        else:
            return False # Don't know how

    def learn(self, signal_code: int, observed_char: str):
        """
        Reinforcement: "I fired X, and saw Y. Link X->Y."
        """
        if observed_char and observed_char != '?':
            # Hebbian Learning: Fire together, wire together
            self.synapses[observed_char] = signal_code
            # print(f"🧠 [Synapse] Learned: Intent('{observed_char}') requires Motor({signal_code})")

if __name__ == "__main__":
    # Unit Test
    kb = VirtualKeyboard()
    brain = MotorCortex(kb)
    
    print("👶 [TEST] Babbling Phase...")
    for i in range(5):
        sig = brain.babble()
        print(f"  Sent Signal {sig} -> Screen: '{kb.get_visual_feedback()}'")
