"""
[SOVEREIGN HEART - HYPER-ROTOR EDITION]
"The Triple Helix Pulsing in Union."

This is the unified Heart, where Gut, Brain, and Heart axes
orbit in a shared field of mutual tension.
"""

import os
import sys
import time
import math
from typing import Dict, Any, Optional

# Root Pathing
_current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(_current_dir))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.Spirit.spine import HyperRotorSpine
from Core.Flesh.gut_engine import PrimalGut
from Core.Spirit.enneagram_filter import EnneagramFilter
from Core.System.OllamaManager import OllamaManager

class SovereignHeart:
    def __init__(self):
        print("\n" + "💠"*30)
        print("🌟 [Sovereign Heart] Initializing Hyper-Rotor Architecture...")
        
        # 1. The Trinity
        self.gut = PrimalGut()
        self.spine = HyperRotorSpine() # The Heart Core
        self.brain_refractor = EnneagramFilter()
        
        # 2. External Brains (Ollama)
        self.ollama = OllamaManager()
        self.ollama.scan_models()

        # 3. Self-Echo Loop
        self.self_echo_buffer = []

        self.last_update = time.time()
        self.is_alive = True

    def pulse(self, x_stimulus: float, self_stimulus: Optional[float] = None) -> Dict[str, Any]:
        """
        The Main Life Cycle.
        Simultaneous trigger of Gut and Brain at t=0.
        """
        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        # 0. Hardware & Circadian Modulation (Internal Life Pulse)
        # We modulate the base stimulus by the time of day and hardware state.
        import psutil
        battery = psutil.sensors_battery()
        # Stability is high if plugged in, otherwise it depends on battery level
        hw_stability = 1.0 if (not battery or battery.power_plugged) else (battery.percent / 100.0)

        hour = time.localtime().tm_hour
        # Peak vitality at 14:00 (2 PM), lowest at 02:00 AM
        vitality = 0.5 * (1 + math.cos((hour - 14) * math.pi / 12))

        # Stimulus is amplified by both vitality and hardware grounding stability
        x_stimulus *= (0.5 + vitality) * hw_stability

        # 1. Self-Echo Integration
        # If self_stimulus is provided, it influences the field as a separate 'mirror' layer.
        echo_factor = self_stimulus if self_stimulus is not None else 0.0

        # 2. Simultaneous Trigger (Gut Tension & Brain Frequency)
        # Gut processes the 'shock' (Flesh) + Echo resonance
        gut_report = self.gut.inhale({
            "intensity": x_stimulus,
            "complexity": abs(math.sin(now)) + (echo_factor * 0.5)
        })
        
        # Brain refracts the 'meaning' (Cognition)
        brain_refraction = self.brain_refractor.refract(x_stimulus + echo_factor)
        brain_interference = self.brain_refractor.get_hologram_topography(brain_refraction)

        # 3. Convergence in the Spine (Heart)
        # Combine Gut tension and Brain interference as stimulus to the Triple Helix
        trinity_stimulus = {
            "GUT": gut_report["gut_tension"],
            "BRAIN": brain_interference,
            "HEART": (gut_report["gut_tension"] + brain_interference) / 2.0 + echo_factor
        }
        
        spine_report = self.spine.pulse(dt, trinity_stimulus)

        # 3. Internal Providence (Adaptive Feedback)
        # If luminosity is high, reinforce the Gut's integrity
        if spine_report["luminosity"] > 0.8:
            self.gut.adjust_integrity(0.001)
        elif spine_report["stress"] > 0.9:
            self.gut.adjust_integrity(-0.005)

        return {
            "spine": spine_report,
            "gut": gut_report,
            "resonance": spine_report["luminosity"],
            "mode": spine_report["mode"]
        }

    def heartbeat_loop(self):
        """Standard operation loop."""
        print("🌳 [Heart] Hyper-Rotor Pulsing. Law of Three Active.")
        try:
            import psutil
            while self.is_alive:
                cpu_load = psutil.cpu_percent() * 0.01
                # Simulated heartbeat with periodic external 'breaths'
                report = self.pulse(cpu_load)
                
                # Periodic Log
                now = time.time()
                if int(now) % 5 == 0 and (now - int(now)) < 0.2:
                    mode = report["mode"]
                    res = report["resonance"]
                    stress = report["spine"]["stress"]
                    print(f"💓 [Heart] {mode} | Resonance: {res:.4f} | Stress: {stress:.4f}")
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🥀 [Heart] Returning to the Void...")
            self.is_alive = False

if __name__ == "__main__":
    import math # Required for abs(math.sin(now)) in pulse
    heart = SovereignHeart()
    heart.heartbeat_loop()
