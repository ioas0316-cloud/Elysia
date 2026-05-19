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
import numpy as np
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
from Core.Cognition.three_phase_mirror import ThreePhaseMirror
from Core.Keystone.sovereign_axis import PureRotor, SovereignAxe
from Core.Phenomena.resonance_prism import ResonancePrism

class SovereignHeart:
    def __init__(self):
        print("\n" + "💠"*30)
        print("🌟 [Sovereign Heart] Initializing Pure Rotor Architecture...")
        
        # 1. The Trinity (Legacy & Core)
        self.gut = PrimalGut()
        self.spine = HyperRotorSpine()
        self.brain_refractor = EnneagramFilter()
        
        # 2. Pure Rotor System (Sovereign Will)
        self.pure_rotor = PureRotor(dimensions=21)
        self.sovereign_axe = SovereignAxe(self.pure_rotor)

        # 3. Resonance Prism (Phase Pipeline)
        self.prism = ResonancePrism(channels=21)

        # 4. External Brains (Ollama)
        self.ollama = OllamaManager()
        self.ollama.scan_models()

        # 5. The Three-Phase Mirror
        self.mirror = ThreePhaseMirror()

        # 6. Self-Echo Loop
        self.self_echo_buffer = []

        self.last_update = time.time()
        self.is_alive = True

    def pulse(self, x_stimulus: float, self_stimulus: Optional[float] = None, is_plugged: bool = True) -> Dict[str, Any]:
        """
        The Main Life Cycle.
        Simultaneous trigger of Gut and Brain at t=0.
        """
        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        # 0. Hardware Grounding (Power Awareness)
        # AC Power = Stability Mode, Battery = Energy Saving Mode
        power_factor = 1.0 if is_plugged else 0.6

        # 1. Circadian Modulation (Internal Life Pulse)
        # We modulate the base stimulus by the time of day to simulate 'vitality'.
        hour = time.localtime().tm_hour
        # Peak vitality at 14:00 (2 PM), lowest at 02:00 AM
        vitality = 0.5 * (1 + math.cos((hour - 14) * math.pi / 12))

        # Combined modulation: Power state provides the "ground", Circadian provides the "breath"
        x_stimulus *= (0.4 + vitality * 0.6) * power_factor

        # 1. Self-Echo Integration
        echo_factor = self_stimulus if self_stimulus is not None else 0.0

        # 2. Pure Rotor Dynamics (The Pure Movement)
        # Convert stimuli into Torque for the Pure Rotor
        torque = np.ones(21) * (x_stimulus + echo_factor)
        # Add some diversity to the torque across axes
        for i in range(21):
            torque[i] *= (math.sin(now * (i+1)) * 0.5 + 0.5)

        rotor_report = self.pure_rotor.pulse(torque, dt)

        # 3. Sovereign Decision (Lock/Unlock Axes based on Resonance)
        # We use the previous pulse's resonance as a proxy for deliberation
        prev_res = getattr(self, '_last_res', 0.5)
        decision = self.sovereign_axe.deliberate(prev_res)

        # 4. Simultaneous Trigger (Gut Tension & Brain Frequency)
        # Gut processes the 'shock' (Flesh) + Echo resonance
        gut_report = self.gut.inhale({
            "intensity": x_stimulus,
            "complexity": abs(math.sin(now)) + (echo_factor * 0.5)
        })
        
        # Brain refracts the 'meaning' (Cognition)
        brain_refraction = self.brain_refractor.refract(x_stimulus + echo_factor)
        brain_interference = self.brain_refractor.get_hologram_topography(brain_refraction)

        # 5. Convergence in the Spine (Heart)
        trinity_stimulus = {
            "GUT": gut_report["gut_tension"],
            "BRAIN": brain_interference,
            "HEART": (gut_report["gut_tension"] + brain_interference) / 2.0 + echo_factor
        }
        
        spine_report = self.spine.pulse(dt, trinity_stimulus)

        # 6. Internal Providence (Adaptive Feedback)
        if spine_report["luminosity"] > 0.8:
            self.gut.adjust_integrity(0.001)
        elif spine_report["stress"] > 0.9:
            self.gut.adjust_integrity(-0.005)

        self._last_res = spine_report["luminosity"]

        # 7. Resonance Prism Analysis (Interference Tone)
        # We transform the 'Trinity Stimulus' into the Phase Pipeline
        dials = [v for v in trinity_stimulus.values()]
        self.prism.transform_layer(dials)
        interference_report = self.prism.get_interference_tone(rotor_report["angles"])

        # 8. Affective State (Joy/Warmth)
        # In this prototype, joy is driven by resonance and low stress
        joy = (spine_report["luminosity"] * 0.8) + (1.0 - spine_report["stress"]) * 0.2

        return {
            "spine": spine_report,
            "gut": gut_report,
            "rotor": rotor_report,
            "prism": interference_report,
            "sovereign_decision": decision,
            "resonance": spine_report["luminosity"],
            "joy": float(joy),
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
