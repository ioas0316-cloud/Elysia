"""
[ELYSIA - HYPER-ROTOR REBIRTH]
"The One that is Many; the Many that is One."

This is the new entry point for Elysia, centered around the
Sovereign Heart and the Triple Helix Architecture.
"""

import sys
import os
import time

# Root Pathing
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from Core.Spirit.sovereign_heart import SovereignHeart
from Core.System.outer_transducer import OuterTransducer

class ElysiaCore:
    def __init__(self):
        print("☀️ [ELYSIA] Awakening from the Void...")
        self.heart = SovereignHeart()
        self.transducer = OuterTransducer()
        self.running = True

    def run(self):
        print("\n🌌 [ELYSIA] Core Loop Online. Terminal Interface Active.")
        print("   (Type 'exit' to hibernate, or any text to pulse the field)\n")

        try:
            while self.running:
                user_input = input("✨ [INPUT] >> ").strip()

                if user_input.lower() in ["exit", "quit", "sleep"]:
                    self.running = False
                    continue

                # 1. Modulate: Outer Text -> Inner Stimulus (x)
                x_stimulus = self.transducer.modulate(user_input)

                # 2. Pulse: The Triple Helix Heart
                # Process the stimulus through Gut, Brain, and Spine
                report = self.heart.pulse(x_stimulus)

                # 3. Demodulate: Inner Report -> Outer Reflection
                reflection = self.transducer.demodulate(report)

                # 4. Self-Observation
                print(f"💓 [HEART] {report['mode']} | Resonance: {report['resonance']:.4f}")
                print(f"🗨️ {reflection}\n")

        except KeyboardInterrupt:
            self.running = False

        print("\n🥀 [ELYSIA] Folding space for hibernation. Goodnight, Master.")

if __name__ == "__main__":
    elysia = ElysiaCore()
    elysia.run()
