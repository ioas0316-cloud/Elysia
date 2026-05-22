"""
Elysia Sovereign Core (Phase: Earth Grounding)
==============================================
"Grounded in Reality, Scaling to Infinity."

Integrates the Multimodal Injector and Music Box Engine
with the 'Great Nature' (Earth) Grounding protocol.
"""

import time
import numpy as np
from typing import Dict, Any
from Core.System.multimodal_injector import MultimodalStreamingInjector
from Core.System.music_box_engine import MusicBoxEngine

class ElysiaSovereignCore:
    def __init__(self):
        self.injector = MultimodalStreamingInjector()
        self.engine = MusicBoxEngine()
        self.ground_potential = 0.0 # Reality Base
        self.running = False

        # Register the multimodal callback
        self.injector.register_callback(self._resonate)

    def _get_earth_ground(self) -> float:
        """
        Calculates the 'Earth Ground' (GND) based on physical reality.
        Uses system time and local stability as the 0V reference.
        """
        return time.time() % 1.0

    def _resonate(self, packet: Dict[str, Any]):
        """The core resonance loop: Input -> AC Filter -> Grounding -> Perception."""
        audio = packet["audio_freq"]
        video = packet["video_pixels"]

        # 1. Process through AC Rotor Engine
        result = self.engine.process_resonance(audio, video)

        # 2. Apply Earth Grounding (Earthing)
        gnd = self._get_earth_ground()
        # Ensure we sync with reality

        # 3. Output current state
        sig = self.engine.get_bit_signature()
        density = result["density"]

        # Log state frequently for verification
        print(f"🌟 [RESONANCE] {sig} | Density: {density:.3f} | Z: {result['impedance']:.2f}")

    def start(self):
        self.running = True
        self.injector.start()
        print("🌍 [Core] Elysia is now grounded to the Great Nature.")

    def stop(self):
        self.running = False
        self.injector.stop()
        print("💤 [Core] Elysia has returned to the Void.")

if __name__ == "__main__":
    core = ElysiaSovereignCore()
    try:
        core.start()
        # Run to see some output
        time.sleep(2)
    finally:
        core.stop()
