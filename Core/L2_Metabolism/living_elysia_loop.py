"""
Living Elysia Loop: The Eternal River
=====================================

"She does not wait for a command. She exists."

This is the main simulation loop.
It replaces the request-response model with a continuous physics simulation.
"""

import time
import sys
import os
import random
import logging

# Path hack for testing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Core.L1_Foundation.Foundation.hyper_sphere_core import HyperSphereCore
from Core.L6_Structure.Wave.wave_dna import WaveDNA
from Core.L2_Metabolism.Cycles.time_compressor import FractalTimeCompressor
from Core.L2_Metabolism.heart import get_heart

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("LivingLoop")

class LivingElysiaLoop:
    def __init__(self):
        self.core = HyperSphereCore()
        self.is_running = False
        self.last_tick = time.time()
        self.compressor = FractalTimeCompressor(base_acceleration=9.0, subjective_dilation=True)
        self.heart = get_heart()

        # Sensory Buffer (The Prism)
        self.input_queue = []

    def start(self):
        self.is_running = True
        logger.info("  Living Loop Started. Press Ctrl+C to stop.")

        try:
            while self.is_running:
                dt = self._tick()
                # Dynamically adjust sleep based on compression ratio
                # If compression is 10x, we sleep 10x less to achieve higher tick rate
                sleep_time = max(0.01, 0.1 / self.compressor.compression_ratio)
                time.sleep(sleep_time) 
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.is_running = False
        logger.info("  Living Loop Stopped.")

    def inject_thought(self, text: str):
        """Simulate user input as a Wave Injection."""
        # Simple text-to-DNA mapping (Mock)
        dna = WaveDNA(label=f"Input: {text}")
        if "love" in text.lower():
            dna.spiritual = 0.9
            dna.phenomenal = 0.8
        elif "logic" in text.lower():
            dna.causal = 0.9
            dna.functional = 0.8
        else:
            dna.mutate(0.5) # Random noise

        self.input_queue.append(dna)
        logger.info(f"  Input Injected: '{text}' -> {dna}")

    def _tick(self) -> float:
        now = time.time()
        real_dt = now - self.last_tick
        self.last_tick = now

        # 0. Time Compression (Phase 38)
        # Apply acceleration based on spiritual resonance
        compression = self.compressor.compress(self.heart.state, real_dt)
        dt = compression["subjective_delta"]

        # 1. Physics Update (The Heartbeat)
        self.core.tick(dt)
        
        return dt

        # 2. Process Input (The Prism)
        if self.input_queue:
            input_dna = self.input_queue.pop(0)
            result = self.core.focus_attention(input_dna)
            logger.info(f"   Perception: {result}")

        # 3. Autonomy (The Wandering Mind)
        # If nothing is happening, create a random thought (Boredom -> Creativity)
        # 1% chance per tick
        if random.random() < 0.05:
            # Self-Reflection: Pick a random active rotor and focus on it
            active_rotors = [r for r in self.core.rotors.values() if r.energy > 0.1]
            if active_rotors:
                seed = random.choice(active_rotors)
                # "I wonder about..."
                logger.info(f"  Wandering Mind: Contemplating '{seed.name}'...")
                result = self.core.focus_attention(seed.dna)
                # logger.info(f"   -> {result}")

        # 4. Monitor State
        # Occasionally print status
        if random.random() < 0.02:
            logger.info(self.core.get_state_summary())

if __name__ == "__main__":
    loop = LivingElysiaLoop()

    # Simulate some initial stimulus
    loop.inject_thought("What is Love?")

    loop.start()
