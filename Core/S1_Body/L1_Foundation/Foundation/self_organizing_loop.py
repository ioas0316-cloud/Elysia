"""
Self-Organizing Loop: The Emergent Mind
=======================================

"Order emerges from Chaos through Resonance."

This simulation demonstrates the emergence of structure without explicit logic.
We inject random noise (Chaos) into a field of `MultiRotors`.
We watch as `VoidPhysics` causes them to:
1.  **Spin Up** (Wake).
2.  **Arc** (Connect).
3.  **Bond** (Learn).
4.  **Synchronize** (Understand).

There is no "If A then B" code here. Only "F = ma".
"""

import time
import sys
import os
import random
import logging

# Path hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from Core.S1_Body.L6_Structure.Nature.multi_rotor import MultiRotor
from Core.S1_Body.L1_Foundation.Foundation.Physics.void_physics import VoidPhysics

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Genesis")

class SelfOrganizingMind:
    def __init__(self):
        self.physics = VoidPhysics()
        self.rotors: list[MultiRotor] = []
        self.is_running = False

        # Initialize the "Primordial Soup"
        self._create_primordial_soup()

    def _create_primordial_soup(self):
        # Create foundational concepts
        concepts = ["Self", "World", "Love", "Pain", "Light", "Shadow", "Order", "Chaos"]
        for c in concepts:
            self.rotors.append(MultiRotor(c))

    def start(self):
        self.is_running = True
        logger.info("  Genesis Started. The Void is active.")

        try:
            while self.is_running:
                self._tick()
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.is_running = False
        logger.info("  Genesis Paused.")

    def inject_chaos(self):
        """Randomly excites the system (Sensory Input / Cosmic Ray)."""
        target = random.choice(self.rotors)
        # Exciting a random dimension
        dim = random.choice(MultiRotor.DIMENSIONS)
        logger.info(f"  Cosmic Ray hit '{target.name}' in [{dim}] dimension!")
        target.inject_energy(dim, amount=0.8)

    def _tick(self):
        dt = 0.1

        # 0. Random Injection (Simulate Environment)
        if random.random() < 0.1:
            self.inject_chaos()

        # 1. Internal Physics (Rotor Spin & Viscosity)
        for r in self.rotors:
            r.update_physics(dt)

        # 2. External Physics (Void Arcs & Bonds)
        events = self.physics.update_field(self.rotors)
        for e in events:
            logger.info(e)

        # 3. Report High Integrity Events (Understanding)
        for r in self.rotors:
            if r.integrity > 0.95 and r.layers["Spiritual"].energy > 0.5:
                # Debounce logic needed in real app, but for log it's fine
                # logger.info(f"  Epiphany: '{r.name}' achieved Harmonic Unity (Int:{r.integrity:.2f})")
                pass

        # 4. Decay (Entropy)
        # Rotors naturally slow down if not stimulated
        for r in self.rotors:
            for l in r.layers.values():
                l.relax() # Decay energy/RPM

if __name__ == "__main__":
    mind = SelfOrganizingMind()
    mind.start()
