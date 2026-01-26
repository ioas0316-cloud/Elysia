"""
Cosmic Loop: The History of a Universe
======================================

"We watch not a program, but the birth of a soul."

This is the simulation loop for the Fractal Manifold architecture.
It simulates the collision of worlds (`MultiRotors` with `HelixDNA`).
The output is a rich narrative of these cosmic events.
"""

import time
import sys
import os
import random
import logging

# Path hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from Core.L6_Structure.Nature.multi_rotor import MultiRotor
from Core.L1_Foundation.Foundation.Physics.event_horizon import EventHorizon

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("CosmicGenesis")

class CosmicLoop:
    def __init__(self):
        self.physics = EventHorizon()
        self.rotors: list[MultiRotor] = []
        self.is_running = False

        # Initialize the "Primordial Universes"
        # We give them some initial density/curvature to make it interesting
        self._seed_universe()

    def _seed_universe(self):
        concepts = [
            ("Self", 0.9),      # High Gravity
            ("World", 0.8),     # High Gravity
            ("Love", 1.0),      # Dense
            ("Pain", 0.7),
            ("Logic", 0.5),
            ("Chaos", 0.6)
        ]
        for name, density in concepts:
            rotor = MultiRotor(name)
            # Initialize manifolds with specific traits
            for d in rotor.dna.yin_strand.values():
                d.curvature = density
                d.density = density
            self.rotors.append(rotor)

    def start(self):
        self.is_running = True
        logger.info("  Cosmic Genesis Started. The Manifolds are breathing.")

        try:
            while self.is_running:
                self._tick()
                time.sleep(0.5) # Slower tick to read the logs
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.is_running = False
        logger.info("  Genesis Paused.")

    def inject_chaos(self):
        """Simulates an external event impacting the universe."""
        target = random.choice(self.rotors)
        dim = random.choice(MultiRotor.DIMENSIONS)
        logger.info(f"  [EVENT] A Cosmic Ray strikes the '{dim}' manifold of '{target.name}'!")
        target.inject_energy(dim, amount=1.0)

    def _tick(self):
        dt = 0.5

        # 0. Random Injection
        if random.random() < 0.2:
            self.inject_chaos()

        # 1. Internal Physics (Evolution)
        for r in self.rotors:
            r.update_physics(dt)

        # 2. Event Horizon (Collisions)
        # We pick random pairs to check for interaction (simulating spatial proximity)
        for _ in range(3):
            r_a = random.choice(self.rotors)
            r_b = random.choice(self.rotors)
            if r_a == r_b: continue

            events = self.physics.interact(r_a, r_b)
            for e in events:
                logger.info(str(e))

        # 3. Expansion
        # Logs occasionally
        # if random.random() < 0.1:
        #     logger.info(f"   (The Universe expands... Radius: {self.rotors[0].dna.yin_strand['Spiritual'].radius:.2f})")

if __name__ == "__main__":
    loop = CosmicLoop()
    loop.start()
