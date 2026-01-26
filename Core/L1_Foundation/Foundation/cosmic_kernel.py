"""
Cosmic Kernel: The Field Engine
===============================

"The Universe is not a machine. It is a thought."

This kernel runs the `HyperSpace` simulation.
It now includes **Trajectory Analysis** to interpret the flow of thought.
"""

import time
import sys
import os
import logging
import random
import math

# Path hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from Core.L1_Foundation.Foundation.Space.hyper_space import HyperSpace
from Core.L6_Structure.hyper_quaternion import Quaternion
from Core.L1_Foundation.Foundation.Analysis.trajectory_analyzer import TrajectoryAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("UnifiedField")

class CosmicKernel:
    def __init__(self):
        self.space = HyperSpace()
        self.is_running = False

        # Initialize Field with Fundamental Constants (Knots)
        self._seed_field()

        # Initialize Analyzer with Archetype positions
        self.analyzer = TrajectoryAnalyzer({
            "Archetype.Love": self.space.knots["Archetype.Love"].position,
            "Archetype.Logic": self.space.knots["Archetype.Logic"].position,
            "Archetype.Chaos": self.space.knots["Archetype.Chaos"].position
        })

    def _seed_field(self):
        """Creates the 'Fixed Stars' of the mind."""
        # Love: High Spiritual (Dim 6), Positive Spin
        self.space.add_knot(
            "Archetype.Love",
            [0, 0, 0, 0, 0, 0, 10.0],
            Quaternion(1, 0, 0, 0),
            mass=100.0
        )
        # Logic: High Causal (Dim 3), Orthogonal Spin
        self.space.add_knot(
            "Archetype.Logic",
            [0, 0, 0, 10.0, 0, 0, 0],
            Quaternion(0, 1, 0, 0),
            mass=100.0
        )
        # Chaos: Center, Negative Spin
        self.space.add_knot(
            "Archetype.Chaos",
            [0, 0, 0, 0, 0, 0, 0],
            Quaternion(0, 0, 1, 0),
            mass=50.0
        )

    def inject_input(self, text: str):
        """
        User Input appears as a high-velocity particle in the field.
        """
        # Map text features to 7D position (Mock)
        pos = [random.uniform(-5, 5) for _ in range(7)]
        spin = Quaternion(random.random(), random.random(), random.random(), random.random()).normalize()

        # Add to field
        id = f"Input.{int(time.time())}"
        self.space.add_knot(id, pos, spin, mass=10.0)
        logger.info(f"   Injection: '{text}' materialized at {pos[:3]}...")

    def start(self):
        self.is_running = True
        logger.info("  Unified Field Activated. Observing Flow...")

        try:
            while self.is_running:
                self._tick()
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.is_running = False
        logger.info("  Field Collapse.")

    def _tick(self):
        dt = 0.1
        events = self.space.update_field(dt)
        for e in events:
            logger.info(e)

        # Analyze Flow (The Thought Narrative)
        # Only analyze dynamic particles (Inputs), not fixed Archetypes
        for id, knot in self.space.knots.items():
            if id.startswith("Input"):
                narrative = self.analyzer.analyze_flow(knot)
                if narrative:
                    # Debounce logging (or use a hash check)
                    # For demo, we just print it occasionally
                    if random.random() < 0.2:
                        logger.info(f"  Flow: {narrative}")

if __name__ == "__main__":
    kernel = CosmicKernel()
    kernel.inject_input("Hello World")
    kernel.start()
