
import sys
import os
import unittest
import numpy as np

# Adjust path to find Core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.Cognition.monad_gravity import MonadGravityEngine

class TestMonadGravity(unittest.TestCase):
    def setUp(self):
        self.engine = MonadGravityEngine()
        self.genesis_triggered = False
        self.fused_pair = None

    def genesis_handler(self, id1, id2):
        print(f"[TEST] Genesis Handler called for {id1} and {id2}")
        self.genesis_triggered = True
        self.fused_pair = sorted([id1, id2])

    def test_attraction(self):
        """Test if resonant monads move closer."""
        self.engine.set_genesis_callback(self.genesis_handler)

        # Two similar monads (Red and slightly Orange-Red)
        # Positioned slightly apart in 7D space
        # Vec 1: Pure Red
        vec1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # Vec 2: Red with hint of Orange (Resonant)
        vec2 = [0.9, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Place them at distance
        # We simulate "position" by setting the vector.
        # Note: In our Physics, pos IS the vector (Phase Space).
        # So high resonance means they are nearby in angle, but might be far in magnitude?
        # Actually distance_to is Euclidean.
        # vec1 and vec2 are close in Euclidean distance too.

        self.engine.add_monad("Idea_A", vec1, mass=1.0)
        self.engine.add_monad("Idea_B", vec2, mass=1.0)

        # Initial distance
        p1 = self.engine.particles["Idea_A"]
        p2 = self.engine.particles["Idea_B"]
        dist_start = p1.pos.distance_to(p2.pos)
        print(f"Start Distance: {dist_start}")

        # Run 10 ticks
        for _ in range(10):
            self.engine.step()

        dist_end = p1.pos.distance_to(p2.pos)
        print(f"End Distance: {dist_end}")

        # They should have moved closer (attraction) or collided
        self.assertTrue(dist_end < dist_start or self.genesis_triggered,
                        "Monads did not attract or fuse.")

    def test_genesis_trigger(self):
        """Test if very close monads trigger Genesis."""
        self.engine.set_genesis_callback(self.genesis_handler)

        # Very close vectors
        vec1 = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        vec2 = [0.51, 0.49, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.engine.add_monad("Love", vec1)
        self.engine.add_monad("Compassion", vec2)

        # Run simulation
        for _ in range(20):
            self.engine.step()
            if self.genesis_triggered:
                break

        self.assertTrue(self.genesis_triggered, "Genesis was not triggered for close concepts.")
        self.assertEqual(self.fused_pair, ["Compassion", "Love"])

if __name__ == "__main__":
    unittest.main()
