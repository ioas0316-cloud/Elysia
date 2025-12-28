
import unittest
from Core.Ether.ether_node import EtherNode, Quaternion
from Core.Ether.void import Void
from Core.Ether.field_operators import LawOfGravity, DynamicsEngine
import math
import numpy as np

class TestLawOfGravity(unittest.TestCase):
    def setUp(self):
        self.void = Void()
        self.law = LawOfGravity()

    def test_gravity_attraction(self):
        # Two nodes with mass, far apart on X axis
        # Use keyword arguments to be explicit about coordinates
        node1 = EtherNode(mass=10.0, position=Quaternion(w=0, x=0, y=0, z=0))
        node2 = EtherNode(mass=10.0, position=Quaternion(w=0, x=10, y=0, z=0))

        self.void.add(node1)
        self.void.add(node2)

        self.law.apply(self.void, dt=1.0)

        # Node 1 should move towards Node 2 (positive x)
        self.assertTrue(node1.velocity.x > 0, f"Node 1 velocity x should be positive, got {node1.velocity.x}")
        # Node 2 should move towards Node 1 (negative x)
        self.assertTrue(node2.velocity.x < 0, f"Node 2 velocity x should be negative, got {node2.velocity.x}")

    def test_repulsion_close_range(self):
        # Two nodes very close on X axis
        node1 = EtherNode(mass=1.0, position=Quaternion(w=0, x=0, y=0, z=0))
        # Dist 0.1 < MIN_DIST 0.5
        node2 = EtherNode(mass=1.0, position=Quaternion(w=0, x=0.1, y=0, z=0))

        self.void.add(node1)
        self.void.add(node2)

        self.law.apply(self.void, dt=1.0)

        # Repulsion means node1 moves AWAY from node2 (negative x)
        self.assertTrue(node1.velocity.x < 0, "Node 1 should be repelled (negative x velocity)")
        # Node 2 moves AWAY from node1 (positive x)
        self.assertTrue(node2.velocity.x > 0, "Node 2 should be repelled (positive x velocity)")

    def test_zero_mass_ignored(self):
        node1 = EtherNode(mass=0.0, position=Quaternion(w=0, x=0, y=0, z=0))
        node2 = EtherNode(mass=10.0, position=Quaternion(w=0, x=10, y=0, z=0))

        self.void.add(node1)
        self.void.add(node2)

        self.law.apply(self.void, dt=1.0)

        # Node 1 has 0 mass, so apply_force should ignore it.
        self.assertEqual(node1.velocity.x, 0.0)

if __name__ == '__main__':
    unittest.main()
