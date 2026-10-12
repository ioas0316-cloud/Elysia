import unittest
import numpy as np
from core.physics.ontological_gravity_terrain_engine import OntologicalGravityTerrainEngine, BoundaryObstacle
from core.physics.causal_field import CausalField, InformationVoxel

class TestOntologicalGravityTerrain(unittest.TestCase):

    def setUp(self):
        self.engine = OntologicalGravityTerrainEngine(
            dimensions=16,
            anchor_stiffness=0.5,
            gravitational_constant=1.0,
            damping_factor=0.85
        )

    def test_potential_and_gradient(self):
        """Test analytical potential and gradient computation without hard rules."""
        pos = np.ones(16, dtype=np.float32)
        v_pot = self.engine.compute_potential(pos)
        grad = self.engine.compute_gradient(pos)

        # Potential = 0.5 * k * ||x||^2 = 0.5 * 0.5 * 16 = 4.0
        self.assertAlmostEqual(v_pot, 4.0, places=4)
        # Gradient = k * x = 0.5 * 1.0 = 0.5
        np.testing.assert_allclose(grad, np.full(16, 0.5, dtype=np.float32))

    def test_smooth_relaxation_flow(self):
        """Test state relaxation trajectory towards Local Zero."""
        initial_pos = np.array([5.0] * 16, dtype=np.float32)
        trajectory = self.engine.simulate_relaxation(initial_position=initial_pos, steps=30, dt=0.1)

        first_point = trajectory[0]
        last_point = trajectory[-1]

        # Verify energy decreases naturally
        self.assertGreater(first_point.potential_energy, last_point.potential_energy)
        self.assertGreater(first_point.gradient_norm, last_point.gradient_norm)
        # Verify position moved towards Local Zero (0_self)
        self.assertLess(np.linalg.norm(last_point.position), np.linalg.norm(first_point.position))

    def test_boundary_obstacle_repulsion(self):
        """Test smooth deflection away from boundary obstacles."""
        self.engine.add_boundary("mountain_1", center=np.array([2.0] * 16), amplitude=5.0, sigma=1.0)

        # Position at 1.9 (between 0_self at 0 and mountain at 2.0)
        # Anchor pulls toward 0 (- direction), mountain pushes away from 2.0 toward 0 (- direction)
        near_pos = np.array([1.9] * 16, dtype=np.float32)
        v_pot = self.engine.compute_potential(near_pos)
        grad = self.engine.compute_gradient(near_pos)

        # Boundary potential adds positive energy bump
        self.assertGreater(v_pot, 0.5 * 0.5 * np.sum(near_pos ** 2))
        # Both forces pull towards 0 (left), so gradient magnitude is greater than anchor alone
        self.assertGreater(np.linalg.norm(grad), np.linalg.norm(0.5 * near_pos))

    def test_inner_outer_discrepancy_calibration(self):
        """Test Metanoia self-calibration based on inner thought vs outer reality gap."""
        inner_pos = np.array([3.0] * 16, dtype=np.float32)
        outer_pos = np.array([1.0] * 16, dtype=np.float32)

        prev_stiffness = self.engine.anchor_stiffness
        calib = self.engine.sense_and_recalibrate_discrepancy(inner_pos, outer_pos)

        self.assertGreater(calib["discrepancy_norm"], 0.0)
        # Stiffness adjusted to ground inner thoughts towards reality
        self.assertGreater(self.engine.anchor_stiffness, prev_stiffness)
        self.assertGreater(np.linalg.norm(self.engine.zero_self), 0.0)

    def test_causal_field_integration(self):
        """Test integration into CausalField."""
        cf = CausalField(dimensions=3)
        voxel = InformationVoxel("v1", "test", np.ones(3), position=np.array([4.0, 4.0, 4.0]))
        cf.add_voxel(voxel)

        init_pos_norm = np.linalg.norm(voxel.position)
        cf.step(0.1)
        step_pos_norm = np.linalg.norm(voxel.position)

        # Voxel position relaxed along ontological gravity terrain
        self.assertLess(step_pos_norm, init_pos_norm)

if __name__ == "__main__":
    unittest.main()
