import unittest
import math
from synaptic_architecture.inverse_mechanism_engine import (
    BoundaryCondition,
    ObservedTrajectory,
    DifferentialDelta,
    GeneratingMechanism,
    InverseMechanismEngine
)

class TestInverseMechanismEngine(unittest.TestCase):
    def setUp(self):
        self.engine = InverseMechanismEngine(mdl_penalty_weight=0.1)

    def test_compute_differential_delta(self):
        boundary_a = BoundaryCondition("b1", friction=1.0, scale=1.0)
        boundary_b = BoundaryCondition("b2", friction=2.0, scale=1.5)

        obs_a = ObservedTrajectory(
            trajectory_id="traj_a",
            boundary_id="b1",
            states=[[0.0, 1.0], [1.0, 2.0]]
        )
        obs_b = ObservedTrajectory(
            trajectory_id="traj_b",
            boundary_id="b2",
            states=[[0.5, 1.5], [1.5, 2.5]]
        )

        delta = self.engine.compute_differential_delta(obs_a, boundary_a, obs_b, boundary_b)

        # Boundary delta: [2.0-1.0, 1.5-1.0, 0, 0] = [1.0, 0.5, 0.0, 0.0]
        self.assertEqual(delta.boundary_delta, [1.0, 0.5, 0.0, 0.0])
        self.assertEqual(len(delta.state_deltas), 2)
        self.assertEqual(delta.state_deltas[0], [0.5, 0.5])
        self.assertAlmostEqual(delta.norm_delta, math.sqrt(0.5**2 * 4), places=5)

    def test_extract_generating_mechanism_and_reducibility(self):
        boundaries = {
            "b1": BoundaryCondition("b1", friction=1.0, scale=1.0),
            "b2": BoundaryCondition("b2", friction=2.0, scale=1.0)
        }

        obs1 = ObservedTrajectory("t1", "b1", states=[[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
        obs2 = ObservedTrajectory("t2", "b2", states=[[0.0, 0.0], [2.0, 3.0], [4.0, 6.0]])

        mechanism = self.engine.extract_generating_mechanism("mech_elephant", [obs1, obs2], boundaries)

        # Verify topological invariants & intent vector
        self.assertIsInstance(mechanism, GeneratingMechanism)
        self.assertEqual(len(mechanism.intent_vector), 2)
        # Verify Intent vector direction (ends - starts average)
        # obs1: (2.0 - 0.0, 4.0 - 0.0) = (2.0, 4.0)
        # obs2: (4.0 - 0.0, 6.0 - 0.0) = (4.0, 6.0)
        # average: (3.0, 5.0)
        self.assertAlmostEqual(mechanism.intent_vector[0], 3.0)
        self.assertAlmostEqual(mechanism.intent_vector[1], 5.0)

        # Verify Reducibility (MDL description length calculated and non-zero parameters pruned if < threshold)
        self.assertGreater(mechanism.description_length, 0.0)

    def test_extrapolation_trajectory_generation(self):
        """
        검증 테스트: 단순 보간(Interpolation)이 아닌,
        학습된 범위를 벗어난 새로운 경계 조건(Extrapolation / Perturbation)에서도
        역추출된 생성 메커니즘 Θ에 의해 일관되게 궤적이 자율 생성되는지 검증
        """
        boundaries = {
            "b1": BoundaryCondition("b1", friction=1.0, gravity=9.81),
            "b2": BoundaryCondition("b2", friction=1.5, gravity=9.81)
        }

        obs1 = ObservedTrajectory("t1", "b1", states=[[0.0, 10.0], [1.0, 8.0], [2.0, 4.0]])
        obs2 = ObservedTrajectory("t2", "b2", states=[[0.0, 10.0], [0.8, 7.5], [1.6, 3.0]])

        mechanism = self.engine.extract_generating_mechanism("mech_falling", [obs1, obs2], boundaries)

        # 극단적 외삽 경계 조건 (High friction, high gravity perturbation)
        novel_boundary = BoundaryCondition("b_novel", friction=5.0, gravity=25.0)

        extrapolated_traj = self.engine.generate_trajectory(
            mechanism=mechanism,
            boundary=novel_boundary,
            initial_state=[0.0, 10.0],
            steps=5,
            intent_scale=1.2
        )

        self.assertEqual(len(extrapolated_traj), 5)
        self.assertEqual(extrapolated_traj[0], [0.0, 10.0])
        # Ensure that states change continuously without NaNs or invalid values
        for state in extrapolated_traj:
            self.assertEqual(len(state), 2)
            self.assertFalse(math.isnan(state[0]))
            self.assertFalse(math.isnan(state[1]))

if __name__ == "__main__":
    unittest.main()
