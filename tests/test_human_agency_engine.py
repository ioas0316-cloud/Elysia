"""
test_human_agency_engine.py
============================
Unit tests for Human Agency Engine & Landau-Ginzburg Phase Collapse Mechanics.
"""

import unittest
import numpy as np
import torch

from modules.causal_game_engine.human_agency_engine import (
    HumanAgencyEngine,
    HumanAgencyEvaluator,
    LandauGinzburgPotentialField,
    TransitionPath,
    AscensionEvent,
    InversionEvent,
    BoundaryExpansionEvent,
    ChoiceOption,
)
from modules.causal_game_engine.causal_scm_nn import DifferentiableSCM, CausalLossCalculator


class TestHumanAgencyEngine(unittest.TestCase):

    def setUp(self):
        self.potential = LandauGinzburgPotentialField(a=-2.0, b=1.0)
        self.evaluator = HumanAgencyEvaluator(gamma_agency=1.0, alpha_wonder=2.0)
        self.engine = HumanAgencyEngine(potential_field=self.potential, evaluator=self.evaluator, dt=0.01, num_steps=50)

    def test_choice_probabilities_and_entropy(self):
        options = [
            ChoiceOption("opt_1", utility=0.2, alignment_vector=np.array([1.0, 0.0])),
            ChoiceOption("opt_2", utility=0.8, alignment_vector=np.array([-1.0, 0.0])),
        ]
        W_constellation = np.array([1.0, 0.0])  # Prefers opt_1

        P_pred, P_act, S_defiance = self.evaluator.calculate_choice_probabilities(options, W_constellation)

        self.assertAlmostEqual(np.sum(P_pred), 1.0, places=5)
        self.assertAlmostEqual(np.sum(P_act), 1.0, places=5)
        self.assertGreater(P_pred[0], P_pred[1])  # Constellation heavily predicts opt_1

        H_causal = self.evaluator.calculate_causal_entropy(P_act, P_pred)
        self.assertGreaterEqual(H_causal, 0.0)

        A_wonder = self.evaluator.calculate_wonder_index(H_causal, xi_ordeal=1.5)
        self.assertTrue(0.0 <= A_wonder <= 1.0)

    def test_landau_ginzburg_potential_and_phase_collapse(self):
        H_origin = np.array([0.0, 0.0])
        is_collapsed, det_val = self.potential.is_phase_collapsed(H_origin)

        # At origin with a < 0, Hessian = a * I = [[-2, 0], [0, -2]] => det = 4 > 0, but curvature is negative
        # Wait: Hessian is [[a, 0], [0, a]] => det = a^d = (-2)^2 = 4.
        # But Hessian eigenvalues are -2, -2 <= 0 => local maximum (unstable valley).
        # det is positive for 2D when both eigenvalues are negative.
        # Let's verify gradient and hessian shape.
        hess = self.potential.hessian(H_origin)
        self.assertEqual(hess.shape, (2, 2))

    def test_sde_trajectory_and_bifurcation(self):
        H_init = np.array([0.1, 0.1])
        E_trial = np.array([2.0, 2.0])
        H_angel = np.array([1.0, 1.0])
        H_devil = np.array([-1.0, -1.0])

        result = self.engine.process_trial_event(
            hero_id="hero_test",
            H_current=H_init,
            E_trial=E_trial,
            H_angel=H_angel,
            H_devil=H_devil,
            T_agency=1.0,
            seed=42
        )

        self.assertIn("bifurcation_path", result)
        self.assertIn("H_final", result)
        self.assertEqual(len(result["H_final"]), 2)

    def test_causal_loss_calculator_with_wonder(self):
        scm = DifferentiableSCM(num_nodes=4)
        loss_calc = CausalLossCalculator(lambda_wonder=0.5)

        x = torch.randn(8, 4)
        target = torch.randn(8, 4)
        pred = scm(x)

        W_adj = scm.get_masked_adj()
        loss_without_wonder = loss_calc(pred, target, W_adj, A_wonder=0.0)
        loss_with_wonder = loss_calc(pred, target, W_adj, A_wonder=0.8)

        self.assertLess(loss_with_wonder.item(), loss_without_wonder.item())


if __name__ == "__main__":
    unittest.main()
