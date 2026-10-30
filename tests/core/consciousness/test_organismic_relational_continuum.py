import unittest
import numpy as np
from L1_causal_field.core.consciousness.organismic_relational_continuum import (
    RelationalEntity,
    OrganismicRelationalContinuum
)


class TestOrganismicRelationalContinuum(unittest.TestCase):

    def setUp(self):
        self.continuum = OrganismicRelationalContinuum(dimension=16)

    def test_relational_entity_initialization(self):
        entity = RelationalEntity("test_node", "cognitive", dimension=16)
        self.assertEqual(entity.entity_id, "test_node")
        self.assertEqual(entity.layer_type, "cognitive")
        self.assertEqual(len(entity.identity_vector), 16)
        self.assertAlmostEqual(np.linalg.norm(entity.identity_vector), 1.0, places=5)

    def test_goodness_conflict_evaluation(self):
        entity = self.continuum.entities["cellular_soma"]
        stimulus = np.ones(16)
        friction, updated_profile = entity.evaluate_goodness_conflict(stimulus)

        self.assertGreaterEqual(friction, 0.0)
        self.assertIn("cellular", updated_profile)
        self.assertIn("mental", updated_profile)

    def test_mutual_transformation_on_coupling(self):
        # Test that coupling entity A and B mutually transforms both entity A and B's
        # internal identity vectors and relational affordance capacities
        ea = self.continuum.entities["cellular_soma"]
        eb = self.continuum.entities["mental_spirit"]

        prev_a_vec = np.copy(ea.identity_vector)
        prev_b_vec = np.copy(eb.identity_vector)
        prev_a_aff = np.copy(ea.affordance_capacity)
        prev_b_aff = np.copy(eb.affordance_capacity)

        report = self.continuum.couple_entities_mutually("cellular_soma", "mental_spirit", coupling_intensity=0.8)

        # Check mutual identity shift
        self.assertGreater(report["a_identity_shift_norm"], 0.0)
        self.assertGreater(report["b_identity_shift_norm"], 0.0)

        # Check mutual affordance capacity shift
        self.assertGreater(report["a_affordance_change_norm"], 0.0)
        self.assertGreater(report["b_affordance_change_norm"], 0.0)

        # Confirm identity vectors updated and normalized
        self.assertFalse(np.allclose(ea.identity_vector, prev_a_vec))
        self.assertFalse(np.allclose(eb.identity_vector, prev_b_vec))
        self.assertAlmostEqual(np.linalg.norm(ea.identity_vector), 1.0, places=5)
        self.assertAlmostEqual(np.linalg.norm(eb.identity_vector), 1.0, places=5)

    def test_macro_orientation_emergence_and_top_down_recalibration(self):
        prev_macro = np.copy(self.continuum.macro_orientation_vector)

        stimulus = np.random.randn(16)
        res = self.continuum.interact_with_world_environment(stimulus, intensity=0.7)

        self.assertGreater(res["interaction_reports_count"], 0)
        # Macro orientation vector should evolve
        self.assertFalse(np.allclose(self.continuum.macro_orientation_vector, prev_macro))
        self.assertAlmostEqual(np.linalg.norm(self.continuum.macro_orientation_vector), 1.0, places=5)

    def test_non_pipeline_co_dependent_dynamics(self):
        # Confirm that multiple couplings lead to emergent, non-linear changes in goodness profiles
        # without hardcoded conditional rules
        ea = self.continuum.entities["sensory_organ"]
        eb = self.continuum.entities["cognitive_mind"]

        init_profile_a = dict(ea.goodness_profile)
        init_profile_b = dict(eb.goodness_profile)

        for _ in range(5):
            self.continuum.couple_entities_mutually("sensory_organ", "cognitive_mind", coupling_intensity=0.5)

        self.assertNotEqual(ea.goodness_profile, init_profile_a)
        self.assertNotEqual(eb.goodness_profile, init_profile_b)


if __name__ == "__main__":
    unittest.main()
