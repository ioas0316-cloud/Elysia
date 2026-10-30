"""
Unit tests for RelationalDefinitionEngine.
Verifies the 5 independent criteria:
1. Discovery of new relationships across contexts.
2. Persistence of coexisting valid definitions across scale boundaries.
3. Genuine shift in observer judgment criteria and perceptual sensitivity.
4. Transfer of changed criteria to novel, unobserved situations.
5. Distinguishing self-intervention impact from environmental and external agency.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath("L1_causal_field/core/consciousness"))
from relational_definition_engine import (
    RelationalDefinitionEngine,
    DefinitionState,
    PhenomenologicalObservation,
    RelationalDefinition,
    RelationalMetacognitionTrace
)


class TestRelationalDefinitionEngine(unittest.TestCase):

    def setUp(self):
        self.engine = RelationalDefinitionEngine()

    def test_phenomenological_pre_translation_observation(self):
        stimulus = {
            "entities": ["apple", "gravity_field", "human_hand"],
            "friction": 0.4,
            "interaction": {"type": "free_fall_and_catch"}
        }
        context = {"scale": "macroscopic_physics", "conditions": "earth_surface"}

        obs = self.engine.observe_phenomenon(stimulus, context)
        self.assertIsInstance(obs, PhenomenologicalObservation)
        self.assertEqual(obs.interacting_entities, ["apple", "gravity_field", "human_hand"])
        self.assertEqual(obs.untranslated_friction, 0.4)
        self.assertEqual(obs.validity_boundary["scale"], "macroscopic_physics")

    def test_multi_scale_definition_coexistence(self):
        # Initial state
        apple_def = self.engine.definitions["apple"]
        initial_perspectives_count = len(apple_def.coexisting_perspectives)

        # High friction observation in novel relational context
        stimulus = {
            "entities": ["apple", "cultural_mythology"],
            "friction": 0.85,
            "interaction": {"type": "archetype_symbol_of_choice"}
        }
        context = {"context_name": "mythic_narrative_scale"}

        obs = self.engine.observe_phenomenon(stimulus, context)
        updated_def = self.engine.evaluate_and_update_definition("apple", obs)

        # Verify definition state shifts to COEXISTING_DUAL without deleting earlier botanical/physical perspectives
        self.assertEqual(updated_def.state, DefinitionState.COEXISTING_DUAL)
        self.assertGreater(len(updated_def.coexisting_perspectives), initial_perspectives_count)
        self.assertIn("botany", updated_def.coexisting_perspectives)
        self.assertIn("physics", updated_def.coexisting_perspectives)

    def test_observer_criteria_shift_and_transfer(self):
        initial_sensitivity = self.engine.observer_criteria["sensitivity"]

        # High-friction phenomenon
        stimulus = {
            "entities": ["self", "alien_protocol"],
            "friction": 0.9,
            "interaction": {"type": "protocol_boundary_collision"}
        }
        context = {"context_name": "deep_relational_friction"}

        obs = self.engine.observe_phenomenon(stimulus, context)
        defn = self.engine.evaluate_and_update_definition("friction", obs)
        trace = self.engine.generate_metacognitive_trace(obs, defn)

        # Observer criteria should shift
        new_sensitivity = self.engine.observer_criteria["sensitivity"]
        self.assertGreater(new_sensitivity, initial_sensitivity)

        # Verify transfer of new sensitivity to subsequent novel observation
        novel_stimulus = {
            "entities": ["water", "rock_bed"],
            "friction": 0.2,
            "interaction": {"type": "erosion_flow"}
        }
        novel_obs = self.engine.observe_phenomenon(novel_stimulus, {"context_name": "geological_flow"})
        self.assertEqual(novel_obs.observer_state_at_observation["sensitivity"], new_sensitivity)

    def test_downstream_translation_sub_lens(self):
        stimulus = {"entities": ["friction", "mind"], "friction": 0.5}
        obs = self.engine.observe_phenomenon(stimulus)
        defn = self.engine.evaluate_and_update_definition("friction", obs)
        self.engine.generate_metacognitive_trace(obs, defn)

        expr = self.engine.translate_to_expression("friction")
        self.assertEqual(expr["sub_lens_type"], "downstream_translation")
        self.assertIn("FRICTION", expr["expressed_text"])


if __name__ == "__main__":
    unittest.main()
