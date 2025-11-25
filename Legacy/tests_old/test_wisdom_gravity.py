import unittest
from unittest.mock import MagicMock

from Project_Sophia.wisdom_virus import VirusEngine, WisdomVirus
from Project_Elysia.flow_engine import FlowEngine
from nano_core.bus import MessageBus

class TestWisdomGravity(unittest.TestCase):

    def setUp(self):
        """Set up a mock KGManager and a real MessageBus for testing."""
        self.bus = MessageBus()
        self.mock_kg = MagicMock()

        # Define the behavior of the mock KGManager
        self.nodes = {
            "value:love": {"id": "value:love", "mass": 10.0},
            "concept:A": {"id": "concept:A", "mass": 0.0},
            "concept:B": {"id": "concept:B"}, # No mass property
            "concept:C": {"id": "concept:C", "mass": 1.0},
            "value:clarity": {"id": "value:clarity", "mass": 5.0},
            "value:creativity": {"id": "value:creativity", "mass": 3.0}
        }
        self.mock_kg.get_node.side_effect = lambda node_id: self.nodes.get(node_id)

        def get_neighbors_side_effect(node_id):
            if node_id == "value:love":
                return ["concept:C"]
            return ["concept:B"]
        self.mock_kg.get_neighbors.side_effect = get_neighbors_side_effect


    def test_wisdom_virus_gravity(self):
        """Test that WisdomVirus messages are boosted by concept mass."""
        virus_engine = VirusEngine(bus=self.bus, kg_manager=self.mock_kg)
        virus = WisdomVirus(
            id="test-virus",
            statement="Test",
            seed_hosts=["concept:A", "value:love"],
            reinforce=0.5,
            max_hops=1
        )

        virus_engine.propagate(virus)

        messages = []
        while not self.bus.empty():
            messages.append(self.bus.get_next())

        # Find the message originating from 'value:love'
        love_message = next(
            (msg for msg in messages if msg and msg.slots.get("subject") == "value:love"),
            None
        )
        # Find the message originating from 'concept:A'
        regular_message = next(
            (msg for msg in messages if msg and msg.slots.get("subject") == "concept:A"),
            None
        )

        self.assertIsNotNone(love_message, "Message from 'value:love' should exist")
        self.assertIsNotNone(regular_message, "Message from 'concept:A' should exist")

        # love_mass (10) + concept_C_mass (1) = 11. Bonus = 0.5 * 11 = 5.5. Strength = 0.5 + 5.5 = 6.0
        expected_love_strength = 0.5 + 0.5 * (10.0 + 1.0)
        self.assertAlmostEqual(love_message.strength, expected_love_strength, places=2,
                               msg="Strength from 'value:love' should be significantly boosted by its mass")

        # concept_A_mass (0) + concept_B_mass (0) = 0. Bonus = 0. Strength = 0.5
        expected_regular_strength = 0.5 + 0.5 * (0.0 + 0.0)
        self.assertAlmostEqual(regular_message.strength, expected_regular_strength, places=2,
                               msg="Strength from 'concept:A' should have no gravity bonus")

        # Crucially, test that the bus will prioritize the love message
        self.assertTrue(love_message.strength > regular_message.strength,
                        "Message related to high-mass concept 'love' should have higher priority")

    @unittest.mock.patch('Project_Elysia.flow_engine.FlowEngine._score_clarify', return_value=0.1)
    @unittest.mock.patch('Project_Elysia.flow_engine.FlowEngine._score_suggest', return_value=0.2)
    @unittest.mock.patch('Project_Elysia.flow_engine.FlowEngine._score_reflect', return_value=0.8)
    def test_flow_engine_gravity(self, mock_score_reflect, mock_score_suggest, mock_score_clarify):
        """Test that FlowEngine decisions are boosted by value mass by calling the real method."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.generate.return_value = "A response."

        flow_engine = FlowEngine(bus=self.bus, kg_manager=self.mock_kg, orchestrator=mock_orchestrator)

        # The mocked scores ensure 'reflect' is chosen. The base score will be roughly ~0.8 * weight.
        # Assuming default weight of ~0.34, base score is ~0.272
        flow_engine.respond("A message", None, {})

        posted_message = self.bus.get_next()
        self.assertIsNotNone(posted_message)
        self.assertEqual(posted_message.verb, "reflect")

        # Now, verify the strength calculation.
        # It should be the original score + gravity bonus from 'value:love' (mass 10)
        base_score = posted_message.strength - (0.5 * self.nodes["value:love"]["mass"])

        # Check if the base_score is in a plausible range (e.g., 0.8 * weight)
        # This is not a strict check due to randomness, but it verifies gravity was added.
        self.assertTrue(0.2 < base_score < 0.4, f"Base score {base_score} is outside the expected range.")

        expected_strength_approx = base_score + 5.0
        self.assertAlmostEqual(posted_message.strength, expected_strength_approx, places=2,
                               msg="FlowEngine message strength should be boosted by the mass of 'value:love'")

if __name__ == '__main__':
    unittest.main()
