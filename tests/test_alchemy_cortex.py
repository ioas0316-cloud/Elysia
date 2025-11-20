import unittest
from Project_Sophia.core.alchemy_cortex import AlchemyCortex

class TestAlchemyCortex(unittest.TestCase):
    def setUp(self):
        self.cortex = AlchemyCortex()

    def test_synthesize_fire_punch(self):
        concepts = ["fire", "punch"]
        action = self.cortex.synthesize_action(concepts)

        self.assertEqual(action["id"], "action:fire_punch")
        logic = action["logic"]

        # Check Cost (Fire 5 Ki + Punch 0 = 5)
        self.assertEqual(logic["cost"]["ki"], 5)

        # Check Effects
        ops = [e["op"] for e in logic["effects"]]
        self.assertIn("damage", ops)
        # Fire effect
        fire_effects = [e for e in logic["effects"] if e.get("type") == "fire"]
        self.assertTrue(len(fire_effects) > 0)

    def test_synthesize_wind_walk(self):
        # Assuming "walk" isn't in essence mapper, it should just use "wind" properties
        concepts = ["wind"]
        action = self.cortex.synthesize_action(concepts)

        self.assertEqual(action["id"], "action:wind")
        # Check effect
        ops = [e["op"] for e in action["logic"]["effects"]]
        self.assertIn("modify_stat", ops)
        self.assertEqual(action["logic"]["effects"][0]["stat"], "agility")

if __name__ == '__main__':
    unittest.main()
