import unittest
from Core.FoundationLayer.Foundation.core.alchemy_cortex import AlchemyCortex

class TestAlchemyCortex(unittest.TestCase):
    def setUp(self):
        self.alchemy = AlchemyCortex()

    def test_synthesize_fire_punch(self):
        """Test combining an element (Fire) and an action (Punch)."""
        concepts = ['fire', 'punch']
        action = self.alchemy.synthesize_action(concepts)

        self.assertEqual(action['id'], "action:fire_punch")
        logic = action['logic']

        # Check combined costs
        # Fire: 5 ki, Punch: (no cost in default mapper, but let's check)
        # Actually punch has no cost in the mapper, only conditions.
        self.assertEqual(logic['cost'].get('ki'), 5)

        # Check effects
        # Fire: damage x0.5 (fire type)
        # Punch: damage x1.0
        # Should have at least these two effects
        effects = logic['effects']
        ops = [e['op'] for e in effects]
        self.assertIn('damage', ops)
        self.assertTrue(len(effects) >= 2)

        print(f"\nSynthesized: {action['label']}")
        print(f"Logic: {logic}")

    def test_synthesize_void_magic(self):
        """Test combining Void (Data Deletion) with an Action."""
        # Assuming 'attack' is a generic action
        concepts = ['void', 'attack']
        action = self.alchemy.synthesize_action(concepts)

        logic = action['logic']
        effects = logic['effects']

        # Void has "overwrite" op
        ops = [e['op'] for e in effects]
        self.assertIn('overwrite', ops)
        self.assertIn('damage', ops) # From attack

        print(f"\nSynthesized: {action['label']}")
        print(f"Effects: {effects}")

    def test_unknown_concept(self):
        """Test handling of unknown concepts."""
        concepts = ['fire', 'unknown_concept_xyz']
        action = self.alchemy.synthesize_action(concepts)

        # Should ignore the unknown and just process fire
        self.assertEqual(action['id'], "action:fire_unknown_concept_xyz")
        # Fire effects should still be present
        effects = action['logic']['effects']
        self.assertTrue(any(e.get('type') == 'fire' for e in effects))

if __name__ == '__main__':
    unittest.main()
