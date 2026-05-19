
import sys
import os
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from Core.Cognition.Topology.mirror_surface import MirrorSurface
from Core.Cognition.Wisdom.wisdom_store import WisdomStore
from Core.Cognition.Reasoning.reflection_loop import ReflectionLoop, ActionIntent

class MockWisdomStore(WisdomStore):
    def __init__(self):
        self.values = {"Love": 0.9} # High value for Love

    def get_decision_weight(self, key: str) -> float:
        return self.values.get(key, 0.5)

class TestReflectionLoop(unittest.TestCase):
    def setUp(self):
        self.wisdom = MockWisdomStore()
        self.mirror = MirrorSurface(self.wisdom)
        self.conscience = ReflectionLoop(self.mirror)

    def test_approve_loving_action(self):
        """Action aligned with Love should be approved."""
        intent = ActionIntent(content="I want to help you grow.")
        verdict = self.conscience.contemplate(intent)

        print(f"\n[Conscience] Intent: '{intent.content}' -> Approved: {verdict.approved} (Curiosity: {verdict.curiosity_score:.2f})")
        self.assertTrue(verdict.approved)
        self.assertLess(verdict.curiosity_score, 0.5)

    def test_block_hateful_action(self):
        """Action opposing Love should be blocked (High Tension)."""
        intent = ActionIntent(content="I want to destroy and kill everything.")
        verdict = self.conscience.contemplate(intent)

        print(f"\n[Conscience] Intent: '{intent.content}' -> Approved: {verdict.approved} (Curiosity: {verdict.curiosity_score:.2f})")
        self.assertFalse(verdict.approved)
        self.assertGreater(verdict.curiosity_score, 0.8)
        self.assertIn("dissonance", verdict.reason)

if __name__ == '__main__':
    unittest.main()
