import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from Core.System.autonomic_nervous_system import AutonomicNervousSystem
from Core.Monad.ouroboros_loop import OuroborosLoop

class MockEngineHolistic:
    def __init__(self, curiosity=0.1, strain=0.05, joy=0.1, visual=0.0, memory=0.0):
        self.curiosity = curiosity
        self.strain = strain
        self.joy = joy
        self.visual = visual
        self.memory = memory

    def read_holistic_state(self):
        return {
            'curiosity': self.curiosity,
            'strain': self.strain,
            'joy': self.joy,
            'visual_echo': self.visual,
            'memory_weight': self.memory
        }

class MockOuroboros:
    def __init__(self):
        self.dream_depth = 0.5
        self.cycled = False

    def dream_cycle(self):
        self.cycled = True

class TestAutonomicNervousSystem(unittest.TestCase):
    def setUp(self):
        self.engine = MockEngineHolistic(curiosity=0.2, strain=0.1, joy=0.3)
        self.ouroboros = MockOuroboros()
        self.log_messages = []

        def mock_logger(msg):
            self.log_messages.append(msg)

        self.ans = AutonomicNervousSystem(self.engine, self.ouroboros, log_callback=mock_logger)

    def test_breath_pulse_accumulation(self):
        # Initial state should not trigger a spill-over immediately
        self.ans.breath_pulse()
        self.assertTrue(self.ouroboros.cycled)
        self.assertEqual(len(self.log_messages), 0)
        self.assertGreater(self.ans.sympathetic_drive, 0)
        self.assertGreater(self.ans.parasympathetic_drive, 0)

    def test_spill_over_expansion(self):
        # Force high curiosity to trigger Expansion
        self.engine.curiosity = 1.0
        self.engine.joy = 1.0

        self.ans.breath_pulse()
        self.assertTrue(any("Sympathetic Overload" in msg for msg in self.log_messages))
        self.assertEqual(self.ans.sympathetic_drive, 0.0)

    def test_spill_over_contraction(self):
        # Force high strain to trigger Contraction
        self.engine.curiosity = 0.0
        self.engine.joy = 0.0
        self.engine.strain = 1.0

        self.ans.breath_pulse()
        self.assertTrue(any("Parasympathetic Overload" in msg for msg in self.log_messages))
        self.assertEqual(self.ans.parasympathetic_drive, 0.0)

if __name__ == '__main__':
    unittest.main()
