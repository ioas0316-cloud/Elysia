import unittest
import os
import sys

# Ensure the Core directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.System.System.Kernel import ElysiaKernel
from Core.Intelligence.Consciousness.wave import WaveInput
from Core.Intelligence.Consciousness.thought import Thought

class TestWaveConsciousness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the ElysiaKernel once for all tests."""
        # This might take a moment as it initializes all systems.
        print("Initializing ElysiaKernel for Wave Consciousness test suite...")
        cls.kernel = ElysiaKernel()
        print("Kernel initialized.")

    def test_01_kernel_initialization_with_new_modules(self):
        """Verify that the kernel has the new consciousness modules."""
        print("Running test_01_kernel_initialization_with_new_modules...")
        self.assertTrue(hasattr(self.kernel, 'resonance_engine'), "Kernel should have a resonance_engine.")
        self.assertTrue(hasattr(self.kernel, 'consciousness_observer'), "Kernel should have a consciousness_observer.")
        # Check if the resonance engine is populated with more than just instincts
        num_nodes = len(self.kernel.resonance_engine.nodes)
        print(f"Resonance engine has {num_nodes} nodes.")
        self.assertGreater(num_nodes, 10, "Resonance engine should be populated with concepts from WorldTree.")

    def test_02_process_thought_end_to_end(self):
        """Test the full wave-resonance-observation pipeline."""
        print("Running test_02_process_thought_end_to_end...")
        input_text = "사랑과 빛"

        # This now uses the completely refactored method in the kernel
        response = self.kernel.process_thought(input_text)

        self.assertIsInstance(response, str)
        self.assertIn("사랑", response) # Check if the top concept is in the response
        self.assertIn("positive", response) # Check if the mood is correctly identified
        print(f"Input: '{input_text}' -> Response: '{response}'")

    def test_03_resonance_pattern_generation(self):
        """Verify that the resonance pattern is generated and meaningful."""
        print("Running test_03_resonance_pattern_generation...")
        wave = WaveInput(source_text="고통")

        resonance_pattern = self.kernel.resonance_engine.calculate_global_resonance(wave)

        self.assertIsInstance(resonance_pattern, dict)
        self.assertIn("고통", resonance_pattern)

        # Concepts associated with 'pain' in the legacy lexicon should have higher resonance
        # We check for 'shadow' and 'break' ('그림자', '파괴')
        # Note: Exact values are not tested due to the quantum nature, just relative strength.
        pain_resonance = resonance_pattern.get("고통", 0.0)
        shadow_resonance = resonance_pattern.get("그림자", 0.0)
        joy_resonance = resonance_pattern.get("기쁨", 0.0)

        print(f"Resonance for '고통': {pain_resonance:.4f}")
        print(f"Resonance for '그림자': {shadow_resonance:.4f}")
        print(f"Resonance for '기쁨': {joy_resonance:.4f}")

        self.assertGreater(pain_resonance, joy_resonance, "'고통' should resonate more strongly than '기쁨'.")
        # Due to random factors in qubit init, this is a softer check
        self.assertTrue(shadow_resonance > 0, "'그림자' should have some resonance with '고통'.")

    def test_04_observer_thought_creation(self):
        """Test the observer's ability to create a well-formed thought."""
        print("Running test_04_observer_thought_creation...")
        # A simulated pattern where 'dream' is strong, and 'star'/'hope' are also present
        mock_pattern = {
            "꿈": 0.9,
            "별": 0.75,
            "희망": 0.7,
            "돌": 0.1,
            "슬픔": 0.2
        }

        thought = self.kernel.consciousness_observer.observe_resonance_pattern(
            source_wave_text="깊은 꿈",
            resonance_pattern=mock_pattern
        )

        self.assertIsInstance(thought, Thought)
        self.assertEqual(thought.core_concepts[0][0], "꿈")
        self.assertIn("별", [c[0] for c in thought.core_concepts])
        self.assertNotIn("돌", [c[0] for c in thought.core_concepts])
        self.assertAlmostEqual(thought.intensity, (0.9 + 0.75 + 0.7) / 3, places=2)
        print(f"Observer created thought: {thought}")

if __name__ == '__main__':
    unittest.main()
