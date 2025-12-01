import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.fractal_kernel import FractalKernel

class TestFractalPhysics(unittest.TestCase):
    def setUp(self):
        self.kernel = FractalKernel()

    def test_mind_landscape_initialization(self):
        """Verify that the field is initialized with wells and rails."""
        print("\nTesting Mind Landscape Initialization...")
        self.assertTrue(len(self.kernel.field.wells) > 0, "Gravity Wells should be initialized")
        self.assertTrue(len(self.kernel.field.rails) > 0, "Railgun Channels should be initialized")
        print(f"Wells: {len(self.kernel.field.wells)}, Rails: {len(self.kernel.field.rails)}")

    @patch('Project_Sophia.fractal_kernel.generate_text')
    def test_process_with_physics(self, mock_generate_text):
        """Verify that process() spawns particles and uses physics context."""
        print("\nTesting Process with Physics...")
        
        # Mock LLM response to avoid API calls
        mock_generate_text.return_value = "Resonated Thought"
        
        signal = "Who am I?"
        response = self.kernel.process(signal, max_depth=1)
        
        # Check if particle was spawned
        self.assertTrue(len(self.kernel.field.particles) > 0, "Particle should be spawned during process")
        particle = self.kernel.field.particles[0]
        print(f"Particle Spawned: {particle.id} at ({particle.pos.x}, {particle.pos.y})")
        
        # Check if physics context was passed to LLM
        call_args = mock_generate_text.call_args[0][0]
        self.assertIn("[Physics Context]", call_args, "Physics Context should be in the prompt")
        print("Physics Context verified in prompt.")

if __name__ == "__main__":
    unittest.main()
