import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.fractal_kernel import FractalKernel
from Core.FoundationLayer.Foundation.genesis_cortex import GenesisCortex

class TestGenesisCortex(unittest.TestCase):
    def setUp(self):
        self.kernel = FractalKernel()
        self.genesis = GenesisCortex(self.kernel)

    @patch('Project_Sophia.genesis_cortex.generate_text')
    def test_evolve_landscape(self, mock_generate_text):
        """Verify that GenesisCortex can modify the field."""
        print("\nTesting Genesis Cortex Evolution...")
        
        # Mock LLM response with a plan to add a "Freedom" well
        mock_plan = {
            "rationale": "The user desires freedom.",
            "add_wells": [
                {"x": 50.0, "y": 50.0, "strength": 30.0, "label": "Freedom"}
            ],
            "add_rails": []
        }
        mock_generate_text.return_value = json.dumps(mock_plan)
        
        # Initial state check
        initial_wells = len(self.kernel.field.wells)
        print(f"Initial Wells: {initial_wells}")
        
        # Trigger evolution
        recent_thoughts = ["I want to break free", "Freedom is key"]
        self.genesis.evolve_landscape(recent_thoughts)
        
        # Final state check
        final_wells = len(self.kernel.field.wells)
        print(f"Final Wells: {final_wells}")
        
        self.assertEqual(final_wells, initial_wells + 1, "A new Gravity Well should have been added")
        
        # Verify the new well properties
        new_well = self.kernel.field.wells[-1]
        self.assertEqual(new_well.pos.x, 50.0)
        self.assertEqual(new_well.strength, 30.0)
        print("New Gravity Well 'Freedom' verified at (50, 50).")

if __name__ == "__main__":
    unittest.main()
