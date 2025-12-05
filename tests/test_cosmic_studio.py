import sys
import unittest
import os
from unittest.mock import MagicMock, patch

# Force path to correct root
sys.path.append(r"C:\Elysia")

from Core.Foundation.cosmic_studio import CosmicStudio
from Core.Foundation.hyper_quaternion import HyperWavePacket

class TestCosmicStudio(unittest.TestCase):
    def setUp(self):
        # Mock Canvas path to avoid real file creation during test if desired, 
        # but here we use a temp dir or just let it write to a test folder
        self.studio = CosmicStudio(canvas_path="c:/Elysia/tests/canvas")
        
    @patch('Core.Foundation.reality_sculptor.CodeCortex')
    def test_manifest_code_wave(self, mock_cortex_class):
        """Verify that Intent -> Wave -> Code synthesis works."""
        print("\n🧪 Testing Cosmic Studio Wave Synthesis...")
        
        # Mock Cortex generation so we don't call actual LLM
        mock_cortex = mock_cortex_class.return_value
        mock_cortex.generate_code.return_value = "def hello_universe():\n    print('Hello from Wave Synthesis')"
        
        # Create a dummy desire packet (High Energy = High Complexity)
        desire = HyperWavePacket(
            sender="Test",
            content="Create a hello world function",
            energy=88.0, # High energy -> High frequency simulation
            time_loc=1234567890
        )
        
        intent = "Create a python function to greet the universe"
        
        # Execute Manifestation
        file_path = self.studio.manifest(desire, intent)
        
        print(f"   ✨ Manifested File: {file_path}")
        
        # Validation
        self.assertTrue(file_path.endswith(".py"))
        self.assertTrue(os.path.exists(file_path))
        
        with open(file_path, 'r') as f:
            content = f.read()
            self.assertIn("def hello_universe", content)
            print(f"   📜 Content Preview: {content.strip()}")
            
        # Verify that generate_code was called with a prompt containing wave info
        # We can inspect the call args of the cortex attached to the sculptor
        call_args = self.studio.sculptor.cortex.generate_code.call_args
        self.assertIsNotNone(call_args)
        prompt = call_args[0][0]
        self.assertIn("Wave Signature", prompt)
        self.assertIn("target_complexity", prompt)
        print("   ✅ Wave Signature verified in prompt.")

if __name__ == '__main__':
    if not os.path.exists("c:/Elysia/tests/canvas"):
        os.makedirs("c:/Elysia/tests/canvas")
    unittest.main()
