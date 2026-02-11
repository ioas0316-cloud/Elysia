import unittest
import torch
import shutil
import os
import sys
from unittest.mock import MagicMock
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.S1_Body.L5_Mental.Exteroception.knowledge_stream import KnowledgeStream
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge

class TestEpistemicInhalation(unittest.TestCase):
    def setUp(self):
        # Mock Engine
        self.mock_engine = MagicMock()
        self.mock_engine.device = "cpu"
        self.mock_engine.grid_shape = (10, 10)
        self.mock_engine.reconfigure_topography = MagicMock()
        
        # Test directory
        self.test_dir = Path("c:/Elysia/Knowledge/TestDocs")
        if not self.test_dir.exists():
            self.test_dir.mkdir(parents=True, exist_ok=True)
            
        self.stream = KnowledgeStream(self.mock_engine)

    def tearDown(self):
        # Clean up test files
        shutil.rmtree(self.test_dir)

    def test_inhale_doctrine_text(self):
        print("\n📚 Testing Doctrine Inhalation...")
        
        # Create a test doctrine
        doctrine_path = self.test_dir / "DOCTRINE_OF_BLUE.md"
        with open(doctrine_path, "w", encoding="utf-8") as f:
            f.write("# Doctrine of Blue\n\nThe sky is blue because of Rayleigh scattering. Blue is a calm color.")
            
        # Inhale
        distilled_count = self.stream.inhale_file(str(doctrine_path))
        
        # Verify
        print(f"Distilled Count: {distilled_count}")
        self.assertTrue(distilled_count > 0, "Should distill at least one concept (Blue, Sky, Rayleigh)")
        
        # Verify engine call
        self.mock_engine.reconfigure_topography.assert_called()
        
        # Check specific calls (heuristic)
        calls = self.mock_engine.reconfigure_topography.call_args_list
        concepts = [c[0][0] for c in calls] # args[0] is 'name'
        print(f"Distilled Concepts: {concepts}")
        
        self.assertIn("blue", [c.lower() for c in concepts])

if __name__ == '__main__':
    unittest.main()
