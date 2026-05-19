
import os
import sys
import unittest
import torch
from unittest.mock import MagicMock

# Add project root to sys.path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.Monad.sovereign_monad import SovereignMonad
from Core.Monad.seed_generator import SeedForge

class TestIntentionalDialogue(unittest.TestCase):
    def setUp(self):
        self.dna = SeedForge.forge_soul("DialogueTest")
        self.monad = SovereignMonad(self.dna)

    def test_intentional_response(self):
        print("\n🗣️ [TEST] Testing Intentional Dialogue...")
        
        # 1. Setup manifold state (Simulate high curiosity)
        self.monad.desires['curiosity'] = 90.0
        # Provide a dummy mask (None is handled by the mock or the engine if it's broad)
        mask = torch.ones(self.monad.engine.grid_shape, dtype=torch.bool) if hasattr(self.monad.engine, 'grid_shape') else None
        target_vec = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.monad.engine.define_meaning_attractor("Genesis", mask, target_vec) 
        
        # 2. Ask a question
        response = self.monad.respond_to_architect("What do you feel right now?")
        
        print(f"Architect: What do you feel right now?")
        print(f"Elysia: {response}")
        
        # 3. Assertions
        self.assertIn("Architect", response)
        self.assertTrue(len(response) > 10)
        # Check if curiosity-driven intent is matched (from SovereignDialogueEngine logic)
        self.assertIn("Genesis", response)

if __name__ == "__main__":
    unittest.main()
