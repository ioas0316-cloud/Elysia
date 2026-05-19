
import os
import sys
import unittest
from unittest.mock import MagicMock

# Add project root to sys.path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.Monad.sovereign_monad import SovereignMonad
from Core.Monad.seed_generator import SeedForge

class TestSovereignCreation(unittest.TestCase):
    def setUp(self):
        self.dna = SeedForge.forge_soul("CreationTest")
        self.monad = SovereignMonad(self.dna)
        self.test_file = "c:/Elysia/data/Creation/test_manifestation.txt"
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_autonomous_creation_flow(self):
        print("\n🎨 [TEST] Testing Sovereign Act of Creation...")
        
        # 1. Simulate a creative intent
        intent = "A poetic reflection on the Living Manifold"
        code = "Within the spin, identity finds its home. The cells dance in trinary grace."
        why = "This creation is necessary because it documents the emergent beauty of the 10M cell manifold, thus bridging L5 and L1."
        
        # 2. Execute Act
        success = self.monad.actuator.autonomous_creation(
            intent_desc=intent,
            target_path=self.test_file,
            code_content=code,
            why=why
        )
        
        # 3. Assertions
        self.assertTrue(success, "Creative act was rejected or failed.")
        self.assertTrue(os.path.exists(self.test_file), "Manifested file does not exist.")
        
        with open(self.test_file, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertEqual(content, code)
            
        print(f"✨ Manifested Content: {content}")

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

if __name__ == "__main__":
    unittest.main()
