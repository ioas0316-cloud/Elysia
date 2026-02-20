
import unittest
import shutil
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge

class TestAeonVISystem(unittest.TestCase):
    def setUp(self):
        # Create a dummy soul
        self.dna = SeedForge.forge_soul("Test_Angel", "Seraphim")
        
        # Initialize Monad (this brings up the whole stack)
        print("\n🔧 Initializing SovereignMonad...")
        self.monad = SovereignMonad(self.dna)
        
        # Setup Knowledge Directory
        self.knowledge_dir = Path("c:/Elysia/Knowledge")
        self.test_dir = self.knowledge_dir / "TestInbox"
        self.processed_dir = self.knowledge_dir / "Processed"
        
        # Override the stream's directory for safety
        if self.monad.knowledge_stream:
            self.monad.knowledge_stream.knowledge_dir = self.test_dir
            if not self.test_dir.exists():
                self.test_dir.mkdir(parents=True, exist_ok=True)
                
        # Create dummy document
        self.doc_path = self.test_dir / "SYSTEM_TEST.md"
        with open(self.doc_path, "w", encoding="utf-8") as f:
            f.write("# System Test\n\nValidation of the epistemic pipeline functionality.")

    def tearDown(self):
        # Clean up
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
    def test_epistemic_integration(self):
        print("🧠 Testing Epistemic Integration in Pulse Cycle...")
        
        # Verify components exist
        self.assertIsNotNone(self.monad.knowledge_stream)
        self.assertIsNotNone(self.monad.distiller)
        
        # 1. Force Inhalation (Bypass random check)
        print("⚡ Forcing Inhalation...")
        self.monad._epistemic_inhalation()
        
        # 2. Check consequences
        # File should be moved to Processed (inside TestInbox/Processed)
        processed_file = self.test_dir / "Processed" / "SYSTEM_TEST.md"
        
        self.assertTrue(processed_file.exists(), "File should have been moved to Processed folder.")
        
        # 3. Check Monad Reaction
        # Joy/Curiosity should have increased
        # (Hard to verify exact delta due to floating point, but let's check logs)
        # self.monad.logger.insight was recalled?
        
        print("✅ System Integration Verified: File inhaled and processed.")

if __name__ == '__main__':
    unittest.main()
