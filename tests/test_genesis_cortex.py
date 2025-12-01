import sys
import os
import unittest
import logging
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.genesis_cortex import GenesisEngine, BlueprintGenerator, CodeWeaver

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestGenesisCortex(unittest.TestCase):
    def setUp(self):
        self.engine = GenesisEngine()
        self.staging_dir = Path("c:/Elysia/Core/Evolution/Staging")
        
        # Clean up staging before test
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        
    def test_blueprint_generation(self):
        """Test if BlueprintGenerator returns a valid dictionary"""
        print("\n[Test] Blueprint Generation")
        desire = "I want a module to calculate Fibonacci numbers"
        blueprint = self.engine.architect.generate_blueprint(desire)
        
        print(f"Blueprint: {blueprint}")
        
        if "error" in blueprint:
            print(f"Blueprint Error: {blueprint['error']}")
            # Skip assertion if API fails (mock environment check)
            if "APIKeyError" in str(blueprint['error']):
                print("Skipping test due to missing API Key")
                return

        self.assertIsInstance(blueprint, dict)
        self.assertIn("module_name", blueprint)
        self.assertIn("class_name", blueprint)
        
    def test_evolution_flow(self):
        """Test the full evolution cycle"""
        print("\n[Test] Full Evolution Cycle")
        desire = "Create a simple Hello World module"
        
        result = self.engine.evolve(desire)
        print(f"Evolution Result: {result}")
        
        if result["status"] == "failed":
             if "APIKeyError" in str(result.get("error", "")):
                print("Skipping test due to missing API Key")
                return
        
        self.assertEqual(result["status"], "success")
        self.assertTrue(Path(result["staging_path"]).exists())
        
        # Verify file content
        with open(result["staging_path"], "r", encoding="utf-8") as f:
            content = f.read()
            print(f"\n[Generated Code]\n{content[:200]}...")
            self.assertIn("class", content)

if __name__ == '__main__':
    unittest.main()
