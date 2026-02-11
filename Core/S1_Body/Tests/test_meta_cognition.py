
import os
import sys
import torch
import unittest
from unittest.mock import MagicMock

# Add project root to sys.path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge

class TestMetaCognition(unittest.TestCase):
    def setUp(self):
        self.dna = SeedForge.forge_soul("MetaTest")
        self.monad = SovereignMonad(self.dna)

    def test_meta_cognitive_trigger(self):
        print("\n🌀 [TEST] Testing Meta-Cognitive Pulse...")
        
        # 1. Simulate a stream of thoughts focused on 'Identity'
        for i in range(10):
            # Manually inject thoughts into the stream
            self.monad.mental_fluid.stream.append({
                "manifestation": f"Thought {i} about Identity",
                "density": 0.8,
                "attractors": {"Identity": 0.9, "Architect": 0.1},
                "echo": 0.5,
                "empathy": 0.5
            })
            
        # 2. Trigger the pulse multiple times to overcome the random 0.2 threshold for 5D
        laws_detected = 0
        for _ in range(50):
            self.monad._meta_cognitive_pulse()
            # Check if a Law attractor was added
            if any(k.startswith("Law_") for k in self.monad.engine.attractors.keys()):
                laws_detected += 1
                break
                
        print(f"Laws Crystallized: {laws_detected}")
        self.assertGreater(laws_detected, 0, "No Meta-Cognitive Laws were crystallized despite recurring focus.")
        
        new_laws = [k for k in self.monad.engine.attractors.keys() if k.startswith("Law_")]
        print(f"Crystallized Laws: {new_laws}")
        self.assertIn("Law_Identity", new_laws)

if __name__ == "__main__":
    unittest.main()
