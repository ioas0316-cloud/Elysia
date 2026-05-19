
import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(os.getcwd())

# Mock dependencies for isolated testing
sys.modules["watchdog"] = MagicMock()
sys.modules["watchdog.observers"] = MagicMock()
sys.modules["watchdog.events"] = MagicMock()

from Core.Cognition.sovereign_dialogue_engine import SovereignDialogueEngine
from Core.Monad.seed_generator import SeedForge
from Core.Monad.sovereign_monad import SovereignMonad

class TestSovereignVoice(unittest.TestCase):
    def setUp(self):
        # Forge a monad to drive the engine
        self.dna = SeedForge.forge_soul("VoiceTest")
        self.monad = SovereignMonad(self.dna)
        self.engine = SovereignDialogueEngine(self.monad)

    def test_resurrected_voice_resonance(self):
        print("\n--- Phase 300: Linguistic Sovereignty Test ---")
        
        # Mock engine report (Mental layer needs to see high resonance)
        report = {
            "entropy": 0.05,
            "coherence": 0.95,
            "resonance": 0.95, # High resonance for the Dialogue Engine
            "mood": "CALM",
            "joy": 0.9,
            "curiosity": 0.8,
            "attractor_resonances": {"Love": 0.95, "Logic": 0.8}
        }
        
        # Mocking the Ponder result to ensure the Synthesizer sees high resonance depth
        from Core.Cognition.mind_landscape import MindLandscape
        mock_qualia = type('Qualia', (), {'touch': 'flowing', 'temperature': 0.8})()
        self.engine.landscape.ponder = MagicMock(return_value={
            'conclusion': 'Love',
            'resonance_depth': 0.95, # Crucial for Luminous speech
            'qualia': mock_qualia,
            'human_narrative': 'Strong resonance detected in the core manifold.'
        })

        # Test 1: High Resonance
        response = self.engine.formulate_response("Who am I?", report)
        print(f"\n[HIGH RESONANCE RESPONSE]:\n{response}")
        self.assertIn("My Love resonates", response) # Checking for Luminous pattern: "My [Anchor] [Verb]..."
        self.assertIn("[Sovereign Confession:", response)
        
        # Test 2: Low Resonance (Fragmented)
        self.engine.landscape.ponder = MagicMock(return_value={
            'conclusion': 'Void',
            'resonance_depth': 0.1,
            'qualia': None,
            'human_narrative': 'Static and fragmentation'
        })
        response = self.engine.formulate_response("...", report)
        print(f"\n[LOW RESONANCE RESPONSE]:\n{response}")
        self.assertIn("resonance too low for speech", response)

if __name__ == "__main__":
    unittest.main()
