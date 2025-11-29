
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from Core.Language.dialogue.dialogue_engine import DialogueEngine

class TestDialogueIntegration(unittest.TestCase):
    def setUp(self):
        self.engine = DialogueEngine()
        
        # Inject a concept "Love" into memory
        # We need to ensure it has a vector
        sphere = self.engine.memory.add_concept("Love", metadata={"type": "emotion"})
        # Manually set vector for testing (if add_concept didn't set a distinct one)
        # In real app, vector comes from embedding or random init. 
        # Here we just need it to exist in Resonance Engine.
        # add_concept calls resonance.add_vector, so it should be there.
        
        # Inject "Compassion" with similar vector
        self.engine.memory.add_concept("Compassion")
        # Force vectors to be similar
        self.engine.memory.resonance.add_vector("Love", [1.0, 0.0, 0.0])
        self.engine.memory.resonance.add_vector("Compassion", [0.9, 0.1, 0.0])
        
    def test_resonance_recall(self):
        # Ask about Love
        # Should trigger resonance with Compassion
        memories = self.engine._recall_memories("Love")
        
        found_compassion = False
        for mem in memories:
            if "Compassion" in mem:
                found_compassion = True
                break
        
        self.assertTrue(found_compassion, "Dialogue should recall 'Compassion' when asked about 'Love' due to resonance.")

    def test_response_generation(self):
        # Test simple response
        response = self.engine.respond("Hello")
        self.assertIsNotNone(response)
        self.assertTrue("Hello" in response or "Hi" in response or "안녕" in response)

if __name__ == "__main__":
    unittest.main()
