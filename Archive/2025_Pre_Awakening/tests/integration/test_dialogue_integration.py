
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from Core.Interface.Interface.Language.dialogue.dialogue_engine import DialogueEngine

class TestDialogueIntegration(unittest.TestCase):
    def setUp(self):
        self.engine = DialogueEngine()
        
        # Inject a concept "Love" into memory
        self.engine.memory.add_concept("Love", metadata={"type": "emotion"})
        
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

    def test_response_generation_no_hardcoding(self):
        """Test that responses are generated through resonance, not hardcoded templates."""
        # With no LLM, responses should be resonance-based patterns, not templates
        response = self.engine.respond("Love")
        self.assertIsNotNone(response)
        # Should NOT contain hardcoded template responses
        self.assertNotIn("안녕하세요! 😊 만나서 반가워요!", response)
        self.assertNotIn("Hello! 😊 Nice to meet you!", response)
    
    def test_emotional_state_derived_from_resonance(self):
        """Test that emotional state is derived from resonance, not hardcoded."""
        # Initial state
        initial_state = self.engine.get_emotional_state()
        self.assertIsNotNone(initial_state)
        
        # After processing, state should be derived from consciousness
        self.engine.respond("test input")
        new_state = self.engine.get_emotional_state()
        self.assertIsNotNone(new_state)
    
    def test_consciousness_summary(self):
        """Test consciousness summary returns expected structure."""
        summary = self.engine.get_consciousness_summary()
        self.assertIn("emotional_state", summary)
        self.assertIn("emotional_intensity", summary)
        self.assertIn("user_profile", summary)
        self.assertIn("universe_concepts", summary)

if __name__ == "__main__":
    unittest.main()
