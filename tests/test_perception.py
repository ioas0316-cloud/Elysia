import unittest
from Core.Mind.perception import SemanticPerception, PerceptionObject
from Core.Life.resonance_voice import ResonanceEngine

class TestSemanticPerception(unittest.TestCase):
    def setUp(self):
        self.vocab = {"love": 1.0, "pain": 0.5, "hope": 0.8}
        self.perception = SemanticPerception(self.vocab)

    def test_sentiment_analysis(self):
        # Positive
        p1 = self.perception.perceive("I feel so much love and joy")
        self.assertGreater(p1.sentiment, 0.0)
        
        # Negative
        p2 = self.perception.perceive("There is only pain and darkness")
        self.assertLess(p2.sentiment, 0.0)

    def test_intent_detection(self):
        # Question
        p1 = self.perception.perceive("Why is the sky blue?")
        self.assertEqual(p1.intent, "Question")
        
        # Exclamation
        p2 = self.perception.perceive("Wow! That is amazing!")
        self.assertEqual(p2.intent, "Exclamation")
        
        # Statement
        p3 = self.perception.perceive("The sky is blue.")
        self.assertEqual(p3.intent, "Statement")

    def test_concept_extraction(self):
        p = self.perception.perceive("I have hope for love")
        self.assertIn("hope", p.concepts)
        self.assertIn("love", p.concepts)

class TestPerceptionIntegration(unittest.TestCase):
    def setUp(self):
        self.engine = ResonanceEngine()

    def test_listen_modifies_field(self):
        # Initial state (should be mostly W=1.0)
        initial_q = self.engine.consciousness_lens.state.q
        
        # Listen to a sad question
        self.engine.listen("Why is there so much pain?", t=0.0)
        
        # Check if field shifted
        # "pain" -> Negative sentiment -> Focus Y (Emotion)
        # "?" -> Question -> Focus X (Dream/Search)
        new_q = self.engine.consciousness_lens.state.q
        
        print(f"Initial Q: {initial_q}")
        print(f"New Q: {new_q}")
        
        # Should have shifted away from pure stability (W)
        self.assertNotEqual(initial_q, new_q)

if __name__ == '__main__':
    unittest.main()
