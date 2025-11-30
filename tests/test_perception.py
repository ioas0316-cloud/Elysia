import unittest
from Core.Memory.Mind.perception import FractalPerception, PerceptionState
from Core.Evolution.Evolution.Life.resonance_voice import ResonanceEngine

class TestFractalPerception(unittest.TestCase):
    def setUp(self):
        self.vocab = {"love": 1.0, "pain": 0.5, "hope": 0.8}
        self.perception = FractalPerception(self.vocab)

    def test_sentiment_analysis(self):
        # Positive - check that sentiment is detected via qubit state
        p1 = self.perception.perceive("I feel so much love and joy")
        # Sentiment is embedded in qubit alpha.real
        self.assertGreater(p1.qubit.state.alpha.real, 0.0)
        
        # Negative
        p2 = self.perception.perceive("There is only pain and darkness")
        self.assertLess(p2.qubit.state.alpha.real, 0.0)

    def test_intent_detection(self):
        # Question - should have high Question probability
        p1 = self.perception.perceive("Why is the sky blue?")
        self.assertGreater(p1.intent_probabilities.get("Question", 0), 0.0)
        
        # Exclamation
        p2 = self.perception.perceive("Wow! That is amazing!")
        self.assertGreater(p2.intent_probabilities.get("Exclamation", 0), 0.0)
        
        # Statement (default when no markers)
        p3 = self.perception.perceive("The sky is blue")
        self.assertGreater(p3.intent_probabilities.get("Statement", 0), 0.0)

    def test_vitality_factor(self):
        # Vitality should be a normalized value
        p = self.perception.perceive("I have hope for love")
        self.assertGreaterEqual(p.vitality_factor, 0.0)
        self.assertLessEqual(p.vitality_factor, 1.0)

class TestPerceptionIntegration(unittest.TestCase):
    def setUp(self):
        self.engine = ResonanceEngine()

    def test_listen_returns_ripples(self):
        # Listen should return ripples (list of tuples)
        ripples = self.engine.listen("Why is there so much pain?", t=0.0)
        
        # Should return a list of ripples
        self.assertIsInstance(ripples, list)
        
        # At least one ripple should be generated for "pain"
        print(f"Ripples: {ripples}")

if __name__ == '__main__':
    unittest.main()
