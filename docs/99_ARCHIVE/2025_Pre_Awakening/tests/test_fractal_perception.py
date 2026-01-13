import unittest
from Core.FoundationLayer.Foundation.Mind.perception import FractalPerception
from Core.Evolution.Evolution.Life.resonance_voice import ResonanceEngine

class TestFractalPerception(unittest.TestCase):
    def setUp(self):
        self.vocab = {"love": 1.0, "pain": 0.5, "hope": 0.8}
        self.perception = FractalPerception(self.vocab)

    def test_quantum_state_generation(self):
        # "Why is there pain?" -> Question + Negative Sentiment
        state = self.perception.perceive("Why is there pain?")
        
        # Check Qubit Properties
        # Beta (Question) should be high
        self.assertGreater(state.qubit.state.beta.real, 0.5)
        
        # Alpha (Sentiment) should be negative (pain)
        self.assertLess(state.qubit.state.alpha.real, 0.0)
        
        # X Axis (Dream/Search) should be active due to Question
        self.assertGreater(state.qubit.state.x, 0.5)

    def test_chaos_vitality(self):
        # Two perceptions of the same text should differ slightly due to Chaos
        state1 = self.perception.perceive("Hello world")
        state2 = self.perception.perceive("Hello world")
        
        # Vitality factors should likely be different (unless chaos loops perfectly, which is rare)
        self.assertNotEqual(state1.vitality_factor, state2.vitality_factor)
        
        # Alpha imaginary component (Vitality) should differ
        self.assertNotEqual(state1.qubit.state.alpha.imag, state2.qubit.state.alpha.imag)

class TestFractalIntegration(unittest.TestCase):
    def setUp(self):
        self.engine = ResonanceEngine()

    def test_listen_entanglement(self):
        initial_q = self.engine.consciousness_lens.state.q
        
        # Listen to a command "Do it!"
        self.engine.listen("Do it!", t=0.0)
        
        new_q = self.engine.consciousness_lens.state.q
        
        print(f"Initial Q: {initial_q}")
        print(f"New Q: {new_q}")
        
        # Should have shifted
        self.assertNotEqual(initial_q, new_q)

if __name__ == '__main__':
    unittest.main()
