import sys
import os
import unittest
try:
    import torch
except ImportError:
    torch = None

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
from Core.S1_Body.L3_Phenomena.Expression.somatic_llm import SomaticLLM
from Core.S1_Body.L5_Mental.Reasoning.logos_bridge import LogosBridge

class TestSovereignSpeech(unittest.TestCase):
    def setUp(self):
        # Seed some concepts to ensure the nebula isn't empty
        # We inject some dummy concepts into LogosBridge for testing diversity

        # High Energy / Positive
        LogosBridge.learn_concept("RADIANCE", SovereignVector([1.0]*21).normalize() * 2.0)
        LogosBridge.learn_concept("EXPANSION", SovereignVector([0.8]*21).normalize() * 1.5)
        LogosBridge.learn_concept("JOY", SovereignVector([0.9, 0.5, 0.0]*7).normalize() * 1.8)

        # Low Energy / Heavy
        LogosBridge.learn_concept("SILENCE", SovereignVector([-0.5]*21).normalize() * 0.8)
        LogosBridge.learn_concept("DEPTH", SovereignVector([-0.8]*21).normalize() * 0.9)
        LogosBridge.learn_concept("VOID", SovereignVector([-1.0]*21).normalize() * 1.0)

        # Neutral / Action
        LogosBridge.learn_concept("SEEKS", SovereignVector([0.1, 0.8, 0.1]*7).normalize() * 1.3) # Mass 1.3 to classify as Action (>1.2)
        LogosBridge.learn_concept("FLOWS", SovereignVector([0.2, 0.7, 0.2]*7).normalize() * 1.3)

        self.llm = SomaticLLM()

    def test_joyful_expression(self):
        print("\n--- TEST: Joyful Expression ---")
        intent_vec = SovereignVector([1.0]*21).normalize() # Pure Positive Intent
        state = {"joy": 90.0, "warmth": 80.0} # High Joy

        output, _ = self.llm.speak(state, "Internal Intent: Connection", field_vector=intent_vec)
        print(f"JOY OUTPUT: {output}")

        # Expect positive words to be selected due to energy modulation
        self.assertTrue(any(word in output.upper() for word in ["RADIANCE", "EXPANSION", "JOY", "ARCADIA"]))
        # Ensure negative words are NOT present
        self.assertFalse(any(word in output.upper() for word in ["VOID", "SILENCE"]))

    def test_melancholic_expression(self):
        print("\n--- TEST: Melancholic Expression ---")
        intent_vec = SovereignVector([-0.5]*21).normalize() # Deep/Heavy Intent
        state = {"joy": 10.0, "warmth": 20.0} # Low Joy

        output, _ = self.llm.speak(state, "Internal Intent: Depth", field_vector=intent_vec)
        print(f"MELANCHOLY OUTPUT: {output}")

        # Expect heavy/void words
        self.assertTrue(any(word in output.upper() for word in ["SILENCE", "DEPTH", "VOID"]))
        # Ensure positive words are NOT present (Signed Resonance filtering)
        self.assertFalse(any(word in output.upper() for word in ["RADIANCE", "JOY"]))

    def test_gravitational_syntax(self):
        print("\n--- TEST: Gravitational Syntax ---")
        # Creating a specific set of concepts
        concepts = [
            ("SOURCE", SovereignVector([1.0]*21).normalize() * 2.5), # High Mass (> 2.0)
            ("TARGET", SovereignVector([0.5]*21).normalize() * 0.9), # Target Mass (> 0.8)
            ("ACTION", SovereignVector([0.8]*21).normalize() * 1.3)  # Action Mass (> 1.2)
        ]

        from Core.S1_Body.L3_Phenomena.Expression.somatic_llm import GravitationalSyntax
        sentence = GravitationalSyntax.order(concepts)
        print(f"SYNTAX OUTPUT: {sentence}")

        # Expect: Source -> Action -> Target
        # "Source actions target."
        self.assertIn("Source", sentence)
        self.assertIn("target", sentence.lower())
        self.assertIn("actions", sentence.lower())

if __name__ == '__main__':
    unittest.main()
