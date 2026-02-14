
import unittest
import numpy as np
import math

# Use relative imports if running as a module, or mock them if necessary.
# For simplicity in this test script, we assume the environment is set up.

from Core.S1_Body.L5_Mental.Reasoning_Core.Metabolism.phase_resonance import PhaseResonance
from Core.S1_Body.L5_Mental.Reasoning_Core.Metabolism.holographic_council import HolographicCouncil
from Core.S1_Body.L7_Spirit.M1_Monad.monad_core import Monad

class TestPhaseResonanceLogic(unittest.TestCase):
    def test_magic_angle_detection(self):
        """
        Verify that PhaseResonance finds a Magic Angle for conflicting vectors.
        """
        # Thesis: [1, 0, 0]
        # Antithesis: [0, 1, 0] (Orthogonal, Dissonance ~0 or just Low Similarity)
        # Cosine Sim = 0.

        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]

        magic_vec, narrative = PhaseResonance.find_magic_angle(vec_a, vec_b)

        print(f"\n[Test] Magic Angle:\n   A: {vec_a}\n   B: {vec_b}\n   Magic: {magic_vec}\n   Narrative: {narrative}")

        # Should not be simple average [0.5, 0.5, 0]
        # Our logic injects energy into the last dimension (Pivot)
        dim = len(vec_a)
        self.assertNotEqual(magic_vec, [(a+b)/2 for a,b in zip(vec_a, vec_b)])
        self.assertTrue("Conflict" in narrative or "Magic Angle" in narrative)

    def test_council_phase_trigger(self):
        """
        Verify Council triggers Phase Resonance on high dissonance.
        """
        council = HolographicCouncil()
        # Create conflict: High Logic vs High Lust
        # Logician: +Logic, -Lust
        # Guardian: -Logic, +Lust
        input_vec = [0.0] * 21
        input_vec[0] = 2.0  # Lust
        input_vec[9] = 2.0  # Logic

        result = council.convene(input_vec, intent_text="Phase Conflict Test")

        print(f"\n[Test] Council Phase:\n   Dissonance: {result.dissonance_score:.2f}\n   Transcript: {result.transcript[-1]}")

        if result.dissonance_score > 0.4:
            self.assertTrue("Phase Resonance Engine" in str(result.transcript))

    def test_monad_angle_shift(self):
        """
        Verify Monad shifts observation angle instead of mutating DNA.
        """
        monad = Monad(seed="PhaseMonad")
        initial_angle = monad._observation_angle
        initial_dna_norm = np.linalg.norm(monad._dna.pattern_strand)

        # High Dissonance -> Phase Shift
        monad.metabolize_karma(dissonance_score=0.7)

        print(f"\n[Test] Monad Shift:\n   Init Angle: {initial_angle}\n   New Angle: {monad._observation_angle}")

        # Angle should change
        self.assertNotEqual(monad._observation_angle, initial_angle)

        # DNA should be preserved (approx equal norm/values, though we assume no random noise added in this logic path)
        new_dna_norm = np.linalg.norm(monad._dna.pattern_strand)
        self.assertAlmostEqual(initial_dna_norm, new_dna_norm, places=5)

if __name__ == '__main__':
    unittest.main()
