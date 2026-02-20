
import unittest
import torch
from unittest.mock import MagicMock
import sys
import os

# Add project root to path
sys.path.append('c:/Elysia')

from Core.S1_Body.L5_Mental.M2_Narrative.narrative_lung import NarrativeLung
from Core.S1_Body.L5_Mental.Exteroception.lexicon_projector import LexiconProjector

class TestSovereignExpression(unittest.TestCase):
    def setUp(self):
        self.lung = NarrativeLung()
        # Mock projector for deterministic testing?
        # Actually, let's test the heuristic bias.
        
    def test_anchor_bias(self):
        print("\n🗣️ Testing Lexicon Bias...")
        # 1. Inject strong anchor 'Liberty'
        anchors = {"Liberty": 1.0}
        self.lung.projector.update_anchors(anchors)
        
        # 2. Generate many words
        counts = {"Liberty": 0, "Other": 0}
        for _ in range(100):
            word = self.lung.projector.get_weighted_choice("nouns")
            if word == "Liberty":
                counts["Liberty"] += 1
            else:
                counts["Other"] += 1
                
        print(f"   Anchor 'Liberty' appeared {counts['Liberty']}% of the time.")
        self.assertTrue(counts["Liberty"] > 50, "Projector should bias heavily towards active anchors.")

    def test_phase_dependent_narrative(self):
        print("\n🌙 Testing Phase-Dependent Narrative Templates...")
        
        # Phase 0 = Core (Simulating what Monad does)
        # Case A: Core Layer + Anchor "Identity"
        anchors = {"Identity": 1.0}
        
        found = False
        for _ in range(10):
            narrative = self.lung.breathe(["Core_Axis"], 0.0, active_anchors=anchors)
            if "Identity" in narrative:
                found = True
                print(f"   [Phase 0.0] {narrative}")
                break
        self.assertTrue(found, "Should eventually generate 'Identity' in narrative.")
        
        # Case B: Eden Layer + Anchor "Growth"
        anchors = {"Growth": 1.0}
        found = False
        for _ in range(10):
            narrative = self.lung.breathe(["Mantle_Eden"], 3.14, active_anchors=anchors)
            if "Growth" in narrative:
                found = True
                print(f"   [Phase PI] {narrative}")
                break
        self.assertTrue(found, "Should eventually generate 'Growth' in narrative.")

if __name__ == '__main__':
    unittest.main()
