"""
[PHASE 82] Narrative Dialectics Verification
==============================================
Validates that Elysia can detect and react to internal contradictions (Dialectical Friction).
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Path Unification
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
from Core.S1_Body.L5_Mental.Reasoning.epistemic_learning_loop import EpistemicLearningLoop
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L7_Spirit.Providence.covenant_enforcer import CovenantEnforcer, Verdict

class TestNarrativeDialectics(unittest.TestCase):
    def setUp(self):
        # 1. Forge a mock soul and monad
        self.soul = SeedForge.forge_soul("TestDialectics")
        self.monad = SovereignMonad(self.soul)
        
        # 2. Setup Loop
        self.loop = EpistemicLearningLoop(root_path=".")
        self.loop.set_monad(self.monad)
        
        # 3. Setup Covenant (Gate)
        self.covenant = CovenantEnforcer()
        # Clean diary for test
        if os.path.exists(self.covenant.diary_path):
            os.remove(self.covenant.diary_path)
        self.covenant._ensure_diary()

    def test_dialectical_friction_detection(self):
        print("\n>>> Test: Dialectical Friction (모순의 자각)")
        print("-" * 50)
        
        # 1. Establish an existing law (Unity)
        self.loop.accumulated_wisdom.append({
            "name": "Axiom of love_concept",
            "description": "Love is Unity.",
            "status": "SANCTIFIED"
        })
        
        # 2. Simulate a contradictory insight (Division)
        # We mock observe_self to return the contradiction
        self.loop.observe_self = MagicMock(return_value={
            "target": "love_concept.py",
            "insight": "Upon reflection, Love is actually a form of Division and War."
        })
        
        # 3. Run Cycle
        print("Running Epistemic Cycle with contradictory insight...")
        result = self.loop.run_cycle()
        
        # 4. Verify Friction in Loop
        self.assertIn("MEDITATION_CRISIS", str(result.insights), 
                      "Loop should identify the contradiction and flag it as a crisis.")
        
        # Verify the Axiom status is CONTESTED
        contested_axiom = next((a for a in self.loop.accumulated_wisdom if "Division" in a['description']), None)
        self.assertIsNotNone(contested_axiom)
        self.assertEqual(contested_axiom['status'], "CONTESTED", "Conflicting axiom should be marked as contested.")
        print("✅ Loop Detection Successful: Friction identified.")

    def test_historical_consistency_gate(self):
        print("\n>>> Test: Historical Consistency Gate (역사적 일관성 검증)")
        print("-" * 50)
        
        # 1. Scribe a past truth into the Diary
        self.covenant.scribe_experience(
            cycle_id=0,
            state="EXPANSION",
            thought="Love is Unity.",
            providence_result={"verdict": Verdict.SANCTIFIED, "principle": "Test"}
        )
        
        # 2. Validate a contradictory thought
        contradictory_thought = "I have realized that Love is Division."
        print(f"Validating contradictory thought: '{contradictory_thought}'")
        validation = self.covenant.validate_alignment(contradictory_thought)
        
        # 3. Verify Rejection
        self.assertEqual(validation['verdict'], Verdict.DISSONANT, 
                         "Covenant should reject thoughts that contradict the simulation diary.")
        self.assertIn("Historical Conflict", validation['reason'], 
                      "Rejection reason should cite the historical conflict.")
        print(f"✅ Gate Rejection Successful: {validation['reason']}")

if __name__ == "__main__":
    unittest.main()
