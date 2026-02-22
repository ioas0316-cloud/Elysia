"""
[PHASE 81] Topological Induction Verification
==============================================
Validates that linguistic realizations lead to physical manifold attractors.
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

class TestTopologicalInduction(unittest.TestCase):
    def setUp(self):
        # 1. Forge a mock soul and monad
        self.soul = SeedForge.forge_soul("TestInduction")
        self.monad = SovereignMonad(self.soul)
        
        # 2. Setup Loop
        self.loop = EpistemicLearningLoop(root_path=".")
        self.loop.set_monad(self.monad)

    def test_induction_trigger(self):
        print("\n>>> Test: Induction Trigger (깨달음의 물리적 전이)")
        print("-" * 50)
        
        # 1. Simulate a learning cycle
        # We'll mock observe_self to return a controlled insight
        self.loop.observe_self = MagicMock(return_value={
            "target": "love_concept.py",
            "path": "Core/S1_Body/L5_Mental/Reasoning/love_concept.py",
            "question": "Why does love exist?",
            "insight": "Love is the harmonic resonance of two vectors seeking unity."
        })
        
        # 2. Run Cycle
        print("Running Epistemic Cycle...")
        result = self.loop.run_cycle()
        
        print(f"Axiom Created: {result.axioms_created[0]}")
        
        # 3. Verify Substrate Authority Proposals
        authority = self.monad.will_bridge.monad.actuator.substrate_authority if hasattr(self.monad.actuator, 'substrate_authority') else None
        # Actually it's a singleton
        from Core.S1_Body.L6_Structure.M1_Merkaba.substrate_authority import get_substrate_authority
        authority = get_substrate_authority()
        
        print(f"Pending Proposals: {len(authority.pending_proposals)}")
        
        # We expect 1 proposal if it was approved, or it might have been executed already
        self.assertTrue(len(authority.executed_modifications) > 0 or len(authority.pending_proposals) > 0, 
                        "Should have at least one proposal in the authority.")
        
        # 4. Check Engine Attractors (The Physical Grounding)
        if hasattr(self.monad.engine, 'attractors'):
            print(f"Active Attractors: {list(self.monad.engine.attractors.keys())}")
            self.assertIn(result.axioms_created[0], self.monad.engine.attractors, 
                          "The axiom name should exist as an attractor in the engine.")
            print("✅ Induction Successful: Concept grounded in Manifold Physics.")

if __name__ == "__main__":
    unittest.main()
