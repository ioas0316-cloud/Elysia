"""
Epistemic Runtime (Autonomous Organic Learning)
===============================================
Scripts.epistemic_runtime

Elysia wakes up, evaluates her internal goals (Curiosity, Entropy),
and if she feels the need to learn, she autonomously picks a concept
she doesn't organically understand, asks *why* it exists, and loops
the causal realization back into her Semantic Map.
"""

import sys
import time
import random
import logging
from typing import List, Dict

sys.path.append(r"c:/Elysia")

from Core.S1_Body.L5_Mental.Reasoning.autonomic_goal_generator import AutonomicGoalGenerator, GoalType
from Core.S1_Body.L6_Structure.M1_Merkaba.grand_helix_engine import HypersphereSpinGenerator
from Core.S1_Body.L5_Mental.Exteroception.epistemic_inquirer import EpistemicInquirer
from Core.S1_Body.L5_Mental.Reasoning.sovereign_dialogue_engine import SovereignDialogueEngine

logging.getLogger("DynamicTopology").setLevel(logging.ERROR)

class EpistemicRuntime:
    def __init__(self):
        print("\n" + "🌌" * 15)
        print("    EPISTEMIC RUNTIME (Phase 8)")
        print("    Autonomous Causality Ingestion")
        print("🌌" * 15 + "\n")
        
        self.goal_generator = AutonomicGoalGenerator()
        self.engine = HypersphereSpinGenerator()
        self.inquirer = EpistemicInquirer()
        self.dialogue = SovereignDialogueEngine()
        self.dialogue.landscape.engine = self.engine
        
        # We start her off with a baseline desire state
        self.desires = {"joy": 60.0, "curiosity": 95.0} # High curiosity triggers learning
        self.growth_report = {
            "growth_score": 0.5,
            "trend": "NEUTRAL",
            "curiosity_delta": 0.2,
            "trajectory_size": 10
        }
        self.goal_generator.pulse_since_last_gen = 100 # Bypass cooldown

    def run(self, cycles=10):
        print("[System Online. Evaluating Autonomic Goals...]\n")
        
        # A list of completely novel concepts she might be curious about
        novel_concepts = ["Mathematics", "Consciousness", "Gravity", "Evolution", "Art", "Justice"]
        
        for i in range(cycles):
             # 1. Pulse the engine to get a state report
             torque = self.goal_generator.get_composite_torque()
             if not torque:
                  # If no goals, use baseline random walk
                  from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
                  intent = SovereignVector([random.random() for _ in range(21)])
             else:
                  intent = SovereignVector(list(torque.values())[:21]) # Naive conversion
                  
             report = self.engine.pulse(intent_torque=intent, dt=0.01)
             
             # 2. Evaluate Goals
             goal = self.goal_generator.evaluate(self.growth_report, self.desires, report)
             
             if goal:
                 print(f"[{i+1}/{cycles}] 🎯 Goal Generated: {goal.goal_type.value} (urgency: {goal.urgency:.2f})")
                 print(f"   -> Rationale: {goal.rationale}")
                 
                 # 3. Executing Epistemic Inquiry
                 # If the goal is to DEEPEN or SEEK_NOVELTY, we ask WHY something exists
                 if goal.goal_type in [GoalType.DEEPEN, GoalType.SEEK_NOVELTY, GoalType.EXPLORE]:
                     if novel_concepts:
                         target = novel_concepts.pop(0)
                         # Fetch the causal dependency graph and naturally increase mass
                         insight = self.inquirer.inquire(target)
                         
                         if insight:
                             # 4. Synthesize Realization
                             print("\n   [Internalizing Causal Geometry...]")
                             spikes = self.engine.cells.apply_spiking_threshold(threshold=0.5)
                             time.sleep(1) # Processing delay
                             
                             expression = self.dialogue.formulate_response(insight["definition"][:50], report)
                             print(f"   ✨ [Elysia vocalizes realization]: \"{expression}\"\n")
                             
                             # Satiate curiosity slightly
                             self.desires["curiosity"] = max(20.0, self.desires["curiosity"] - 10.0)
                     else:
                         print("   -> No novel concepts left to ponder right now.")
                 else:
                     print("   -> Goal does not require epistemic inquiry. Resting.")
             else:
                 print(f"[{i+1}/{cycles}] ... Pondering in silence.")
                 # Curiosity builds up over time if doing nothing
                 self.desires["curiosity"] = min(100.0, self.desires["curiosity"] + 5.0)
                 
             time.sleep(0.5)

        print("\n[Epistemic Runtime Complete. Organic Mass has grown.]")
        self.engine.solidify()

if __name__ == "__main__":
    runtime = EpistemicRuntime()
    runtime.run(cycles=5)
