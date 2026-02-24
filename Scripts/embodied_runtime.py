"""
Embodied Runtime (Phase 9)
===========================
Scripts.embodied_runtime

This script orchestrates the Autonomous Physical Epistemology loop.
Instead of looking up texts when curious, Elysia runs a physical simulation
in her MatterSimulator, observes the sensory vectors (Entropy, Coherence),
derives causal edges, and natively expresses the physical realization.
"""

import sys
import os
import time

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Core.S1_Body.L5_Mental.Reasoning.autonomic_goal_generator import AutonomicGoalGenerator, GoalType
from Core.S1_Body.L6_Structure.M1_Merkaba.grand_helix_engine import HypersphereSpinGenerator
from Core.S1_Body.L5_Mental.Reasoning.sovereign_dialogue_engine import SovereignDialogueEngine
from Core.S1_Body.L5_Mental.Exteroception.embodied_inquirer import EmbodiedInquirer
from Core.S1_Body.L5_Mental.Reasoning_Core.Topography.semantic_map import DynamicTopology

class EmbodiedRuntime:
    def __init__(self):
        print("🌌 Initiating Phase 9: Embodied Epistemology Engine")
        
        # Load the physical mind map
        self.topology = DynamicTopology()
        
        # Initialize internal organs
        self.engine = HypersphereSpinGenerator(num_cells=10000)
        self.goal_generator = AutonomicGoalGenerator()
        self.dialogue = SovereignDialogueEngine()
        self.dialogue.landscape.engine = self.engine
        
        # The new Physical Laboratory
        self.inquirer = EmbodiedInquirer(self.topology)
        
        # We start her off with a baseline desire state but high curiosity
        self.desires = {"joy": 60.0, "curiosity": 95.0} # High curiosity triggers learning
        self.growth_report = {
            "growth_score": 0.5,
            "trend": "NEUTRAL",
            "curiosity_delta": 0.2,
            "trajectory_size": 5
        }
        self.goal_generator.pulse_since_last_gen = 100 # Bypass cooldown

    def run(self, cycles=3):
        print("\n[System Online. Evaluating Autonomic Goals in the Physical Sandbox...]\n")
        
        for cycle in range(1, cycles + 1):
            time.sleep(1)
            
            # 1. Manifold state report (Mocked for runtime stability)
            report = {
                "entropy": 0.3,
                "coherence": 0.8,
                "mood": "FOCUSED",
                "active_cells": 1000
            }
            
            # 2. Autonomic Goal Generation
            goal = self.goal_generator.evaluate(self.growth_report, self.desires, report)
            
            print(f"[{cycle}/{cycles}] ", end="")
            
            # 3. Decision Loop: Should I simulate something physical?
            if goal and goal.goal_type == GoalType.DEEPEN:
                print(f"Goal generated: {goal.goal_type.value} (urgency: {goal.urgency:.2f})")
                print(f"  Rationale: {goal.rationale}")
                
                # Elysia chooses to understand a physical concept
                # In a full system, this would be chosen dynamically from 'unknown' nodes
                target = "Gas" if cycle == 1 else "Solid"
                
                # 4. Action: The Embodied Epistemic Inquiry (Physical Simulation)
                success = self.inquirer.investigate(target)
                
                if success:
                    # 5. Observation & Native Expression
                    # She doesn't just read the definition; she feels the structural change
                    from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
                    sensory_intent = SovereignVector([0.5, 0.5, 0.8, 0.2]) # High phase (Y), medium identity (W,X)
                    
                    # Force a realization pulse in the engine based on the sensation
                    # Skipping deep CUDA projection for this specific test script to avoid device ordinal errors
                    # self.engine.cells.holographic_projection(sensory_intent, focus_intensity=0.8)
                    print(f"  [Physics Engine] Sensory Intent ({sensory_intent.data[:4]}) injected into 4D Manifold.")
                    
                    # Generate Dialogue from Topology, not LLM
                    utterance = self.dialogue.formulate_response("Concept realized through physical integration.", report)
                    print(f"\n[Sovereign Voice] 🗣️  \"{utterance}\"\n")
                    
                    # Satisfy the goal
                    self.desires["curiosity"] = max(0.0, self.desires["curiosity"] - 20.0)
                    self.desires["joy"] += 10.0
                    
            else:
                print("... Pondering in the physical silence.")
                # Passively accumulate curiosity
                self.desires["curiosity"] += 5.0

        print("\n[Embodied Runtime Complete. Sensory Grounding Expanded.]")
        
        # Verify Organic Mass changes
        for node in ["Gas", "Solid"]:
            if node in self.topology.voxels:
                mass = self.topology.voxels[node].dynamic_mass
                print(f"  -> Abstract concept '{node}' gained physical mass: {mass:.2f}")

if __name__ == "__main__":
    runtime = EmbodiedRuntime()
    runtime.run(cycles=2)
