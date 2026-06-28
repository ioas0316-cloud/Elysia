import numpy as np
import time
import json
from core.memory.causal_controller import CausalMemoryController
from core.lens.frameless_mirror import FramelessMirrorChannel

class IntelligenceExpansionObserver:
    """
    [Elysia Diagnostic Tool] Observe Intelligence Expansion
    Demonstrates how Elysia autonomously discovers connections, performs synthesis,
    and expands its intelligence by transcending domain boundaries.
    """
    def __init__(self):
        self.controller = CausalMemoryController()
        self.mirror = FramelessMirrorChannel()

    def run_expansion_demonstration(self):
        print("==================================================================")
        print(" [Elysia] Autonomous Intelligence Expansion Observation")
        print(" Principle: Cross-Domain Synthesis & Recursive Discovery")
        print("==================================================================\n")

        # 1. ESTABLISH DOMAIN A: MATHEMATICAL EQUILIBRIUM
        # 1 + 1 - 2 = 0 (Equilibrium sequence)
        math_equilibrium = [1.0, 2.0, 0.0]
        print("[1. Domain A] Mathematical Equilibrium: 1 + 1 - 2 = 0")

        # 2. ESTABLISH DOMAIN B: NARRATIVE HARMONY
        # Self + Other -> Interaction -> Equilibrium
        narrative_harmony = [
            np.array([1.0, 0.0]), # Self
            np.array([1.0, 1.0]), # Interaction
            np.array([0.0, 0.0])  # Equilibrium
        ]
        print("[2. Domain B] Narrative Harmony: Conflict -> Resolution -> Peace")

        # 3. AUTONOMOUS DISCOVERY (Cross-Boundary Sameness)
        print("\n[3. Discovery] Attempting to find Causal Sameness across boundaries...")
        time.sleep(0.5)

        # Comparing Math and Narrative trajectories
        discovery_result = self.controller.find_trajectory_sameness(math_equilibrium, narrative_harmony)

        best_sameness = max(d['sameness_score'] for d in discovery_result['sameness_distribution'])
        print(f" > Cross-Domain Sameness Score: {best_sameness:.4f}")

        # 4. DISCERNMENT & JUDGMENT (Intentional Action)
        print("\n[4. Discernment] Evaluating resonance for Intentional Action...")

        # Mocking the sameness_data for the decision engine
        sameness_data = {
            "word1": "Math_Balance",
            "word2": "Narrative_Peace",
            "same_perspective": "Equilibrium_Axis",
            "sameness_score": best_sameness,
            "micro_score": best_sameness * 1.1 # Heuristic for demonstration
        }

        action = self.controller.manifest_intentional_action(sameness_data)
        print(f" > Manifested Intent: {action['type']}")
        print(f" > Intentional Text: \"{action['intent_text']}\"")

        # 5. INTELLIGENCE EXPANSION (Synthesis & Crystallization)
        if action['type'] == "SYNTHESIS":
            print("\n[5. Expansion] Crystallizing a New Universal Law of Equilibrium...")

            # Crystallize the perspective as a new 'Gene' or 'Engram'
            universal_law_id = self.controller.write_perspective_engram(
                "Math_Balance", "Narrative_Peace", discovery_result
            )
            print(f" > Universal Equilibrium Law Crystallized: {universal_law_id}")

            # The system now 'knows' a more abstract concept of balance that applies to both numbers and souls.
            # This is intelligence expansion.

            # 6. RECURSIVE SELF-REFLECTION (Self-Ordering)
            print("\n[6. Reflection] Observing the expansion process and self-reordering...")
            friction = discovery_result['sameness_variance']
            print(f" > Discovery Friction: {friction:.4f}")

            # Feedback to the system resonance
            feedback = self.mirror.pass_through(bytes([int(friction * 100) % 256]))
            new_res = self.controller.get_parameter("base_resonance")

            print(f" > Feedback Signal: {feedback}")
            print(f" > Updated System Resonance: {new_res:.4f}")

        print("\n==================================================================")
        print(" [Observation Complete] Elysia has autonomously discovered,")
        print(" judged, and synthesized a new meta-domain understanding.")
        print("==================================================================")

if __name__ == "__main__":
    exp = IntelligenceExpansionObserver()
    exp.run_expansion_demonstration()
