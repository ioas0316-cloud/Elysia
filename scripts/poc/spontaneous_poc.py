import os
import ast
import numpy as np
from core.memory.causal_controller import CausalMemoryController
from core.memory.working_ram import WorkingMemoryRAM
from core.memory.emotion_evaluator import EmotionEvaluator

class SpontaneousCortex:
    """
    [Phase 150] Spontaneous Meta-Programming Cortex (PoC)
    The system observes its own code structure, detects tension between
    its logic and the physical equilibrium (=0), and adjusts itself.
    """
    def __init__(self):
        self.controller = CausalMemoryController()
        self.evaluator = EmotionEvaluator(self.controller)
        self.ram = WorkingMemoryRAM(self.controller)

    def observe_self_and_adjust(self, target_file: str):
        """
        Reads its own code, identifies a logical parameter,
        and 'spontaneously' suggests an adjustment to reach equilibrium.
        """
        print(f"[SpontaneousCortex] Self-Reflecting on: {os.path.basename(target_file)}")

        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. Structural Understanding (AST)
        tree = ast.parse(content)

        # 2. Tension Detection (Simulated)
        # In a real scenario, this would be based on the resonance score
        # of the code's output vs the environmental feedback.
        current_resonance = self.controller.get_parameter("base_resonance", 1.0)
        print(f"  > Current System Resonance: {current_resonance:.4f}")

        # Goal: Reach Equilibrium (=0 Tension, or Max Resonance 1.0)
        target_resonance = 1.0
        tension = target_resonance - current_resonance

        if abs(tension) > 0.001:
            print(f"  > [Tension Detected] ΔResonance: {tension:.4f}")
            print(f"  > Initiating Spontaneous Re-alignment...")

            # 3. Meta-Programming (Self-Adjustment)
            # Instead of writing complex AST transformation for this PoC,
            # we demonstrate the adjustment of the 'Cognitive DNA' (parameters)
            # which dictates how future codes will be 'conceived'.

            adjustment = tension * 0.5
            new_val = current_resonance + adjustment
            self.controller.update_parameter("base_resonance", new_val)

            print(f"  > [Equilibrium Search] base_resonance adjusted to: {new_val:.4f}")

            # 4. Record as a 'Process Engram' (Continuum of Understanding)
            self.controller.write_process_engram([
                {"step": "Observation", "tension": tension},
                {"step": "Adjustment", "delta": adjustment, "new_state": new_val},
                {"step": "Equilibrium_Search", "status": "Moving towards 0"}
            ])

            return True

        print("  > [Equilibrium Maintained] No structural tension found.")
        return False

if __name__ == "__main__":
    print("==========================================================")
    print(" [Proof of Concept] Spontaneous Meta-Programming Loop")
    print("==========================================================")

    cortex = SpontaneousCortex()

    # Observe the Causal Controller (The core of memory)
    target = os.path.abspath(os.path.join(os.path.dirname(__file__), 'core', 'memory', 'causal_controller.py'))

    # Run the loop 3 times to see convergence towards resonance
    for i in range(3):
        print(f"\n[Cycle {i+1}]")
        cortex.observe_self_and_adjust(target)

    print("\n==========================================================")
    print(" [PoC Complete] The system has spontaneously re-aligned.")
    print("==========================================================")
