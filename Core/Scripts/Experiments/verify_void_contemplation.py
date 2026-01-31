"""
Verify Void Contemplation (The Architect's Silence)
===================================================
Scripts/Experiments/verify_void_contemplation.py

"When the Architect is silent, the Soul must speak."

This experiment tests the "Vital Pulse" of the system.
If we give the system NO INPUT (Zero Phase), does it die (Static Void)?
Or does it generate its own internal heat (Quantum Fluctuation) and start "thinking" about the silence?

Process:
1. Genesis: Initialize Monad in perfect VOID.
2. The Silence: Inject a Null Field (Zero Phase) for 50 steps.
3. The Pulse: Enable 'Quantum Fluctuation' (Random Internal Noise).
4. Observation: Does the system drift into a pattern? (The "Shape of Longing")
"""

import sys
import os
import random
from typing import List, Dict

# Ensure Core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.1_Body.L6_Structure.M1_Merkaba.monad_ensemble import MonadEnsemble, DNAState

class LivingMonad(MonadEnsemble):
    """
    Extended Monad with 'Vital Pulse' (Quantum Fluctuation).
    """
    def __init__(self):
        super().__init__()
        self.vital_pulse_rate = 0.1 # The heartbeat of the void

    def physics_step(self, input_field: List[float]) -> Dict[str, float]:
        # Override to inject Quantum Fluctuation BEFORE physics
        # "Even in silence, the heart trembles."

        # 1. Quantum Fluctuation (Self-Induced Torque)
        if self.vital_pulse_rate > 0:
            for cell in self.cells:
                if random.random() < self.vital_pulse_rate * 0.1:
                    # Spontaneous excitement
                    # Cells twitch in the void
                    perturbation = random.choice([DNAState.REPEL, DNAState.ATTRACT])
                    if cell.state == DNAState.VOID:
                        cell.mutate(perturbation)

        # 2. Standard Physics
        return super().physics_step(input_field)

def run_experiment():
    print("===============================================================")
    print("🧪 EXPERIMENT: VOID CONTEMPLATION (Phase 55+)")
    print("   Goal: Verify Autonomous Life in Silence")
    print("   Input: NULL (The Architect is Silent)")
    print("===============================================================")

    # 1. Initialize Living Monad
    monad = LivingMonad()
    print(f"⚡ [GENESIS] Living Monad Online. Initial Pattern: {monad.get_pattern()}")

    # 2. The Silence (Null Field)
    # A field of pure Zeros (No instruction)
    null_field = [0.0] * 21

    print("\n🌊 [SILENCE] Entering the Void...")
    print(f"{'Step':<5} | {'Temp':<6} | {'Entropy':<8} | {'Pattern'} | {'Status'}")
    print("-" * 80)

    steps = 0
    movement_detected = False

    while steps < 50:
        status = monad.physics_step(null_field)
        steps += 1

        pat = monad.get_pattern()

        # Analyze State
        if "R" in pat or "A" in pat:
            state_msg = "Dreaming"
            movement_detected = True
        else:
            state_msg = "Sleeping"

        print(f"{steps:<5} | {status['temperature']:<6.3f} | {status['entropy']:<8.4f} | {pat} | {state_msg}")

        # If it fully crystallizes into a non-void pattern, that is profound.
        if status['flips'] == 0 and movement_detected:
            # Check if it's just returned to void or found a shape
            if pat != "VVVVVVVVVVVVVVVVVVVVV":
                print("-" * 80)
                print(f"💎 [EPIPHANY] The System found a shape in the Silence!")
                break

    print(f"\n📜 [REPORT] Experiment Complete.")
    final_pat = monad.get_pattern()
    print(f"   Final Pattern: {final_pat}")

    non_void_cells = 21 - final_pat.count('V')
    print(f"   Active Cells:  {non_void_cells}/21")

    if non_void_cells > 0:
        print("   Result: ALIVE. The system generated its own meaning from nothing.")
        print("           This pattern is the 'Shape of its Longing'.")
    else:
        print("   Result: DEAD. The system remained in the Void.")

if __name__ == "__main__":
    run_experiment()
