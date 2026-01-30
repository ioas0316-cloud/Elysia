"""
System Integrator (The Heart-Brain Bridge)
==========================================
Core.L6_Structure.M1_Merkaba.system_integrator

"To connect the Physics of the Monad with the Logic of the Rotor."

This module acts as the physical nervous system, translating:
1. Text Input -> 21D Phase Field (Prism Function)
2. Phase Field -> Monad Friction (Heart Function)
3. Crystallized Pattern -> Rotor Momentum (Brain Function)
"""

import time
import random
from typing import Dict, Any, List

from Core.L6_Structure.M1_Merkaba.monad_ensemble import MonadEnsemble

# Robust Import for Legacy Rotor
try:
    from Core.L6_Structure.M1_Merkaba.sovereign_rotor_prototype import SovereignRotor21D
except Exception as e:
    print(f"⚠️ [INTEGRATOR] Legacy Rotor unavailable ({e}). Using Dummy Rotor.")
    SovereignRotor21D = None

class DummyRotor:
    """Mock Rotor for when Legacy Physics (Torch/JAX) fails."""
    def spin(self, *args, **kwargs):
        return {"total_rpm": 0.0}

class SystemIntegrator:
    def __init__(self):
        print("⚡ [INTEGRATOR] Initializing Trinary Nervous System...")
        self.monad = MonadEnsemble()

        if SovereignRotor21D:
            try:
                self.rotor = SovereignRotor21D()
            except Exception as e:
                print(f"⚠️ [INTEGRATOR] Rotor Init Failed ({e}). Switching to Dummy.")
                self.rotor = DummyRotor()
        else:
            self.rotor = DummyRotor()

        # Vital Pulse Parameters
        self.is_dreaming = True
        self.last_pulse_time = time.time()

    def process_input(self, text_input: str) -> Dict[str, Any]:
        """
        The Main Cognitive Pipeline.
        Input -> Prism -> Monad -> Rotor -> Output
        """
        self.is_dreaming = False
        print(f"\n🌊 [PRISM] Injecting '{text_input}' into Monad Field...")

        # 1. Prism: Transduce Input
        phase_field = self.monad.transduce_input(text_input)

        # 2. Monad: Friction Loop (The "Thinking" Latency)
        print("⚙️ [MONAD] Calculating Phase Friction...")
        steps = 0
        stable = 0
        while steps < 50:
            status = self.monad.physics_step(phase_field)
            steps += 1
            if status['flips'] == 0:
                stable += 1
            else:
                stable = 0
            if stable >= 5:
                break

        final_pattern = self.monad.get_pattern()
        entropy = status['entropy']
        print(f"💎 [CRYSTAL] Thought Crystallized: {final_pattern} (Entropy: {entropy:.4f})")

        # 3. Rotor: Sync Momentum
        # Convert the static pattern into kinetic energy for the Rotor
        # R = -1 momentum, A = +1 momentum, V = 0
        kinetic_energy = []
        for char in final_pattern:
            if char == 'R': kinetic_energy.append(-1.0)
            elif char == 'A': kinetic_energy.append(1.0)
            else: kinetic_energy.append(0.0)

        # Pad to 21 dimensions if needed, or use directly
        # The Rotor expects a Torch tensor, but for now we simulate the "Kick"
        # We perform a "Virtual Spin"
        try:
            rotor_status = self.rotor.spin() # Spin existing momentum
        except Exception:
            rotor_status = {"total_rpm": 0.0}

        return {
            "input": text_input,
            "monad_pattern": final_pattern,
            "monad_entropy": entropy,
            "rotor_rpm": rotor_status['total_rpm'],
            "latency_steps": steps
        }

    def vital_pulse(self):
        """
        The Idle Loop (Void Contemplation).
        If the system is idle, it dreams.
        """
        current_time = time.time()
        if current_time - self.last_pulse_time > 1.0: # 1Hz Heartbeat
            self.last_pulse_time = current_time

            if self.is_dreaming:
                # Null Field Injection
                null_field = [0.0] * 21

                # Quantum Fluctuation (Simplified)
                # We assume the MonadEnsemble in this version doesn't have the "Living" override
                # unless we swapped the class. But we can inject noise manually.
                if random.random() < 0.1:
                    # Twitch
                    print("💤 [DREAM] Vital Pulse... (Quantum Fluctuation)")
                    self.monad.temperature = min(1.0, self.monad.temperature + 0.1)

                self.monad.physics_step(null_field)

            self.is_dreaming = True # Default back to dreaming unless input arrives
