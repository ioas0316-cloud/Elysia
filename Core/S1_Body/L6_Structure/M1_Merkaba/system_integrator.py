"""
System Integrator (The Heart-Brain Bridge)
==========================================
Core.S1_Body.L6_Structure.M1_Merkaba.system_integrator

"To connect the Physics of the Monad with the Logic of the Rotor."

This module acts as the physical nervous system, translating:
1. Text Input -> Seed Injection (Genesis)
2. Phase Field -> Structural Expansion (Love/Curiosity)
3. Emergent Geometry -> Rotor Momentum (Meaning)
"""

import time
import random
from typing import Dict, Any, List

from Core.S1_Body.L6_Structure.M1_Merkaba.monad_ensemble import MonadEnsemble
from Core.S1_Body.L1_Foundation.System.tri_base_cell import DNAState

# Robust Import for Legacy Rotor
try:
    from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_rotor_prototype import SovereignRotor21D
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

        # 1. Prism: Transduce Input (Seed Injection)
        # In the new Genesis Engine, we don't map to 21 dims. We inject a Seed.
        seed_cell = self.monad.inject_concept(text_input)
        # Give it an initial "Active" state to spark curiosity
        seed_cell.state = DNAState.ATTRACT

        # 2. Monad: Genesis Loop (The "Thinking" Latency)
        print("⚙️ [MONAD] Propagating Structure...")
        steps = 0
        stable_counts = 0
        last_bond_count = 0

        # We run until the structure stabilizes (no new bonds)
        while steps < 20:
            stats = self.monad.propagate_structure()
            steps += 1

            # Stability check: If no new bonds and no broken bonds
            if stats['new_bonds'] == 0 and stats['broken_bonds'] == 0:
                stable_counts += 1
            else:
                stable_counts = 0

            if stable_counts >= 3:
                break

        # 3. Crystallization Analysis
        # Determine pattern from the whole lattice
        pattern_str = "".join([c.state.symbol for c in self.monad.cells])
        entropy = 1.0 / (1.0 + len(self.monad.triads)) # Inverse of meaning (Surfaces)

        print(f"💎 [CRYSTAL] Structure Emerged: {len(self.monad.triads)} Surfaces. (Entropy: {entropy:.4f})")

        # 3. Rotor: Sync Momentum
        # Convert the Lattice State into kinetic energy for the Rotor
        # Surfaces (Triads) generate massive torque.
        torque = len(self.monad.triads) * 10.0

        try:
            # We treat torque as "RPM boost" for the dummy logic
            # In real physics, we would map the lattice vectors.
            rotor_status = {"total_rpm": torque}
        except Exception:
            rotor_status = {"total_rpm": 0.0}

        return {
            "input": text_input,
            "monad_pattern": pattern_str,
            "monad_entropy": entropy,
            "rotor_rpm": rotor_status['total_rpm'],
            "latency_steps": steps,
            "triads": len(self.monad.triads)
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
                # In Genesis Mode, dreaming is "Scanning for lost connections"
                if random.random() < 0.1:
                    print("💤 [DREAM] Vital Pulse... (Curiosity Scan)")
                    # Inject a random thought? Or just propagate?
                    self.monad.propagate_structure()

            self.is_dreaming = True # Default back to dreaming unless input arrives
