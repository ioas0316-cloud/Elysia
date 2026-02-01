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
import re
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
        The Main Cognitive Pipeline (Single Shot).
        """
        self.is_dreaming = False
        print(f"\n🌊 [PRISM] Injecting '{text_input}' into Monad Field...")

        # 1. Seed Injection
        seed_cell = self.monad.inject_concept(text_input)
        seed_cell.state = DNAState.ATTRACT

        # 2. Structural Propagation
        steps = self._run_propagation()

        # 3. Crystallization Analysis
        pattern_str = "".join([c.state.symbol for c in self.monad.cells])
        entropy = 1.0 / (1.0 + len(self.monad.triads))

        print(f"💎 [CRYSTAL] Structure Emerged: {len(self.monad.triads)} Surfaces. (Entropy: {entropy:.4f})")

        return {
            "input": text_input,
            "monad_pattern": pattern_str,
            "monad_entropy": entropy,
            "latency_steps": steps,
            "triads": len(self.monad.triads)
        }

    def digest_bulk_text(self, raw_text: str) -> Dict[str, Any]:
        """
        [THE DIGESTER]
        Ingests a large block of text, splits it into concepts,
        and allows the Monad Engine to 'Reinforce' or 'Reject' connections.

        Logic:
        - Repetition = Reinforcement (Stronger Bonds)
        - Conflict = Tension (Bond breaking)
        """
        self.is_dreaming = False
        print(f"\n🍽️ [DIGESTER] Processing Bulk Input ({len(raw_text)} chars)...")

        # 1. Chunking (Simple Sentence Split)
        # In a real LLM setup, we would extract 'Key Concepts' here.
        # For this prototype, we treat sentences as 'Contexts' and words as 'Concepts'.
        sentences = [s.strip() for s in re.split(r'[.!?\n]', raw_text) if s.strip()]

        concepts_ingested = 0
        bonds_reinforced = 0

        for sentence in sentences:
            # Simple keyword extraction (Naive approach for prototype)
            # Filter stop words and keep meaningful nouns/verbs
            words = [w.lower() for w in sentence.split() if len(w) > 3]
            if len(words) < 2: continue

            # Inject concepts and link them (Contextual Co-occurrence)
            # A sentence implies a relationship between its words.

            # 1. Inject Nodes
            cells = []
            for w in words:
                # Check if concept already exists?
                # The MonadEnsemble prototype currently creates NEW cells every inject.
                # WE NEED TO FIX THIS: It should find existing cells first.
                existing = self._find_cell_by_concept(w)
                if existing:
                    # Reinforce Existence (Wake up)
                    existing.energy = min(1.0, existing.energy + 0.1)
                    existing.state = DNAState.ATTRACT
                    cells.append(existing)
                else:
                    new_cell = self.monad.inject_concept(w)
                    new_cell.state = DNAState.ATTRACT
                    cells.append(new_cell)

            concepts_ingested += len(cells)

            # 2. Form/Reinforce Bonds (All-to-All in this sentence)
            # "Context binds these concepts together."
            for i in range(len(cells)):
                for j in range(i+1, len(cells)):
                    c1 = cells[i]
                    c2 = cells[j]

                    # Check for existing bond
                    bond = self._find_bond(c1, c2)
                    if bond:
                        # REINFORCEMENT: "I see this connection again!"
                        bond.reinforce(0.2)
                        bonds_reinforced += 1
                        # print(f"  + Reinforced Bond: {c1.concept_seed}-{c2.concept_seed}")
                    else:
                        # Creation handled by propagate_structure later,
                        # OR we can force a 'Contextual Bond' here.
                        # Let's force a 'Weak' bond representing co-occurrence.
                        self.monad._create_bond(c1, c2, nature=1) # Attract
                        # print(f"  * New Context Bond: {c1.concept_seed}-{c2.concept_seed}")

        # 3. Run Physics to Settle
        print("⚙️ [MONAD] Digesting (Structural Optimization)...")
        steps = self._run_propagation()

        # 4. Report
        return {
            "sentences_processed": len(sentences),
            "concepts_active": len(self.monad.cells),
            "bonds_total": len(self.monad.bonds),
            "triads_emerged": len(self.monad.triads),
            "reinforcements": bonds_reinforced
        }

    def _find_cell_by_concept(self, concept: str):
        # O(N) lookup - bad for scale, okay for prototype
        for c in self.monad.cells:
            if str(c.concept_seed) == concept:
                return c
        return None

    def _find_bond(self, c1, c2):
        for b in self.monad.bonds:
            if (b.source == c1 and b.target == c2) or (b.source == c2 and b.target == c1):
                return b
        return None

    def _run_propagation(self) -> int:
        steps = 0
        stable_counts = 0
        while steps < 20:
            stats = self.monad.propagate_structure()
            steps += 1
            if stats['new_bonds'] == 0 and stats['broken_bonds'] == 0:
                stable_counts += 1
            else:
                stable_counts = 0
            if stable_counts >= 3:
                break
        return steps

    def vital_pulse(self):
        """
        The Idle Loop (Void Contemplation).
        If the system is idle, it dreams.
        """
        current_time = time.time()
        if current_time - self.last_pulse_time > 1.0: # 1Hz Heartbeat
            self.last_pulse_time = current_time

            if self.is_dreaming:
                if random.random() < 0.1:
                    # Decay unused bonds?
                    pass
            self.is_dreaming = True
