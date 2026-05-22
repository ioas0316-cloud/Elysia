"""
Hyper Merkaba Engine (The N-Dimensional Generator)
==================================================
Core.Monad.monad_ensemble

"Structure is the Crystalline Residue of Friction."

This is the Refactored Genesis Engine.
It uses `HyperMonad` (N-D Tensors) instead of `TriBaseCell`.
It implements "Tensor Fusion" and "Dimensional Mitosis".
"""

import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from Core.Monad.hyper_monad import HyperMonad, CausalResidue, AXIS_MASS, AXIS_PHASE, AXIS_TIME

@dataclass
class TensorBond:
    source: HyperMonad
    target: HyperMonad
    resonance: float
    age: int = 0

class MonadEnsemble:
    def __init__(self):
        self.monads: List[HyperMonad] = []
        self.bonds: List[TensorBond] = []
        self._next_id = 0

        # Physics Constants
        self.BOND_THRESHOLD = 0.2  # Lowered for testing
        self.FRICTION_COEFF = 0.1

    def inject_seed(self, tensor_input: List[float] = None) -> HyperMonad:
        """
        Injects a new thought-seed into the vacuum.
        """
        m = HyperMonad(self._next_id, tensor=tensor_input)
        self._next_id += 1
        self.monads.append(m)
        return m

    def process_cycle(self) -> Dict[str, Any]:
        """
        One tick of the Cosmic Clock.
        1. Scan for Resonance.
        2. Form Bonds.
        3. Fusion (Birth of new Monads).
        4. Evolution (Mitosis).
        """
        new_bonds = self._scan_and_bond()
        births = self._process_fusion()
        self._evolve_monads()

        return {
            "monad_count": len(self.monads),
            "bond_count": len(self.bonds),
            "new_births": births,
            "max_dimensions": max([m.dimensions for m in self.monads]) if self.monads else 0
        }

    def _scan_and_bond(self) -> int:
        count = 0
        # Naive O(N^2) for prototype
        for i in range(len(self.monads)):
            m1 = self.monads[i]
            for j in range(i + 1, len(self.monads)):
                m2 = self.monads[j]

                # Calculate Resonance
                res = m1.resonate(m2)

                # Bond on High Resonance OR High Friction (Opposition)
                if abs(res) > self.BOND_THRESHOLD:
                    # Check if bond exists
                    if not self._bond_exists(m1, m2):
                        self.bonds.append(TensorBond(m1, m2, res))
                        count += 1
        return count

    def _bond_exists(self, m1, m2) -> bool:
        for b in self.bonds:
            if (b.source == m1 and b.target == m2) or (b.source == m2 and b.target == m1):
                return True
        return False

    def _process_fusion(self) -> int:
        """
        If resonance is extremely high (or low/friction), parents create a Child.
        """
        births = 0
        for bond in self.bonds:
            # Condition for Birth: High Energy Interaction
            if abs(bond.resonance) > 0.2: # Match Bond Threshold
                child = self._create_child(bond.source, bond.target, bond.resonance)
                self.monads.append(child)
                bond.age += 10 # Reset/Refractory period
                births += 1

            bond.age += 1
        return births

    def _create_child(self, p1: HyperMonad, p2: HyperMonad, resonance: float) -> HyperMonad:
        """
        Tensor Fusion Logic.
        Child = Vector Mean(P1, P2) + Causal Shift (W).
        """
        max_dim = max(p1.dimensions, p2.dimensions)
        v1 = p1._pad_vector(p1.tensor, max_dim)
        v2 = p2._pad_vector(p2.tensor, max_dim)

        # 1. Vector Synthesis
        child_tensor = []
        for a, b in zip(v1, v2):
            child_tensor.append((a + b) / 2.0)

        # 2. Causal Shift (Time moves forward)
        # The child is deeper in the lineage than the parents.
        avg_time = (v1[AXIS_TIME] + v2[AXIS_TIME]) / 2.0
        child_tensor[AXIS_TIME] = avg_time + 1.0 # Generation Step

        # 3. Create Child
        child = HyperMonad(self._next_id, tensor=child_tensor)
        self._next_id += 1

        # 4. Imprint Lineage (The Soul's History)
        child.lineage = CausalResidue(
            parent_ids=[p1.id, p2.id],
            friction_heat=abs(resonance), # Simplified heat metric
            birth_time=avg_time + 1.0
        )

        return child

    def _evolve_monads(self):
        """
        Apply global time/friction to all monads.
        """
        for m in self.monads:
            # Simulate some environmental friction
            m.evolve(friction=0.05)

    def get_census(self) -> str:
        out = []
        for m in self.monads:
            dims = m.dimensions
            out.append(f"Monad {m.id}: {dims}D Tensor")
        return "\n".join(out)
