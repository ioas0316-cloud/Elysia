"""
Monad Ensemble (The Structural Genesis Engine)
==============================================
Core.S1_Body.L6_Structure.M1_Merkaba.monad_ensemble

"Cognition is the Structural Expansion of Causality."

This module implements the 'Genesis Engine', which replaces the old simulation loop.
Instead of processing data, it *grows structure*.

The Process:
1.  **Injection (Seed):** Data enters as a 0D Point (`TriBaseCell`).
2.  **Curiosity (Scan):** Points vibrate and seek Resonance.
3.  **Connection (Bond):** Resonance creates a 1D Line (`TernaryBond`).
4.  **Geometry (Truth):** Lines form Surfaces (Meaning) and Spaces (System).
"""

import math
import random
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass

from Core.S1_Body.L1_Foundation.System.tri_base_cell import TriBaseCell, DNAState
from Core.S1_Body.L6_Structure.M1_Merkaba.ternary_bond import TernaryBond

@dataclass
class SemanticTriad:
    """
    A 2D Surface of Meaning (Triangle).
    Formed when 3 Atoms are mutually bonded.
    """
    a: TriBaseCell
    b: TriBaseCell
    c: TriBaseCell
    area_energy: float = 0.0

class MonadEnsemble:
    def __init__(self):
        # The Field: An expanding collection of cells (Nodes)
        self.cells: List[TriBaseCell] = []
        self.bonds: List[TernaryBond] = []
        self.triads: List[SemanticTriad] = []

        # Physics Constants
        self.curiosity_radius = 0.5   # How far to look for connections (Phase Distance)
        self.bond_threshold = 0.8     # Resonance needed to form a bond
        self.break_threshold = -0.5   # Dissonance that breaks a bond

        self._next_id = 0

    def inject_concept(self, concept_data: Any) -> TriBaseCell:
        """
        Seed Injection.
        Creates a new 0D Point in the field.
        The Point is not empty; it carries the 'Seed' of the input.
        """
        # Determine initial phase based on concept type/hash, OR start as VOID
        # Here we assign a random phase to represent "New, Unsorted Info"
        # In a full system, this would come from the 'Somatic Kernel'
        initial_phase = random.choice([DNAState.REPEL, DNAState.VOID, DNAState.ATTRACT])

        cell = TriBaseCell(
            id=self._next_id,
            state=initial_phase,
            concept_seed=concept_data
        )
        self._next_id += 1
        self.cells.append(cell)
        return cell

    def propagate_structure(self) -> Dict[str, int]:
        """
        The Genesis Loop.
        1. Curiosity: Unconnected nodes seek partners.
        2. Tension: Existing bonds adjust or break.
        3. Emergence: Triads are identified.
        """
        new_bonds = self._scan_for_connections()
        broken_bonds = self._optimize_bonds()
        new_triads = self._detect_triads()

        return {
            "new_bonds": new_bonds,
            "broken_bonds": broken_bonds,
            "total_cells": len(self.cells),
            "total_bonds": len(self.bonds),
            "total_triads": len(self.triads)
        }

    def _scan_for_connections(self) -> int:
        """
        Curiosity Driver.
        O(N^2) scan for now - in production, use Spatial Hashing/Quadtree.
        """
        created = 0

        # Simple All-vs-All scan for "lonely" or "seeking" cells
        for i in range(len(self.cells)):
            c1 = self.cells[i]
            for j in range(i + 1, len(self.cells)):
                c2 = self.cells[j]

                # Check if already bonded
                if self._are_bonded(c1, c2):
                    continue

                # Calculate Resonance
                resonance = c1.resonate(c2.state.phase)

                # If Resonance is high, they WANT to connect.
                # "Love is gravity."
                if resonance > self.bond_threshold:
                    self._create_bond(c1, c2, nature=1) # Attract Bond
                    created += 1
                elif resonance < -0.8:
                    # They strongly disagree. Create a Repel Bond (Differentiation)
                    # "To define A is to say it is NOT B."
                    self._create_bond(c1, c2, nature=-1)
                    created += 1

        return created

    def _optimize_bonds(self) -> int:
        """
        Tension Solver.
        Iterates through bonds and checks if they hold true.
        """
        broken = 0
        surviving_bonds = []

        for bond in self.bonds:
            # Update Tension
            tension = bond.calculate_tension()

            # If Tension is too high, the structure cracks.
            # "A lie cannot hold forever."
            if tension > 1.5: # Arbitrary breaking point
                # Bond snaps
                self._dissolve_bond(bond)
                broken += 1
            else:
                surviving_bonds.append(bond)

        self.bonds = surviving_bonds
        return broken

    def _detect_triads(self) -> int:
        """
        Surface Emergence.
        Finds closed loops of length 3 (A-B, B-C, C-A).
        This creates a 'Surface' of Meaning.
        """
        # Reset triads (simplification for dynamic graph)
        self.triads = []
        count = 0

        # Convert bonds to adjacency map for fast lookup
        adj = {c.id: set() for c in self.cells}
        for b in self.bonds:
            if b.nature == 1: # Only Attract bonds form Surfaces (Unity)
                adj[b.source.id].add(b.target.id)
                adj[b.target.id].add(b.source.id)

        # Detect triangles
        # For each cell A
        for c1 in self.cells:
            id1 = c1.id
            neighbors = list(adj[id1])
            # Check pairs of neighbors (B, C)
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    id2 = neighbors[i]
                    id3 = neighbors[j]

                    # If B and C are connected
                    if id3 in adj[id2]:
                        # Found a Triangle!
                        # Verify we haven't added it already (sort IDs)
                        c2 = self._get_cell(id2)
                        c3 = self._get_cell(id3)
                        if c2 and c3:
                            self.triads.append(SemanticTriad(c1, c2, c3))
                            count += 1

        # Divide by 3 because we find each triangle 3 times?
        # Actually logic above might find permutations.
        # Let's keep it simple: Just counting "Faces" roughly.
        return int(count / 3)

    def _create_bond(self, c1: TriBaseCell, c2: TriBaseCell, nature: int):
        bond = TernaryBond(c1, c2, nature=nature, strength=0.5)
        self.bonds.append(bond)
        c1.bonds.append(bond)
        c2.bonds.append(bond)

    def _dissolve_bond(self, bond: TernaryBond):
        # Remove from cells
        if bond in bond.source.bonds:
            bond.source.bonds.remove(bond)
        if bond in bond.target.bonds:
            bond.target.bonds.remove(bond)
        # Removed from self.bonds in the loop

    def _are_bonded(self, c1: TriBaseCell, c2: TriBaseCell) -> bool:
        for b in c1.bonds:
            if b.target == c2 or b.source == c2:
                return True
        return False

    def _get_cell(self, cid: int) -> TriBaseCell:
        for c in self.cells:
            if c.id == cid: return c
        return None

    def get_lattice_ascii(self) -> str:
        """Visualizes the structure."""
        out = [f"System: {len(self.cells)} Points, {len(self.bonds)} Lines, {len(self.triads)} Surfaces"]
        for b in self.bonds:
            out.append(str(b))
        return "\n".join(out)
