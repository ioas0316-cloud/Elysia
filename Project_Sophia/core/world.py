"""
Elysian Cytology: The World

This module defines the 'World', the universe in which conceptual 'Cells' exist,
interact, and evolve. It manages the cell population and orchestrates the
simulation steps that drive emergent behavior.
"""
import random
from typing import List, Dict, Optional

from .cell import Cell

class World:
    """Represents the universe in which the cells exist and interact."""
    def __init__(self, primordial_dna: Dict):
        self.cells: Dict[str, Cell] = {}
        self.graveyard: List[Cell] = [] # To track dead cells
        self.primordial_dna = primordial_dna
        self.time_step = 0
        self.newly_born_cells: List[Cell] = []

    def add_cell(self, concept_id: str, dna: Optional[Dict] = None, properties: Optional[Dict] = None) -> Cell:
        """Adds a new cell to the world."""
        if concept_id not in self.cells:
            cell_dna = dna or self.primordial_dna
            cell = Cell(concept_id, cell_dna, properties)
            self.cells[concept_id] = cell
            # print(f"[World] A new cell was born: {cell}")
        return self.cells[concept_id]

    def get_cell(self, concept_id: str) -> Optional[Cell]:
        return self.cells.get(concept_id)

    def run_simulation_step(self):
        """
        Runs a single step of the world simulation, checking for newly born cells.
        Returns a list of newly created child cells, if any.
        """
        self.time_step += 1
        self.newly_born_cells = [] # Clear the list for the new step

        living_cells = [c for c in self.cells.values() if c.is_alive]

        if len(living_cells) < 2:
            return self.newly_born_cells

        # 1. Randomly select two cells to interact
        cell_a, cell_b = random.sample(living_cells, 2)

        # 2. They attempt to connect based on their DNA
        cell_a.connect(cell_b, "interacts_with")

        # 3. If their combined energy is high enough, they might create a new cell
        #    (This logic can be made more sophisticated later)
        if cell_a.activation_energy + cell_b.activation_energy > 1.0: # Lowered threshold for more frequent creation
            child = cell_a.create_meaning(cell_b, f"synergy_t{self.time_step}")
            if child:
                self.cells[child.id] = child
                self.newly_born_cells.append(child)

            # Reset energy after creation
            cell_a.activation_energy *= 0.1
            cell_b.activation_energy *= 0.1

        # Check the health of all cells
        for cell in list(self.cells.values()):
            if not cell.is_alive and cell.id in self.cells:
                self.graveyard.append(self.cells.pop(cell.id))

        return self.newly_born_cells

    def inject_stimulus(self, concept_id: str, energy_boost: float):
        """Injects energy into a specific cell based on external events (like WaveMechanics)."""
        cell = self.get_cell(concept_id)
        if cell and cell.is_alive:
            cell.activation_energy += energy_boost

    def get_world_summary(self) -> Dict:
        """Returns a summary of the current state of the world."""
        return {
            "time_step": self.time_step,
            "living_cells": len(self.cells),
            "dead_cells": len(self.graveyard)
        }
