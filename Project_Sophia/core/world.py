import random
from typing import List, Dict, Optional, Tuple

from .cell import Cell

class World:
    """Represents the universe where cells exist, interact, and evolve."""

    def __init__(self, primordial_dna: Dict):
        self.cells: Dict[str, Cell] = {}
        self.graveyard: List[Cell] = []
        self.primordial_dna = primordial_dna
        self.time_step = 0

    def add_cell(self, concept_id: str, dna: Optional[Dict] = None, properties: Optional[Dict] = None, initial_energy: float = 0.0) -> Cell:
        """Adds a new cell to the world or returns the existing one."""
        if concept_id not in self.cells:
            cell_dna = dna or self.primordial_dna
            cell = Cell(concept_id, cell_dna, properties, initial_energy=initial_energy)
            self.cells[concept_id] = cell
        return self.cells[concept_id]

    def get_cell(self, concept_id: str) -> Optional[Cell]:
        """Retrieves a cell by its ID."""
        return self.cells.get(concept_id)

    def run_simulation_step(self) -> List[Cell]:
        """
        Runs a single, more realistic simulation step where all cells propagate energy.
        This is a deterministic process based on the current state.
        """
        self.time_step += 1
        energy_deltas: Dict[str, float] = {cell_id: 0.0 for cell_id in self.cells}
        newly_born_cells: List[Cell] = []

        # Use a sorted list for deterministic iteration
        sorted_cell_ids = sorted(self.cells.keys())

        # 1. Calculate energy transfers for all cells
        for cell_id in sorted_cell_ids:
            cell = self.cells[cell_id]
            if not cell.is_alive or cell.energy <= 0.01:
                continue

            # Propagate energy to connected cells
            for conn in cell.connections:
                target_id = conn.get('target_id')
                if target_id in self.cells:
                    strength = conn.get('strength', 0.5)
                    transfer_amount = cell.energy * strength * 0.1  # Transfer 10% of energy scaled by strength

                    if transfer_amount > 0:
                        energy_deltas[cell_id] -= transfer_amount
                        energy_deltas[target_id] += transfer_amount

        # 2. Apply all energy deltas simultaneously
        for cell_id, delta in energy_deltas.items():
            if cell_id in self.cells:
                self.cells[cell_id].add_energy(delta)

        # 3. Check for cell death or new life (optional, can be expanded)
        # ... (Apoptosis or creation logic can be added here) ...

        return newly_born_cells

    def inject_stimulus(self, concept_id: str, energy_boost: float):
        """Injects energy into a specific cell."""
        cell = self.get_cell(concept_id)
        if cell and cell.is_alive:
            cell.add_energy(energy_boost)

    def print_world_summary(self):
        """Prints a summary of the world state for debugging."""
        print(f"\n--- World State (Time: {self.time_step}) ---")
        living_cells = [c for c in self.cells.values() if c.is_alive]
        print(f"Living Cells: {len(living_cells)}, Dead Cells: {len(self.graveyard)}")
        # Sort by ID for consistent output
        for cell in sorted(living_cells, key=lambda x: x.id):
            print(f"  - {cell}")
        print("-------------------------\n")
