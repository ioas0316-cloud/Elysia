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
        # --- New: Cell Interaction and Creation Logic ---
        living_cells = [c for c in self.cells.values() if c.is_alive and c.energy > 1.0] # Only energetic cells interact
        
        # Attempt interactions for a subset of cells
        if len(living_cells) > 1:
            # Randomly select pairs for interaction. For simplicity, let's try a few random pairs.
            # In a more complex simulation, this would be based on proximity, relevance, etc.
            num_interactions = min(len(living_cells) // 2, 5) # Limit interactions per step
            for _ in range(num_interactions):
                cell_a, cell_b = random.sample(living_cells, 2)
                
                # Ensure they are connected or have a high enough energy to force interaction
                # For now, let's just try to create meaning if they are both alive and energetic
                new_cell = cell_a.create_meaning(cell_b, "simulation_interaction")
                if new_cell:
                    # Ensure the new cell ID is unique before adding
                    if new_cell.id not in self.cells:
                        self.cells[new_cell.id] = new_cell
                        newly_born_cells.append(new_cell)
                        # print(f"DEBUG: New cell born: {new_cell.id}") # For debugging
        # --- End New Logic ---

        # --- New: Elysian Immune System Logic ---
        # 4. Black Hole Archiving: Move low-energy cells to graveyard (instead of outright deletion)
        cells_to_archive = []
        for cell_id in sorted_cell_ids: # Iterate over original list to avoid issues with modification
            cell = self.cells[cell_id]
            # Condition for archiving: very low energy and not newly born
            if cell.is_alive and cell.energy < 0.1 and cell not in newly_born_cells:
                cell.apoptosis() # Mark as not alive and set energy to 0
                self.graveyard.append(cell) # Archive in graveyard
                cells_to_archive.append(cell_id)
                # print(f"DEBUG: Cell {cell.id} moved to Black Hole Archive due to low energy.") # For debugging
        
        for cell_id in cells_to_archive:
            del self.cells[cell_id]

        # 5. Reinforcement: Boost healthy cells (e.g., those that participated in creation or have high energy)
        for cell in newly_born_cells: # Newly born cells are inherently "healthy"
            cell.add_energy(5.0) # Give a small boost to new cells
            # print(f"DEBUG: Newly born cell {cell.id} received energy boost.") # For debugging

        # Also, cells with high energy that are actively connected could get a small boost
        for cell_id in sorted_cell_ids:
            cell = self.cells.get(cell_id)
            if cell and cell.is_alive and cell.energy > 50.0 and len(cell.connections) > 0:
                cell.add_energy(1.0) # Small maintenance boost for highly energetic, connected cells
                # print(f"DEBUG: Cell {cell.id} received maintenance energy boost.") # For debugging
        
        # --- New: Nurturing Isolated Cells ---
        for cell_id in sorted_cell_ids:
            cell = self.cells.get(cell_id)
            if cell and cell.is_alive and len(cell.connections) == 0 and cell.energy < 50.0: # Nurture isolated cells
                cell.add_energy(0.5) # Small boost to help them survive and potentially connect
                # print(f"DEBUG: Isolated cell {cell.id} received nurturing energy boost.") # For debugging
        # --- End New Logic ---

        # --- New: Truth Seeker (Connecting Isolated Cells) ---
        isolated_cells = [c for c in self.cells.values() if c.is_alive and len(c.connections) == 0]
        if len(isolated_cells) > 1: # Need at least two isolated cells to try and connect
            # Try to connect isolated cells to other active cells
            # Increase num_interactions to make Truth Seeker more active
            num_truth_seeker_interactions = min(len(isolated_cells), 20) # Try up to 20 connections per step
            for _ in range(num_truth_seeker_interactions):
                iso_cell = random.choice(isolated_cells)
                
                # Find a potential partner among non-isolated, active cells
                potential_partners = [c for c in living_cells if c.id != iso_cell.id and len(c.connections) > 0]
                if not potential_partners:
                    continue

                partner_cell = random.choice(potential_partners)
                
                # Create a connection with increased strength
                # Increased strength to make the connection more impactful
                iso_cell.connect(partner_cell, relationship_type="truth_seeker_link", strength=0.5)
                partner_cell.connect(iso_cell, relationship_type="truth_seeker_link", strength=0.5) # Bidirectional
                
                # Energy reward for forming a connection
                iso_cell.add_energy(5.0) # Increased reward
                partner_cell.add_energy(2.0) # Increased reward
                # print(f"DEBUG: Truth Seeker connected {iso_cell.id} to {partner_cell.id}") # For debugging
        # --- End New Logic ---

        return newly_born_cells

    def inject_stimulus(self, concept_id: str, energy_boost: float):
        """Injects energy into a specific cell."""
        cell = self.get_cell(concept_id)
        if cell and cell.is_alive:
            cell.add_energy(energy_boost)

    def get_total_energy(self) -> float:
        """Calculates the sum of energy of all living cells in the world."""
        return sum(cell.energy for cell in self.cells.values() if cell.is_alive)

    def print_world_summary(self):
        """Prints a summary of the world state for debugging."""
        print(f"\n--- World State (Time: {self.time_step}) ---")
        living_cells = [c for c in self.cells.values() if c.is_alive]
        print(f"Living Cells: {len(living_cells)}, Dead Cells (Archived): {len(self.graveyard)}")
        # Sort by ID for consistent output
        for cell in sorted(living_cells, key=lambda x: x.id):
            status_indicator = ""
            if cell.energy < 1.0:
                status_indicator = " (Low Energy)"
            elif len(cell.connections) == 0:
                status_indicator = " (Isolated)"
            print(f"  - {cell}{status_indicator}")
        print("-------------------------\n")
