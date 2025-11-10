
import random
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from .cell import Cell
from ..wave_mechanics import WaveMechanics


class World:
    """Represents the universe where cells exist, interact, and evolve, optimized with NumPy."""

    def __init__(self, primordial_dna: Dict, wave_mechanics: WaveMechanics, logger: Optional[logging.Logger] = None):
        # --- Core Attributes ---
        self.primordial_dna = primordial_dna
        self.wave_mechanics = wave_mechanics
        self.time_step = 0
        self.logger = logger or logging.getLogger(__name__)

        # --- Quantum State Management ---
        # Stores the potential state of all cells, not their full object representation.
        self.quantum_states: Dict[str, Dict[str, float]] = {}

        # --- Materialized Cell Management ---
        # Stores the actual, fully instantiated Cell objects that are currently being observed.
        self.materialized_cells: Dict[str, Cell] = {}
        self.graveyard: List[Cell] = []

        # --- NumPy Data Structures for Optimization ---
        self.cell_ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}

        # --- Dynamic-size arrays for cell properties ---
        self.energy = np.array([], dtype=np.float32)
        self.is_alive_mask = np.array([], dtype=bool)
        self.connection_counts = np.array([], dtype=np.int32)

        # --- SciPy Sparse Matrix for Connections ---
        # Using lil_matrix for efficient row-wise additions
        self.adjacency_matrix = lil_matrix((0, 0), dtype=np.float32)

    def _resize_matrices(self, new_size: int):
        """Resizes all NumPy arrays and the sparse matrix to accommodate more cells."""
        current_size = len(self.cell_ids)
        if new_size <= current_size:
            return

        # Resize NumPy arrays
        self.energy = np.pad(self.energy, (0, new_size - current_size), 'constant')
        self.is_alive_mask = np.pad(self.is_alive_mask, (0, new_size - current_size), 'constant', constant_values=False)
        self.connection_counts = np.pad(self.connection_counts, (0, new_size - current_size), 'constant')

        # Resize SciPy sparse matrix
        new_adj = lil_matrix((new_size, new_size), dtype=np.float32)
        if self.adjacency_matrix.shape[0] > 0:
            new_adj[:current_size, :current_size] = self.adjacency_matrix
        self.adjacency_matrix = new_adj

    def add_cell(self, concept_id: str, dna: Optional[Dict] = None, properties: Optional[Dict] = None, initial_energy: float = 0.0):
        """Adds a new cell's quantum state to the world. Does not create a full Cell object."""
        if concept_id in self.quantum_states:
            return

        # --- Create and store the quantum state ---
        self.quantum_states[concept_id] = {
            'existence_probability': 1.0,
            'energy_potential': initial_energy,
            'age': 0
        }

        # --- Update Optimized Data Structures ---
        idx = len(self.cell_ids)
        if idx >= self.adjacency_matrix.shape[0]:
            self._resize_matrices(max(idx + 1, idx + 100)) # Grow by a chunk to reduce re-allocations

        self.cell_ids.append(concept_id)
        self.id_to_idx[concept_id] = idx

        self.energy[idx] = initial_energy
        self.is_alive_mask[idx] = True  # A new quantum state is considered "alive"
        self.connection_counts[idx] = 0 # No connections initially

        # Connections are not handled at the quantum state level in this design.
        # They are established when cells are materialized and interact.

    def materialize_cell(self, concept_id: str) -> Optional[Cell]:
        """
        Retrieves a materialized cell. If not materialized, it 'collapses' the quantum state
        into a full Cell object.
        """
        if concept_id in self.materialized_cells:
            return self.materialized_cells[concept_id]

        if concept_id in self.quantum_states:
            idx = self.id_to_idx.get(concept_id)
            # This should always be found if a quantum state exists, but we check for safety.
            if idx is None:
                self.logger.error(f"Quantum state for '{concept_id}' exists, but it has no index in the world.")
                return None

            # Collapse the state using the MOST RECENT data from the NumPy arrays
            state = self.quantum_states[concept_id]
            current_energy = self.energy[idx]
            is_currently_alive = self.is_alive_mask[idx]

            cell = Cell(
                concept_id,
                self.primordial_dna,
                initial_energy=current_energy
            )
            cell.age = state.get('age', 0)
            cell.is_alive = is_currently_alive

            self.materialized_cells[concept_id] = cell
            return cell

        return None

    def _sync_states_to_objects(self):
        """Syncs the state from NumPy arrays back to the Cell objects."""
        for i, cell_id in enumerate(self.cell_ids):
            if cell_id in self.materialized_cells:
                cell = self.materialized_cells[cell_id]
                cell.energy = self.energy[i]
                cell.is_alive = self.is_alive_mask[i]

    def run_simulation_step(self) -> List[Cell]:
        """
        Runs a single simulation step using optimized NumPy and SciPy operations.
        """
        self.time_step += 1

        # Increment age for all quantum states
        for state in self.quantum_states.values():
            state['age'] += 1

        num_cells = len(self.cell_ids)
        if num_cells == 0:
            return []

        # Convert to CSR for efficient arithmetic
        adj_matrix_csr = self.adjacency_matrix.tocsr()

        # --- Vectorized Law of Love ---
        # For simplicity, this remains a loop for now as it involves external calls.
        # Can be optimized further if wave_mechanics supports batch operations.
        energy_boost = np.zeros_like(self.energy, dtype=np.float32)
        if self.wave_mechanics and self.wave_mechanics.kg_manager:
            # Get the 'love' node's value mass (activation_energy) to use as a multiplier
            love_node = self.wave_mechanics.kg_manager.get_node('love')
            love_energy_multiplier = love_node.get('activation_energy', 1.0) if love_node else 1.0

            # Iterate only up to the current number of valid cells
            for i in range(num_cells):
                if self.is_alive_mask[i]:
                    cell_id = self.cell_ids[i]
                    try:
                        resonance = self.wave_mechanics.get_resonance_between(cell_id, 'love')
                        # The Arc Reactor: energy boost is proportional to resonance and love's own energy
                        energy_boost[i] = resonance * love_energy_multiplier * 0.5
                    except Exception:
                        pass # Ignore if resonance fails

        # --- Vectorized Energy Propagation ---
        # 1. Calculate energy to be transferred out from each cell
        transfer_rate = 0.1
        # Element-wise multiplication of the adjacency matrix with the energy vector
        energy_out_matrix = adj_matrix_csr.multiply(self.energy[:, np.newaxis]) * transfer_rate

        # 2. Sum up the energy transferred out for each cell
        total_energy_out = np.array(energy_out_matrix.sum(axis=1)).flatten()

        # 3. Sum up the energy transferred in for each cell (transpose of the out matrix)
        total_energy_in = np.array(energy_out_matrix.sum(axis=0)).flatten()

        # 4. Calculate the net change in energy
        energy_deltas = total_energy_in - total_energy_out + energy_boost

        # 5. Apply the energy changes
        self.energy += energy_deltas

        # --- Logic that is harder to vectorize remains similar ---
        newly_born_cells: List[Cell] = []

        # Sync state before complex, non-vectorized logic
        self._sync_states_to_objects()

        # The rest of the logic (chemical reactions, generic interactions, etc.)
        # requires the object representation. It is less of a bottleneck.
        living_cells = [self.materialized_cells[self.cell_ids[i]] for i in range(num_cells) if self.is_alive_mask[i] and self.energy[i] > 1.0 and self.cell_ids[i] in self.materialized_cells]

        # --- Chemical Reactions & Generic Interactions ---
        newly_born_molecules = self._run_chemical_reactions(living_cells)
        newly_born_cells.extend(newly_born_molecules)
        # (Generic interaction logic remains the same)

        # Sync back any new cells created
        for cell in newly_born_cells:
            if cell.id not in self.id_to_idx:
                 self.add_cell(cell.id, cell.nucleus['dna'], cell.organelles, cell.energy)

        # --- Vectorized Cell State Updates (Apoptosis, Reinforcement, etc.) ---
        
        # Apoptosis: Mark cells with very low energy as dead
        apoptosis_mask = (self.energy < 0.1) & self.is_alive_mask
        self.is_alive_mask[apoptosis_mask] = False
        self.energy[apoptosis_mask] = 0.0

        # Reinforcement for newly born cells
        for cell in newly_born_cells:
            idx = self.id_to_idx.get(cell.id)
            if idx is not None:
                self.energy[idx] += 5.0

        # Maintenance for highly energetic, connected cells
        maintenance_mask = (self.energy > 50.0) & (self.connection_counts > 0) & self.is_alive_mask
        self.energy[maintenance_mask] += 1.0

        # Nurturing isolated cells
        nurture_mask = (self.connection_counts == 0) & (self.energy < 50.0) & self.is_alive_mask
        self.energy[nurture_mask] += 0.5

        # --- Final Sync ---
        self._sync_states_to_objects()

        # Update the main cells dictionary by removing dead cells
        # This is inefficient, but necessary for compatibility. A full refactor would change this.
        dead_cell_ids = [self.cell_ids[i] for i in range(num_cells) if not self.is_alive_mask[i]]
        for cell_id in dead_cell_ids:
            if cell_id in self.materialized_cells:
                self.graveyard.append(self.materialized_cells[cell_id])
                del self.materialized_cells[cell_id]
            if cell_id in self.quantum_states:
                self.quantum_states[cell_id]['existence_probability'] = 0.0

        # Note: This implementation does not yet rebuild the numpy arrays after cell death.
        # For a long-running simulation, this would lead to memory bloat.
        # A full implementation would require re-indexing, which is a major change.
        # This version focuses on optimizing the hot path (energy calculation).

        return newly_born_cells

    def add_connection(self, source_id: str, target_id: str, strength: float = 0.5):
        """Adds a directed connection to the adjacency matrix."""
        if source_id in self.id_to_idx and target_id in self.id_to_idx:
            source_idx = self.id_to_idx[source_id]
            target_idx = self.id_to_idx[target_id]
            self.adjacency_matrix[source_idx, target_idx] = strength
            self.connection_counts[source_idx] += 1

    # --- Other methods remain largely the same, but need to be compatible ---
    def inject_stimulus(self, concept_id: str, energy_boost: float):
        """Injects energy into a specific cell."""
        if concept_id in self.id_to_idx:
            idx = self.id_to_idx[concept_id]
            if self.is_alive_mask[idx]:
                self.energy[idx] += energy_boost

    def _run_chemical_reactions(self, living_cells: List[Cell]) -> List[Cell]:
        """Runs chemical reactions based on elemental types. (Logic unchanged)"""
        newly_born_molecules = []

        elements_map: Dict[str, List[Cell]] = {}
        for cell in living_cells:
            elements_map.setdefault(cell.element_type, []).append(cell)

        if 'existence' in elements_map and 'emotion' in elements_map:
            for existence_cell in elements_map['existence']:
                if not elements_map['emotion']: continue
                emotion_cell = random.choice(elements_map['emotion'])
                new_molecule = existence_cell.create_meaning(emotion_cell, "ionic_bond")
                if new_molecule and new_molecule.id not in self.materialized_cells and new_molecule.id not in self.quantum_states:
                    # self.materialized_cells[new_molecule.id] = new_molecule # Will be handled by add_cell
                    newly_born_molecules.append(new_molecule)
                    self.logger.info(f"[Ionic Bond] '{existence_cell.id}' ({existence_cell.element_type}) and '{emotion_cell.id}' ({emotion_cell.element_type}) created '{new_molecule.id}'.")

        return newly_born_molecules

    def print_world_summary(self):
        """Prints a summary of the world state for debugging."""
        # Ensure object states are up-to-date before printing
        self._sync_states_to_objects()

        print(f"\n--- World State (Time: {self.time_step}) ---")
        living_cells = [c for c in self.materialized_cells.values() if c.is_alive]
        print(f"Living Cells: {len(living_cells)}, Dead Cells (Archived): {len(self.graveyard)}")
        for cell in sorted(living_cells, key=lambda x: x.id):
            status_indicator = ""
            if cell.energy < 1.0:
                status_indicator = " (Low Energy)"
            elif len(cell.connections) == 0:
                status_indicator = " (Isolated)"
            print(f"  - {cell}{status_indicator}")
        print("-------------------------\n")
