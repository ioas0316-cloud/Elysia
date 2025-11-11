
import random
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from .cell import Cell
from .chronicle import Chronicle
from ..wave_mechanics import WaveMechanics


class World:
    """Represents the universe where cells exist, interact, and evolve, optimized with NumPy."""

    def __init__(self, primordial_dna: Dict, wave_mechanics: WaveMechanics,
                 chronicle: Optional[Chronicle] = None, logger: Optional[logging.Logger] = None,
                 branch_id: str = "main", parent_event_id: Optional[str] = None):
        # --- Core Attributes ---
        self.primordial_dna = primordial_dna
        self.wave_mechanics = wave_mechanics
        self.chronicle = chronicle
        self.time_step = 0
        self.logger = logger or logging.getLogger(__name__)

        # --- Celestial Cycle ---
        self.day_length = 20
        self.time_of_day = 'day'

        # --- Chronos Engine Attributes ---
        self.branch_id = branch_id
        self.parent_event_id = parent_event_id

        # --- Quantum State Management ---
        self.quantum_states: Dict[str, Dict[str, float]] = {}

        # --- Materialized Cell Management ---
        self.materialized_cells: Dict[str, Cell] = {}
        self.graveyard: List[Cell] = []

        # --- NumPy Data Structures for Optimization ---
        self.cell_ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}
        self.energy = np.array([], dtype=np.float32)
        self.is_alive_mask = np.array([], dtype=bool)
        self.connection_counts = np.array([], dtype=np.int32)
        self.element_types = np.array([], dtype='<U10')

        # --- SciPy Sparse Matrix for Connections ---
        self.adjacency_matrix = lil_matrix((0, 0), dtype=np.float32)

    def _resize_matrices(self, new_size: int):
        current_size = len(self.cell_ids)
        if new_size <= current_size:
            return
        self.energy = np.pad(self.energy, (0, new_size - current_size), 'constant')
        self.is_alive_mask = np.pad(self.is_alive_mask, (0, new_size - current_size), 'constant', constant_values=False)
        self.connection_counts = np.pad(self.connection_counts, (0, new_size - current_size), 'constant')
        self.element_types = np.pad(self.element_types, (0, new_size - current_size), 'constant')
        new_adj = lil_matrix((new_size, new_size), dtype=np.float32)
        if self.adjacency_matrix.shape[0] > 0:
            new_adj[:current_size, :current_size] = self.adjacency_matrix
        self.adjacency_matrix = new_adj

    def add_cell(self, concept_id: str, dna: Optional[Dict] = None, properties: Optional[Dict] = None, initial_energy: float = 0.0, _record_event: bool = True):
        if concept_id in self.quantum_states:
            return
        if self.chronicle and _record_event:
            details = {'concept_id': concept_id, 'initial_energy': initial_energy, 'properties': properties or {}}
            scopes = [concept_id]
            event = self.chronicle.record_event('cell_added', details, scopes, self.branch_id, self.parent_event_id)
            self.parent_event_id = event['id']
        self.quantum_states[concept_id] = {'existence_probability': 1.0, 'energy_potential': initial_energy, 'age': 0}
        idx = len(self.cell_ids)
        if idx >= self.adjacency_matrix.shape[0]:
            self._resize_matrices(max(idx + 1, 100))
        self.cell_ids.append(concept_id)
        self.id_to_idx[concept_id] = idx
        self.energy[idx] = initial_energy
        self.is_alive_mask[idx] = True
        self.connection_counts[idx] = 0
        temp_cell = self.materialize_cell(concept_id, force_materialize=True)
        if temp_cell:
            self.element_types[idx] = temp_cell.element_type
        else:
            self.element_types[idx] = 'unknown'

    def materialize_cell(self, concept_id: str, force_materialize: bool = False) -> Optional[Cell]:
        if not force_materialize and concept_id in self.materialized_cells:
            return self.materialized_cells[concept_id]
        if concept_id in self.quantum_states:
            idx = self.id_to_idx.get(concept_id)
            if idx is None:
                self.logger.error(f"Quantum state for '{concept_id}' exists, but it has no index in the world.")
                return None
            state = self.quantum_states[concept_id]
            cell = Cell(concept_id, self.primordial_dna, initial_energy=self.energy[idx])
            cell.age = state.get('age', 0)
            cell.is_alive = self.is_alive_mask[idx]
            self.element_types[idx] = cell.element_type
            self.materialized_cells[concept_id] = cell
            return cell
        return None

    def _sync_states_to_objects(self):
        for i, cell_id in enumerate(self.cell_ids):
            if cell_id in self.materialized_cells:
                cell = self.materialized_cells[cell_id]
                cell.energy = self.energy[i]
                cell.is_alive = self.is_alive_mask[i]

    def run_simulation_step(self) -> List[Cell]:
        if self.chronicle:
            event = self.chronicle.record_event('simulation_step_run', {}, [], self.branch_id, self.parent_event_id)
            self.parent_event_id = event['id']
        self.time_step += 1
        for state in self.quantum_states.values():
            state['age'] += 1
        num_cells = len(self.cell_ids)
        if num_cells == 0:
            return []
        adj_matrix_csr = self.adjacency_matrix.tocsr()
        energy_boost = np.zeros_like(self.energy, dtype=np.float32)
        if self.wave_mechanics and self.wave_mechanics.kg_manager:
            love_node = self.wave_mechanics.kg_manager.get_node('love')
            love_energy_multiplier = love_node.get('activation_energy', 1.0) if love_node else 1.0
            for i in range(num_cells):
                if self.is_alive_mask[i]:
                    cell_id = self.cell_ids[i]
                    try:
                        resonance = self.wave_mechanics.get_resonance_between(cell_id, 'love')
                        energy_boost[i] = resonance * love_energy_multiplier * 0.5
                    except Exception:
                        pass
        energy_out_matrix = adj_matrix_csr.multiply(self.energy[:, np.newaxis]) * 0.1
        total_energy_out = np.array(energy_out_matrix.sum(axis=1)).flatten()
        total_energy_in = np.array(energy_out_matrix.sum(axis=0)).flatten()
        energy_deltas = total_energy_in - total_energy_out + energy_boost
        cycle_position = self.time_step % self.day_length
        self.time_of_day = 'day' if cycle_position < self.day_length / 2 else 'night'
        if self.time_of_day == 'day':
            if self.wave_mechanics and self.wave_mechanics.kg_manager:
                sun_node = self.wave_mechanics.kg_manager.get_node('sun')
                if sun_node:
                    sunlight_energy = sun_node.get('activation_energy', 2.0)
                    life_mask = (self.element_types == 'life') & self.is_alive_mask
                    energy_deltas[life_mask] += sunlight_energy
        life_mask = (self.element_types == 'life') & self.is_alive_mask
        if np.any(life_mask):
            water_mask = self.element_types == 'water'
            earth_mask = self.element_types == 'earth'
            nurturing_mask = water_mask | earth_mask
            if np.any(nurturing_mask):
                life_to_nurture_connections = adj_matrix_csr[life_mask][:, nurturing_mask]
                nurturing_counts = np.array(life_to_nurture_connections.sum(axis=1)).flatten()
                energy_deltas[life_mask] += nurturing_counts * 0.5
        if self.time_of_day == 'night':
            energy_deltas[self.is_alive_mask] -= 0.2
        animal_mask = (self.element_types == 'animal') & self.is_alive_mask
        eco_life_mask = (self.element_types == 'life') & self.is_alive_mask
        if np.any(animal_mask) and np.any(eco_life_mask):
            animal_indices = np.where(animal_mask)[0]
            for i in animal_indices:
                connections_to_life = adj_matrix_csr[i, :][:, eco_life_mask]
                prey_indices = connections_to_life.indices
                if prey_indices.size > 0:
                    target_prey_local_idx = random.choice(prey_indices)
                    target_prey_global_idx = np.where(eco_life_mask)[0][target_prey_local_idx]
                    energy_transfer = 5.0
                    energy_deltas[i] += energy_transfer
                    energy_deltas[target_prey_global_idx] -= energy_transfer
                if self.energy[i] + energy_deltas[i] > 50.0:
                    energy_deltas[i] -= self.energy[i] / 2
        self.energy += energy_deltas
        newly_born_cells = []
        self._sync_states_to_objects()
        living_cells = [self.materialized_cells[self.cell_ids[i]] for i in range(num_cells) if self.is_alive_mask[i] and self.energy[i] > 1.0 and self.cell_ids[i] in self.materialized_cells]
        newly_born_molecules = self._run_chemical_reactions(living_cells)
        newly_born_cells.extend(newly_born_molecules)
        for cell in newly_born_cells:
            if cell.id not in self.id_to_idx:
                 self.add_cell(cell.id, cell.nucleus['dna'], cell.organelles, cell.energy)
        apoptosis_mask = (self.energy < 0.1) & self.is_alive_mask
        self.is_alive_mask[apoptosis_mask] = False
        self.energy[apoptosis_mask] = 0.0
        for cell in newly_born_cells:
            idx = self.id_to_idx.get(cell.id)
            if idx is not None:
                self.energy[idx] += 5.0
        maintenance_mask = (self.energy > 50.0) & (self.connection_counts > 0) & self.is_alive_mask
        self.energy[maintenance_mask] += 1.0
        nurture_mask = (self.connection_counts == 0) & (self.energy < 50.0) & self.is_alive_mask
        self.energy[nurture_mask] += 0.5
        self._sync_states_to_objects()
        dead_cell_ids = [self.cell_ids[i] for i in range(num_cells) if not self.is_alive_mask[i]]
        for cell_id in dead_cell_ids:
            if cell_id in self.materialized_cells:
                self.graveyard.append(self.materialized_cells[cell_id])
                del self.materialized_cells[cell_id]
            if cell_id in self.quantum_states:
                self.quantum_states[cell_id]['existence_probability'] = 0.0
        return newly_born_cells

    def add_connection(self, source_id: str, target_id: str, strength: float = 0.5, _record_event: bool = True):
        if source_id in self.id_to_idx and target_id in self.id_to_idx:
            if self.chronicle and _record_event:
                details = {'source': source_id, 'target': target_id, 'strength': strength}
                scopes = [source_id, target_id]
                event = self.chronicle.record_event('connection_added', details, scopes, self.branch_id, self.parent_event_id)
                self.parent_event_id = event['id']
            source_idx = self.id_to_idx[source_id]
            target_idx = self.id_to_idx[target_id]
            self.adjacency_matrix[source_idx, target_idx] = strength
            self.connection_counts[source_idx] += 1

    def inject_stimulus(self, concept_id: str, energy_boost: float, _record_event: bool = True):
        if concept_id in self.id_to_idx:
            if self.chronicle and _record_event:
                details = {'concept_id': concept_id, 'energy_boost': energy_boost}
                scopes = [concept_id]
                event = self.chronicle.record_event('stimulus_injected', details, scopes, self.branch_id, self.parent_event_id)
                self.parent_event_id = event['id']
            idx = self.id_to_idx[concept_id]
            if self.is_alive_mask[idx]:
                self.energy[idx] += energy_boost

    def _run_chemical_reactions(self, living_cells: List[Cell]) -> List[Cell]:
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
                    newly_born_molecules.append(new_molecule)
                    self.logger.info(f"[Ionic Bond] '{existence_cell.id}' ({existence_cell.element_type}) and '{emotion_cell.id}' ({emotion_cell.element_type}) created '{new_molecule.id}'.")
        return newly_born_molecules

    def print_world_summary(self):
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
