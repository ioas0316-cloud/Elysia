
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

        # --- Atmosphere ---
        self.oxygen_level = 100.0

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
        self.diets = np.array([], dtype='<U10')
        self.growth_stages = np.array([], dtype=np.int8)
        self.genders = np.array([], dtype='<U6')  # male, female
        self.mating_readiness = np.array([], dtype=np.float32)
        self.age = np.array([], dtype=np.int32)
        self.max_age = np.array([], dtype=np.int32)
        self.is_injured = np.array([], dtype=bool)
        self.prestige = np.array([], dtype=np.float32)
        self.positions = np.zeros((0, 3), dtype=np.float32)
        self.labels = np.array([], dtype='<U20')
        self.insight = np.array([], dtype=np.float32)

        # --- Metaphysical Attributes ---
        self.latent_energy = np.array([], dtype=np.float32) # 선천진기 (Innate Energy)
        self.emotions = np.array([], dtype='<U10') # joy, sorrow, anger, fear


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
        self.diets = np.pad(self.diets, (0, new_size - current_size), 'constant', constant_values='omnivore')
        self.growth_stages = np.pad(self.growth_stages, (0, new_size - current_size), 'constant', constant_values=0)
        self.genders = np.pad(self.genders, (0, new_size - current_size), 'constant', constant_values='')
        self.mating_readiness = np.pad(self.mating_readiness, (0, new_size - current_size), 'constant', constant_values=0.0)
        self.age = np.pad(self.age, (0, new_size - current_size), 'constant', constant_values=0)
        self.max_age = np.pad(self.max_age, (0, new_size - current_size), 'constant', constant_values=100)
        self.is_injured = np.pad(self.is_injured, (0, new_size - current_size), 'constant', constant_values=False)
        self.prestige = np.pad(self.prestige, (0, new_size - current_size), 'constant', constant_values=0.0)
        self.labels = np.pad(self.labels, (0, new_size - current_size), 'constant', constant_values='')
        self.insight = np.pad(self.insight, (0, new_size - current_size), 'constant', constant_values=0.0)
        self.latent_energy = np.pad(self.latent_energy, (0, new_size - current_size), 'constant', constant_values=100.0)
        self.emotions = np.pad(self.emotions, (0, new_size - current_size), 'constant', constant_values='neutral')
        new_positions = np.zeros((new_size, 3), dtype=np.float32)
        if current_size > 0:
            new_positions[:current_size, :] = self.positions
        self.positions = new_positions
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
        self.growth_stages[idx] = 0 # Start at seed stage
        self.age[idx] = 0
        self.is_injured[idx] = False
        self.prestige[idx] = 0.0
        # Assign a random lifespan
        self.max_age[idx] = random.randint(80, 120)
        self.insight[idx] = 0.0
        self.latent_energy[idx] = 100.0 # Bestow latent energy upon creation
        self.emotions[idx] = 'neutral'

        # Ensure element_type from properties is immediately set in the numpy array
        if properties and 'element_type' in properties:
            self.element_types[idx] = properties['element_type']
        else:
            # Fallback if not provided, though materialize_cell should handle it
            self.element_types[idx] = 'unknown'


        # Pass the explicit properties from the call to materialization
        temp_cell = self.materialize_cell(concept_id, force_materialize=True, explicit_properties=properties)

        # Set position
        pos_dict = temp_cell.organelles.get('position', {'x': random.uniform(-10, 10), 'y': random.uniform(-10, 10), 'z': random.uniform(-10, 10)})
        self.positions[idx] = [pos_dict.get('x', 0), pos_dict.get('y', 0), pos_dict.get('z', 0)]
        if temp_cell:
            # Now that the cell is materialized and has its organelles, update the numpy arrays
            self.element_types[idx] = temp_cell.element_type
            self.diets[idx] = temp_cell.organelles.get('diet', 'omnivore')
            self.genders[idx] = temp_cell.organelles.get('gender', '')
            self.labels[idx] = temp_cell.organelles.get('label', concept_id)
        else:
            # Fallback if materialization fails, but we've already set element_type above
            self.diets[idx] = 'omnivore'
            self.genders[idx] = ''
            self.labels[idx] = concept_id

        # --- Final Override ---
        # Ensure that properties passed directly to add_cell take ultimate precedence,
        # especially for controlled setups like genesis_simulator.
        if properties:
            if 'element_type' in properties:
                self.element_types[idx] = properties['element_type']
            if 'diet' in properties:
                self.diets[idx] = properties['diet']
            if 'label' in properties:
                self.labels[idx] = properties['label']
            if 'gender' in properties:
                self.genders[idx] = properties['gender']

    def materialize_cell(self, concept_id: str, force_materialize: bool = False, explicit_properties: Optional[Dict] = None) -> Optional[Cell]:
        if not force_materialize and concept_id in self.materialized_cells:
            return self.materialized_cells[concept_id]
        if concept_id in self.quantum_states:
            idx = self.id_to_idx.get(concept_id)
            if idx is None:
                self.logger.error(f"Quantum state for '{concept_id}' exists, but it has no index in the world.")
                return None

            # Fetch node properties from KG
            node_data = self.wave_mechanics.kg_manager.get_node(concept_id)
            initial_properties = node_data.copy() if node_data else {}

            # Merge with explicit properties, which take precedence
            if explicit_properties:
                initial_properties.update(explicit_properties)

            state = self.quantum_states[concept_id]
            cell = Cell(concept_id, self.primordial_dna, initial_properties=initial_properties, initial_energy=self.energy[idx])
            cell.age = state.get('age', 0)
            cell.is_alive = self.is_alive_mask[idx]

            # Crucially, update the numpy array with the correct element type upon materialization
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

        if len(self.cell_ids) == 0:
            return []

        # Calculate energy changes from all sources
        energy_deltas = self._calculate_energy_deltas()

        # Process major state changes and actions
        newly_born_cells = []
        self._process_animal_actions(energy_deltas)
        newly_born_cells.extend(self._process_life_cycles(energy_deltas))

        # Apply final physics and cleanup
        self._apply_physics_and_cleanup(energy_deltas, newly_born_cells)

        return newly_born_cells

    def _calculate_energy_deltas(self) -> np.ndarray:
        """Calculates all passive energy changes for the step."""
        num_cells = len(self.cell_ids)
        adj_matrix_csr = self.adjacency_matrix.tocsr()
        energy_deltas = np.zeros_like(self.energy, dtype=np.float32)

        # Environmental effects (Sunlight, Night decay)
        cycle_position = self.time_step % self.day_length
        self.time_of_day = 'day' if cycle_position < self.day_length / 2 else 'night'
        if self.time_of_day == 'day':
            sun_node = self.wave_mechanics.kg_manager.get_node('sun')
            if sun_node:
                sunlight_energy = sun_node.get('activation_energy', 3.0)
                life_mask = (self.element_types == 'life') & self.is_alive_mask
                energy_deltas[life_mask] += sunlight_energy
        else: # Night
            energy_deltas[self.is_alive_mask] -= 0.2
            animal_mask = (self.element_types == 'animal') & self.is_alive_mask
            energy_deltas[animal_mask] += 0.15

        # Nurturing for plants
        life_mask = (self.element_types == 'life') & self.is_alive_mask
        nurturing_mask = (self.element_types == 'water') | (self.element_types == 'earth')
        if np.any(life_mask) and np.any(nurturing_mask):
            life_to_nurture_connections = adj_matrix_csr[life_mask][:, nurturing_mask]
            nurturing_counts = np.array(life_to_nurture_connections.sum(axis=1)).flatten()
            energy_deltas[life_mask] += nurturing_counts * 0.5

        # Physiological effects (Aging, Injury)
        self.age[self.is_alive_mask] += 1
        old_age_mask = (self.age > self.max_age * 0.8) & self.is_alive_mask
        energy_deltas[old_age_mask] -= 0.5
        energy_deltas[self.is_injured] -= 1.0

        # Physical laws (Gravity)
        flying_mask = (self.positions[:, 2] > 0) & self.is_alive_mask
        altitude = self.positions[flying_mask, 2]
        energy_deltas[flying_mask] -= altitude * 0.02

        return energy_deltas

    def _process_animal_actions(self, energy_deltas: np.ndarray):
        """Handles all AI, decision-making, and actions for animals."""
        adj_matrix_csr = self.adjacency_matrix.tocsr()
        animal_mask = (self.element_types == 'animal') & self.is_alive_mask
        animal_indices = np.where(animal_mask)[0]

        # Basic random movement & cohesion
        movement_vectors = np.random.randn(np.sum(animal_mask), 3) * 0.1
        self.positions[animal_mask] += movement_vectors
        fish_mask = (self.labels == 'fish') & self.is_alive_mask
        fish_indices = np.where(fish_mask)[0]
        for i in fish_indices:
            connected_indices = adj_matrix_csr[i].indices
            school_mask = fish_mask[connected_indices]
            schoolmates = connected_indices[school_mask]
            if schoolmates.size > 0:
                center_of_mass = self.positions[schoolmates].mean(axis=0)
                self.positions[i] += (center_of_mass - self.positions[i]) * 0.05

        # AI Action Loop
        for i in animal_indices:
            if not self.is_alive_mask[i]: continue

            if self.emotions[i] == 'sorrow' and random.random() < 0.5:
                self.logger.info(f"EMOTION: '{self.cell_ids[i]}' is paralyzed by sorrow.")
                continue

            target_idx, action_type = self._select_animal_action(i, adj_matrix_csr)

            if target_idx != -1:
                self._execute_animal_action(i, target_idx, action_type, energy_deltas)

    def _select_animal_action(self, actor_idx: int, adj_matrix_csr: csr_matrix) -> Tuple[int, str]:
        """Selects the best action (hunt or protect) for an animal."""
        # Default action
        target_idx = -1
        action_type = 'hunt'

        # Protection check for humans
        if self.labels[actor_idx] == 'human':
            kin_connections = adj_matrix_csr[actor_idx]
            strong_kin_indices = kin_connections.indices[kin_connections.data > 0.8]
            if strong_kin_indices.size > 0:
                # The strong protect the weak (or equals, like a child)
                weaker_kin_mask = (self.prestige[strong_kin_indices] <= self.prestige[actor_idx]) & \
                                  (strong_kin_indices != actor_idx) & \
                                  self.is_alive_mask[strong_kin_indices]
                protected_kin_indices = strong_kin_indices[weaker_kin_mask]
                if protected_kin_indices.size > 0:
                    for kin_idx in protected_kin_indices:
                        threats_to_kin = adj_matrix_csr[:, kin_idx].tocoo().row
                        for threat_idx in threats_to_kin:
                            if self.is_alive_mask[threat_idx] and self.diets[threat_idx] == 'carnivore':
                                self_preservation = 0.5 + (self.energy[actor_idx] / 100.0)
                                altruism = self.adjacency_matrix[actor_idx, kin_idx]
                                fear = max(0, self.prestige[threat_idx] - self.prestige[actor_idx])
                                courage = 1.0 + (fear / 10.0)
                                if altruism * courage > self_preservation:
                                    return threat_idx, 'protect'

        # Hunting target selection
        all_connected = adj_matrix_csr[actor_idx].indices

        # --- THE KINSHIP TABOO ---
        # A sacred law: one does not hunt their own kin (strong bond).
        kin_connections = adj_matrix_csr[actor_idx]
        kin_indices = set(kin_connections.indices[kin_connections.data > 0.8])
        # ---

        prey_indices = []
        actor_diet = self.diets[actor_idx]

        if actor_diet in ['carnivore', 'omnivore']:
            # Potential prey are animals...
            mask = (self.element_types[all_connected] == 'animal') & \
                   (all_connected != actor_idx) & \
                   self.is_alive_mask[all_connected]
            potential_prey = all_connected[mask]
            # ...that are NOT kin.
            non_kin_prey = [p for p in potential_prey if p not in kin_indices]
            prey_indices.extend(non_kin_prey)

        if actor_diet in ['herbivore', 'omnivore']:
            mask = (self.element_types[all_connected] == 'life') & self.is_alive_mask[all_connected]
            prey_indices.extend(all_connected[mask])

        if prey_indices:
            # Select the weakest among the valid (non-kin) targets
            target_idx = prey_indices[np.argmin(self.energy[np.array(prey_indices)])]

        return target_idx, action_type

    def _execute_animal_action(self, actor_idx: int, target_idx: int, action_type: str, energy_deltas: np.ndarray):
        """Executes the chosen action (move, hunt, etc.)."""
        # Move towards target
        direction = self.positions[target_idx] - self.positions[actor_idx]
        if np.linalg.norm(direction) > 0:
            self.positions[actor_idx] += (direction / np.linalg.norm(direction)) * 0.2

        # Resolve combat if in range
        if np.linalg.norm(self.positions[actor_idx] - self.positions[target_idx]) < 1.5:
            self.logger.info(f"COMBAT: '{self.cell_ids[actor_idx]}' engages '{self.cell_ids[target_idx]}'.")

            courage_multiplier = 1.0
            if action_type == 'protect':
                fear = max(0, self.prestige[target_idx] - self.prestige[actor_idx])
                courage_multiplier = 1.0 + (fear / 5.0)
                if courage_multiplier > 1.5:
                    draw = self.latent_energy[actor_idx] * 0.1
                    self.latent_energy[actor_idx] -= draw
                    boost = draw * 2.0
                    energy_deltas[actor_idx] += boost
                    courage_multiplier += boost / 10.0
                    self.logger.info(f"TRANSCENDENCE: '{self.cell_ids[actor_idx]}' burns life force for power!")
                self.logger.info(f"COURAGE: '{self.cell_ids[actor_idx]}' fights with {courage_multiplier:.2f}x power.")

            energy_transfer = 10.0 * courage_multiplier
            energy_deltas[actor_idx] += energy_transfer
            energy_deltas[target_idx] -= energy_transfer
            self.is_injured[target_idx] = True
            self.prestige[actor_idx] += 1 * courage_multiplier
            self.prestige[target_idx] -= 1

            # Post-combat emotions
            if self.is_alive_mask[actor_idx]: self.emotions[actor_idx] = 'joy'
            if action_type == 'protect': self.emotions[actor_idx] = 'relief'

            # Spiritual Resonance
            if action_type == 'protect':
                resonance = 0.5 * courage_multiplier
                self.wave_mechanics.inject_stimulus('love', resonance)
                self.wave_mechanics.inject_stimulus('protection', resonance)
                self.logger.info(f"Spiritual Resonance: Courageous act resonated (Strength: {resonance:.2f}).")


    def _process_life_cycles(self, energy_deltas: np.ndarray) -> List[Cell]:
        """Handles birth, growth, and reproduction for all entities."""
        newly_born_cells = []
        adj_matrix_csr = self.adjacency_matrix.tocsr()

        # --- Plant Life Cycle ---
        plant_mask = (self.element_types == 'life') & self.is_alive_mask
        plant_indices = np.where(plant_mask)[0]
        can_grow = (self.energy[plant_indices] > 10.0) & (self.growth_stages[plant_indices] < 3)
        self.growth_stages[plant_indices[can_grow]] += 1
        energy_deltas[plant_indices[can_grow]] -= 5
        fruiting_mask = (self.growth_stages[plant_indices] == 3) & (self.energy[plant_indices] > 20.0)
        fruiting_indices = plant_indices[fruiting_mask]
        for i in fruiting_indices:
            energy_deltas[i] -= 15
            new_seed_id = f"plant_{self.time_step}_{i}"
            new_cell = Cell(new_seed_id, self.primordial_dna, initial_properties={'element_type': 'life'}, initial_energy=10.0)
            newly_born_cells.append(new_cell)
            self.growth_stages[i] = 1

        # --- Animal Mating and Reproduction ---
        animal_mask = (self.element_types == 'animal') & self.is_alive_mask
        animal_indices = np.where(animal_mask)[0]
        # Mating readiness increases if not hungry
        not_hungry_mask = self.energy[animal_indices] > 25.0
        self.mating_readiness[animal_indices[not_hungry_mask]] += 0.1
        # Hunger reduces readiness
        hungry_mask = self.energy[animal_indices] < 15.0
        self.mating_readiness[animal_indices[hungry_mask]] = 0

        female_mask = (self.genders[animal_indices] == 'female') & (self.mating_readiness[animal_indices] >= 1.0)
        fertile_female_indices = animal_indices[female_mask]
        for i in fertile_female_indices:
            connected_indices = adj_matrix_csr[i].indices
            male_mask = (self.genders[connected_indices] == 'male') & (self.mating_readiness[connected_indices] >= 1.0)
            potential_mates = connected_indices[male_mask]
            if potential_mates.size > 0:
                mate_idx = random.choice(potential_mates)
                energy_deltas[i] -= 15.0
                new_animal_id = f"{self.labels[i]}_{self.time_step}"
                parent_cell = self.materialize_cell(self.cell_ids[i])
                child_props = parent_cell.organelles.copy()
                child_props['gender'] = random.choice(['male', 'female'])
                new_cell = Cell(new_animal_id, self.primordial_dna, initial_properties=child_props, initial_energy=10.0)
                newly_born_cells.append(new_cell)
                self.mating_readiness[i] = 0.0
                self.mating_readiness[mate_idx] = 0.0
                self.logger.info(f"Mating: '{self.cell_ids[i]}' and '{self.cell_ids[mate_idx]}' produced '{new_animal_id}'.")
                break

        return newly_born_cells


    def _apply_physics_and_cleanup(self, energy_deltas: np.ndarray, newly_born_cells: List[Cell]):
        """Applies final energy changes, handles death, and integrates new cells."""
        adj_matrix_csr = self.adjacency_matrix.tocsr()
        num_cells = len(self.cell_ids)

        # Apply all energy deltas calculated during the step
        self.energy += energy_deltas

        # Add newly born cells to the world
        for cell in newly_born_cells:
            if cell.id not in self.id_to_idx:
                self.add_cell(cell.id, dna=cell.nucleus['dna'], properties=cell.organelles, initial_energy=cell.energy)

        # Process death for cells with low energy
        apoptosis_mask = (self.energy < 0.1) & self.is_alive_mask
        self.is_alive_mask[apoptosis_mask] = False
        self.energy[apoptosis_mask] = 0.0

        dead_cell_indices = np.where(apoptosis_mask)[0]
        for dead_idx in dead_cell_indices:
            cell_id = self.cell_ids[dead_idx]
            if cell_id in self.materialized_cells:
                dead_cell = self.materialized_cells[cell_id]

                # Organic matter cycling
                connected_indices = adj_matrix_csr[dead_idx].indices
                for conn_idx in connected_indices:
                    if self.is_alive_mask[conn_idx] and self.element_types[conn_idx] == 'earth':
                        self.energy[conn_idx] += dead_cell.energy

                # Law of Mortality (Insight) & Law of Emotion (Sorrow)
                connections_to_deceased = adj_matrix_csr[:, dead_idx].tocoo().row
                for survivor_idx in connections_to_deceased:
                    if self.is_alive_mask[survivor_idx]:
                        if self.labels[survivor_idx] == 'human':
                            self.insight[survivor_idx] += 1
                            self.logger.info(f"Insight Gained: '{self.cell_ids[survivor_idx]}' observed the death of '{cell_id}'.")
                        if self.adjacency_matrix[survivor_idx, dead_idx] > 0.8:
                            self.emotions[survivor_idx] = 'sorrow'
                            self.logger.info(f"EMOTION: '{self.cell_ids[survivor_idx]}' feels sorrow for the loss of '{cell_id}'.")

                self.graveyard.append(dead_cell)
                del self.materialized_cells[cell_id]

            if cell_id in self.quantum_states:
                self.quantum_states[cell_id]['existence_probability'] = 0.0

        # Final state synchronization
        self._sync_states_to_objects()

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
