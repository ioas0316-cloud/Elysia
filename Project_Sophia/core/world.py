
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
        for state in self.quantum_states.values():
            state['age'] += 1
        num_cells = len(self.cell_ids)
        if num_cells == 0:
            return []

        # --- Law of Atmosphere (Oxygen Cycle) ---
        num_plants = np.sum((self.element_types == 'life') & self.is_alive_mask)
        num_animals = np.sum((self.element_types == 'animal') & self.is_alive_mask)

        # Plants produce oxygen during the day
        if self.time_of_day == 'day':
            self.oxygen_level += num_plants * 0.1

        # Animals consume oxygen
        self.oxygen_level -= num_animals * 0.05
        self.oxygen_level = max(0, min(200, self.oxygen_level)) # Clamp oxygen level

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

        # Oxygen level effect on animal energy
        animal_mask_for_o2 = (self.element_types == 'animal') & self.is_alive_mask
        if self.oxygen_level > 120.0:
            energy_deltas[animal_mask_for_o2] += 0.1 # Energy efficiency boost
        elif self.oxygen_level < 80.0:
            energy_deltas[animal_mask_for_o2] -= 0.2 # Energy drain from hypoxia

        cycle_position = self.time_step % self.day_length
        self.time_of_day = 'day' if cycle_position < self.day_length / 2 else 'night'
        if self.time_of_day == 'day':
            if self.wave_mechanics and self.wave_mechanics.kg_manager:
                sun_node = self.wave_mechanics.kg_manager.get_node('sun')
                if sun_node:
                    sunlight_energy = sun_node.get('activation_energy', 3.0) # Increased sunlight energy
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

        # --- Law of Sleep (Energy Conservation at Night) ---
        if self.time_of_day == 'night':
            # General energy decay for all living things
            energy_deltas[self.is_alive_mask] -= 0.2
            # Animals conserve more energy by 'sleeping' (reduced decay)
            animal_mask = (self.element_types == 'animal') & self.is_alive_mask
            energy_deltas[animal_mask] += 0.15 # Effectively making their decay -0.05

        # --- Laws of Deprivation (Aging, Injury, Hunger) ---
        self.age[self.is_alive_mask] += 1

        # Aging effect
        old_age_mask = (self.age > self.max_age * 0.8) & self.is_alive_mask
        energy_deltas[old_age_mask] -= 0.5

        # Injury effect
        energy_deltas[self.is_injured] -= 1.0

        # Hunger effect on mating readiness
        hungry_mask = (self.energy < 15.0) & (self.element_types == 'animal')
        self.mating_readiness[hungry_mask] = 0

        # --- Law of Gravity (Energy cost for flying) ---
        flying_mask = (self.positions[:, 2] > 0) & self.is_alive_mask
        altitude = self.positions[flying_mask, 2]
        energy_deltas[flying_mask] -= altitude * 0.02 # Energy cost proportional to height

        # --- Law of Movement and Swarming ---
        # Basic random movement for all animals
        animal_mask = (self.element_types == 'animal') & self.is_alive_mask
        movement_vectors = np.random.randn(np.sum(animal_mask), 3) * 0.1
        self.positions[animal_mask] += movement_vectors

        # Cohesion for fish
        fish_mask = (self.labels == 'fish') & self.is_alive_mask
        fish_indices = np.where(fish_mask)[0]
        for i in fish_indices:
            connected_indices = adj_matrix_csr[i].indices
            school_mask = fish_mask[connected_indices]
            schoolmates = connected_indices[school_mask]
            if schoolmates.size > 0:
                center_of_mass = self.positions[schoolmates].mean(axis=0)
                direction_to_center = center_of_mass - self.positions[i]
                self.positions[i] += direction_to_center * 0.05 # Move towards center


        # --- Law of Predation (Diet-based) ---
        animal_mask = (self.element_types == 'animal') & self.is_alive_mask
        animal_indices = np.where(animal_mask)[0]

        for i in animal_indices:
            # Carnivores hunt other animals
            if self.diets[i] == 'carnivore':
                # New, more robust method to find prey
                all_connected_indices = adj_matrix_csr[i].indices

                # Filter for connections that are animals and not the hunter itself
                prey_mask = (self.element_types[all_connected_indices] == 'animal') & (all_connected_indices != i)
                potential_prey_global_indices = all_connected_indices[prey_mask]

                if potential_prey_global_indices.size > 0:
                    target_prey_global_idx = -1
                    # --- Law of Swarm Hunting (for Sharks) ---
                    if self.labels[i] == 'shark':
                        best_target_idx = -1
                        min_school_size = float('inf')

                        for target_global_idx in potential_prey_global_indices:
                            # Don't attack other sharks
                            if self.labels[target_global_idx] == 'shark':
                                continue

                            # Count schoolmates around the target
                            target_connections = adj_matrix_csr[target_global_idx].indices
                            # We need to check if the connections are both fish and alive
                            schoolmate_mask = (self.labels[target_connections] == 'fish') & self.is_alive_mask[target_connections]
                            school_size = np.sum(schoolmate_mask)

                            if school_size < min_school_size:
                                min_school_size = school_size
                                best_target_idx = target_global_idx

                        if best_target_idx != -1:
                            target_prey_global_idx = best_target_idx
                    else:
                        # Default random target selection for other carnivores
                        target_prey_global_idx = random.choice(potential_prey_global_indices)

                    # If no valid target was found (e.g., shark only saw other sharks), skip to next hunter
                    if target_prey_global_idx == -1:
                        continue

                    # --- Law of Prey Tracking ---
                    # Move towards the chosen target
                    direction_to_prey = self.positions[target_prey_global_idx] - self.positions[i]
                    # Normalize and apply a small step, added to the random movement
                    self.positions[i] += (direction_to_prey / np.linalg.norm(direction_to_prey)) * 0.2

                    # --- Law of "Prestige" and "Submission" ---
                    if self.prestige[target_prey_global_idx] > self.prestige[i] + 5:
                        self.logger.info(f"Submission: {self.cell_ids[i]} avoided fighting stronger {self.cell_ids[target_prey_global_idx]}.")
                        continue # Avoid the fight

                    # --- Law of Altitude (Positional Advantage) ---
                    attacker_altitude = self.positions[i, 2]
                    prey_altitude = self.positions[target_prey_global_idx, 2]
                    altitude_advantage = 0
                    if attacker_altitude > prey_altitude + 5: # Attacking from significantly above
                        altitude_advantage = 2
                        self.logger.info(f"Altitude Advantage: {self.cell_ids[i]} attacks from above!")

                    # Fight happens
                    energy_transfer = 10.0
                    energy_deltas[i] += energy_transfer
                    energy_deltas[target_prey_global_idx] -= energy_transfer
                    self.is_injured[target_prey_global_idx] = True

                    # Prestige changes
                    self.prestige[i] += 1 + altitude_advantage
                    self.prestige[target_prey_global_idx] -= 1

                    self.logger.info(f"Predation: Carnivore {self.cell_ids[i]} defeated {self.cell_ids[target_prey_global_idx]}.")

            # Herbivores eat plants
            elif self.diets[i] == 'herbivore':
                plant_mask = (self.element_types == 'life') & self.is_alive_mask
                if np.any(plant_mask):
                    connections_to_plants = adj_matrix_csr[i, :][:, plant_mask]
                    prey_indices = connections_to_plants.indices
                    if prey_indices.size > 0:
                        target_prey_local_idx = random.choice(prey_indices)
                        target_prey_global_idx = np.where(plant_mask)[0][target_prey_local_idx]

                        energy_transfer = 5.0
                        energy_deltas[i] += energy_transfer
                        energy_deltas[target_prey_global_idx] -= energy_transfer
                        self.logger.info(f"Grazing: Herbivore {self.cell_ids[i]} ate plant {self.cell_ids[target_prey_global_idx]}.")

            # Reproduction for animals
            if self.energy[i] + energy_deltas[i] > 50.0:
                energy_deltas[i] -= self.energy[i] / 2

        self.energy += energy_deltas
        newly_born_cells = []

        # --- Law of Life's Cycles (Growth, Fruiting, Reproduction) for Plants ---
        plant_mask = (self.element_types == 'life') & self.is_alive_mask
        plant_indices = np.where(plant_mask)[0]

        # Growth Stage Advancement
        # Stage 0: Seed, 1: Growing, 2: Flowering, 3: Fruiting
        can_grow = (self.energy[plant_indices] > 10.0) & (self.growth_stages[plant_indices] < 3)
        self.growth_stages[plant_indices[can_grow]] += 1
        self.energy[plant_indices[can_grow]] -= 5 # Energy cost to grow

        # Reproduction for Fruiting Plants
        fruiting_mask = (self.growth_stages[plant_indices] == 3) & (self.energy[plant_indices] > 20.0)
        fruiting_indices = plant_indices[fruiting_mask]

        # Temp list to handle connections for new cells
        new_seeds_to_connect = []

        for i in fruiting_indices:
            self.energy[i] -= 15 # Energy cost to reproduce

            new_seed_id = f"plant_{self.time_step}_{i}"
            # Create the cell object but don't add to world yet, just to newly_born_cells
            new_cell = Cell(new_seed_id, self.primordial_dna, initial_properties={'element_type': 'life'}, initial_energy=10.0)
            newly_born_cells.append(new_cell)

            # Seed dispersal mechanism: find nearby earth
            connected_indices = adj_matrix_csr[i].indices
            earth_mask = self.element_types[connected_indices] == 'earth'
            fertile_ground_indices = connected_indices[earth_mask]

            if fertile_ground_indices.size > 0:
                # Connect the new seed to a random patch of fertile ground
                target_earth_idx = random.choice(fertile_ground_indices)
                new_seeds_to_connect.append((new_seed_id, self.cell_ids[target_earth_idx]))

            self.logger.info(f"Reproduction: Plant {self.cell_ids[i]} produced a new seed {new_seed_id}.")
            self.growth_stages[i] = 1 # Reset to growing stage after fruiting

        # --- Law of Mating (Hormones and Reproduction) for Animals ---
        animal_mask = (self.element_types == 'animal') & self.is_alive_mask
        animal_indices = np.where(animal_mask)[0]

        # Hormone (Mating Readiness) Increase, considering food availability
        for idx in animal_indices:
            if self.energy[idx] > 25.0:
                has_food = False
                connected_indices = adj_matrix_csr[idx].indices
                if self.diets[idx] == 'carnivore':
                    # Check for nearby animals to eat
                    animal_prey_mask = self.element_types[connected_indices] == 'animal'
                    if np.any(animal_prey_mask):
                        has_food = True
                elif self.diets[idx] == 'herbivore':
                    # Check for nearby plants to eat
                    plant_prey_mask = self.element_types[connected_indices] == 'life'
                    if np.any(plant_prey_mask):
                        has_food = True

                if has_food:
                    self.mating_readiness[idx] += 0.1

        # Mating and Reproduction
        female_mask = (self.genders[animal_indices] == 'female') & (self.mating_readiness[animal_indices] >= 1.0)
        fertile_female_indices = animal_indices[female_mask]

        for i in fertile_female_indices:
            # Find connected, ready males
            connected_indices = adj_matrix_csr[i].indices
            male_mask = (self.genders[connected_indices] == 'male') & (self.mating_readiness[connected_indices] >= 1.0)
            potential_mates = connected_indices[male_mask]

            if potential_mates.size > 0:
                mate_idx = random.choice(potential_mates)

                # Energy cost and birth
                self.energy[i] -= 15.0
                new_animal_id = f"{self.cell_ids[i].split('_')[0]}_{self.time_step}"
                new_gender = random.choice(['male', 'female'])

                # Get parent properties to pass to child
                parent_cell = self.materialize_cell(self.cell_ids[i])
                child_properties = parent_cell.organelles.copy()
                child_properties['gender'] = new_gender

                new_cell = Cell(new_animal_id, self.primordial_dna, initial_properties=child_properties, initial_energy=10.0)
                newly_born_cells.append(new_cell)

                # Reset readiness
                self.mating_readiness[i] = 0.0
                self.mating_readiness[mate_idx] = 0.0

                self.logger.info(f"Mating: {self.cell_ids[i]} and {self.cell_ids[mate_idx]} produced offspring {new_animal_id}.")
                break # One offspring per step per female

        self._sync_states_to_objects()
        living_cells = [self.materialized_cells[self.cell_ids[i]] for i in range(num_cells) if self.is_alive_mask[i] and self.energy[i] > 1.0 and self.cell_ids[i] in self.materialized_cells]
        newly_born_molecules = self._run_chemical_reactions(living_cells)
        newly_born_cells.extend(newly_born_molecules)
        for cell in newly_born_cells:
            if cell.id not in self.id_to_idx:
                 self.add_cell(cell.id, dna=cell.nucleus['dna'], properties=cell.organelles, initial_energy=cell.energy)

        # Connect new seeds to fertile ground after they have been added to the world
        for seed_id, earth_id in new_seeds_to_connect:
            self.add_connection(seed_id, earth_id, strength=0.5)

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
                dead_cell = self.materialized_cells[cell_id]
                dead_cell_idx = self.id_to_idx.get(dead_cell.id)

                # Organic matter cycling
                if dead_cell_idx is not None:
                    connected_indices = adj_matrix_csr[dead_cell_idx].indices
                    for conn_idx in connected_indices:
                        if self.element_types[conn_idx] == 'earth':
                            self.energy[conn_idx] += dead_cell.energy # Transfer remaining energy to the earth

                self.graveyard.append(dead_cell)
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
