
import random
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from .cell import Cell
from .chronicle import Chronicle
from .game_objects import Inventory, Item, Recipe
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

        # --- Civilization Attributes ---
        self.continent = np.array([], dtype='<U10') # e.g., 'East', 'West'
        self.culture = np.array([], dtype='<U10') # e.g., 'wuxia', 'knight'

        # --- Game System Attributes ---
        self.hp = np.array([], dtype=np.float32)
        self.max_hp = np.array([], dtype=np.float32)
        self.mp = np.array([], dtype=np.float32)
        self.max_mp = np.array([], dtype=np.float32)
        self.power_system = np.array([], dtype='<U10') # e.g., 'aura', 'mana', 'none'
        self.power_system_awakened = np.array([], dtype=bool)
        self.hunger = np.array([], dtype=np.float32)
        self.max_hunger = np.array([], dtype=np.float32)
        self.strength = np.array([], dtype=np.float32)
        self.intelligence = np.array([], dtype=np.float32)
        self.max_strength = np.array([], dtype=np.float32)
        self.max_intelligence = np.array([], dtype=np.float32)
        self.wisdom = np.array([], dtype=np.float32)
        self.max_wisdom = np.array([], dtype=np.float32)
        self.talent_strength = np.array([], dtype=np.float32)
        self.talent_intelligence = np.array([], dtype=np.float32)
        self.talent_wisdom = np.array([], dtype=np.float32)
        self.rank = np.array([], dtype=np.int8) # The Starlight Hierarchy (1-7)

        # --- Metaphysical Attributes ---
        self.aether = np.array([], dtype=np.float32) # 에테르 (Primordial Energy)
        self.emotions = np.array([], dtype='<U10') # joy, sorrow, anger, fear


        # --- SciPy Sparse Matrix for Connections ---
        self.adjacency_matrix = lil_matrix((0, 0), dtype=np.float32)

        # --- Game Object Management ---
        self.inventories: List[Optional[Inventory]] = []
        self.recipes: Dict[str, Recipe] = {}
        self._initialize_recipes()

        # --- Farmland Attributes ---
        self.farmland_state = np.array([], dtype='<U10') # e.g., 'none', 'tilled', 'planted'
        self.water_level = np.array([], dtype=np.float32)
        self.crop_type = np.array([], dtype='<U20') # e.g., 'wheat', 'bean'
        self.crop_growth_stage = np.array([], dtype=np.int8)


    def _resize_matrices(self, new_size: int):
        current_size = len(self.cell_ids)
        if new_size <= current_size:
            return
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
        self.continent = np.pad(self.continent, (0, new_size - current_size), 'constant', constant_values='')
        self.culture = np.pad(self.culture, (0, new_size - current_size), 'constant', constant_values='')
        self.emotions = np.pad(self.emotions, (0, new_size - current_size), 'constant', constant_values='neutral')
        self.aether = np.pad(self.aether, (0, new_size - current_size), 'constant', constant_values=100.0)

        # Game System Attributes Padding
        self.hp = np.pad(self.hp, (0, new_size - current_size), 'constant', constant_values=100.0)
        self.max_hp = np.pad(self.max_hp, (0, new_size - current_size), 'constant', constant_values=100.0)
        self.mp = np.pad(self.mp, (0, new_size - current_size), 'constant', constant_values=10.0)
        self.max_mp = np.pad(self.max_mp, (0, new_size - current_size), 'constant', constant_values=10.0)
        self.power_system = np.pad(self.power_system, (0, new_size - current_size), 'constant', constant_values='none')
        self.power_system_awakened = np.pad(self.power_system_awakened, (0, new_size - current_size), 'constant', constant_values=False)
        self.hunger = np.pad(self.hunger, (0, new_size - current_size), 'constant', constant_values=100.0)
        self.max_hunger = np.pad(self.max_hunger, (0, new_size - current_size), 'constant', constant_values=100.0)
        self.strength = np.pad(self.strength, (0, new_size - current_size), 'constant', constant_values=5.0)
        self.intelligence = np.pad(self.intelligence, (0, new_size - current_size), 'constant', constant_values=5.0)
        self.max_strength = np.pad(self.max_strength, (0, new_size - current_size), 'constant', constant_values=50.0)
        self.max_intelligence = np.pad(self.max_intelligence, (0, new_size - current_size), 'constant', constant_values=50.0)
        self.wisdom = np.pad(self.wisdom, (0, new_size - current_size), 'constant', constant_values=5.0)
        self.max_wisdom = np.pad(self.max_wisdom, (0, new_size - current_size), 'constant', constant_values=50.0)
        self.talent_strength = np.pad(self.talent_strength, (0, new_size - current_size), 'constant', constant_values=0.1)
        self.talent_intelligence = np.pad(self.talent_intelligence, (0, new_size - current_size), 'constant', constant_values=0.1)
        self.talent_wisdom = np.pad(self.talent_wisdom, (0, new_size - current_size), 'constant', constant_values=0.1)
        self.rank = np.pad(self.rank, (0, new_size - current_size), 'constant', constant_values=1)

        new_positions = np.zeros((new_size, 3), dtype=np.float32)
        if current_size > 0:
            new_positions[:current_size, :] = self.positions
        self.positions = new_positions

        new_adj = lil_matrix((new_size, new_size), dtype=np.float32)
        if self.adjacency_matrix.shape[0] > 0:
            new_adj[:current_size, :current_size] = self.adjacency_matrix
        self.adjacency_matrix = new_adj

        # Resize inventories list
        if new_size > len(self.inventories):
            self.inventories.extend([None] * (new_size - len(self.inventories)))

        # Farmland Attributes Padding
        self.farmland_state = np.pad(self.farmland_state, (0, new_size - current_size), 'constant', constant_values='none')
        self.water_level = np.pad(self.water_level, (0, new_size - current_size), 'constant', constant_values=0.0)
        self.crop_type = np.pad(self.crop_type, (0, new_size - current_size), 'constant', constant_values='')
        self.crop_growth_stage = np.pad(self.crop_growth_stage, (0, new_size - current_size), 'constant', constant_values=0)

    def _initialize_recipes(self):
        """Initializes all available crafting recipes in the world."""
        stone_axe = Recipe(
            name='stone_axe',
            ingredients={'wood': 1, 'stone': 1},
            output=Item(name='stone_axe', weight=3.0)
        )
        self.recipes[stone_axe.name] = stone_axe

        cloth_armor = Recipe(
            name='cloth_armor',
            ingredients={'fiber': 5},
            output=Item(name='cloth_armor', weight=2.0)
        )
        self.recipes[cloth_armor.name] = cloth_armor

    def add_cell(self, concept_id: str, dna: Optional[Dict] = None, properties: Optional[Dict] = None, _record_event: bool = True):
        if concept_id in self.quantum_states:
            return
        if self.chronicle and _record_event:
            details = {'concept_id': concept_id, 'properties': properties or {}}
            scopes = [concept_id]
            event = self.chronicle.record_event('cell_added', details, scopes, self.branch_id, self.parent_event_id)
            self.parent_event_id = event['id']
        self.quantum_states[concept_id] = {'existence_probability': 1.0, 'age': 0}
        idx = len(self.cell_ids)
        if idx >= self.adjacency_matrix.shape[0]:
            self._resize_matrices(max(idx + 1, 100))
        self.cell_ids.append(concept_id)
        self.id_to_idx[concept_id] = idx
        self.is_alive_mask[idx] = True
        self.connection_counts[idx] = 0
        self.growth_stages[idx] = 0 # Start at seed stage
        self.age[idx] = 0
        self.is_injured[idx] = False
        self.prestige[idx] = 0.0
        self.max_age[idx] = random.randint(80, 120)
        self.insight[idx] = 0.0
        self.aether[idx] = 100.0
        self.emotions[idx] = 'neutral'

        # Initialize Game System Attributes
        # TODO: Base these initial values on species, dna, properties etc.
        self.max_hp[idx] = 100.0
        self.hp[idx] = self.max_hp[idx]
        self.max_mp[idx] = 10.0
        self.mp[idx] = self.max_mp[idx]
        self.power_system[idx] = 'none'
        self.power_system_awakened[idx] = False
        self.max_hunger[idx] = 100.0
        self.hunger[idx] = self.max_hunger[idx]
        self.strength[idx] = 5.0
        self.intelligence[idx] = 5.0
        self.talent_strength[idx] = random.uniform(0.05, 0.2) # Natural talent variation
        self.talent_intelligence[idx] = random.uniform(0.05, 0.2)
        self.wisdom[idx] = 5.0
        self.talent_wisdom[idx] = random.uniform(0.05, 0.2)
        self.max_strength[idx] = 50.0 + (self.talent_strength[idx] * 100) # Talent affects max potential
        self.max_intelligence[idx] = 50.0 + (self.talent_intelligence[idx] * 100)
        self.max_wisdom[idx] = 50.0 + (self.talent_wisdom[idx] * 100)
        self.rank[idx] = 1 # Start at Seedling rank

        # Pass the explicit properties from the call to materialization
        temp_cell = self.materialize_cell(concept_id, force_materialize=True, explicit_properties=properties)

        # Set position
        pos_dict = temp_cell.organelles.get('position', {'x': random.uniform(-10, 10), 'y': random.uniform(-10, 10), 'z': random.uniform(-10, 10)})
        self.positions[idx, 0] = pos_dict.get('x', 0)
        self.positions[idx, 1] = pos_dict.get('y', 0)
        self.positions[idx, 2] = pos_dict.get('z', 0)

        # Now that the cell is materialized, update the numpy arrays from its properties
        self.element_types[idx] = temp_cell.element_type
        self.diets[idx] = temp_cell.organelles.get('diet', 'omnivore')
        self.genders[idx] = temp_cell.organelles.get('gender', '')
        self.labels[idx] = temp_cell.organelles.get('label', concept_id)

        # --- Final Override ---
        # Ensure that explicit properties passed to add_cell take ultimate precedence.
        if properties:
            self.element_types[idx] = properties.get('element_type', self.element_types[idx])
            self.diets[idx] = properties.get('diet', self.diets[idx])
            self.labels[idx] = properties.get('label', self.labels[idx])
            self.genders[idx] = properties.get('gender', self.genders[idx])
            self.continent[idx] = properties.get('continent', self.continent[idx])
            self.culture[idx] = properties.get('culture', self.culture[idx])

        # --- Inventory Initialization for Humans ---
        if self.labels[idx] == 'human':
            base_weight = 10.0
            max_weight = base_weight + (self.strength[idx] * 2)
            self.inventories[idx] = Inventory(max_weight=max_weight)
            # Temporary: Give humans some seeds to start with for testing
            self.inventories[idx].add_item(Item(name='wheat_seed', quantity=5, weight=0.1))

        # --- Farmland Initialization ---
        self.farmland_state[idx] = 'none'
        self.water_level[idx] = 0.0
        self.crop_type[idx] = ''
        self.crop_growth_stage[idx] = 0

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
            # The Cell object might not need all game stats, but it's cleaner to remove energy
            cell = Cell(concept_id, self.primordial_dna, initial_properties=initial_properties)
            cell.age = state.get('age', 0)
            cell.is_alive = self.is_alive_mask[idx]

            self.materialized_cells[concept_id] = cell
            return cell
        return None

    def _sync_states_to_objects(self):
        for i, cell_id in enumerate(self.cell_ids):
            if cell_id in self.materialized_cells:
                cell = self.materialized_cells[cell_id]
                # Energy is no longer part of the lightweight Cell object
                cell.is_alive = self.is_alive_mask[i]

    def run_simulation_step(self) -> List[Cell]:
        if self.chronicle:
            event = self.chronicle.record_event('simulation_step_run', {}, [], self.branch_id, self.parent_event_id)
            self.parent_event_id = event['id']
        self.time_step += 1

        if len(self.cell_ids) == 0:
            return []

        # Calculate HP changes from all sources
        hp_deltas = self._calculate_hp_deltas()

        # Process major state changes and actions
        newly_born_cells = []
        self._process_animal_actions(hp_deltas)
        newly_born_cells.extend(self._process_life_cycles(hp_deltas))

        # Apply final physics and cleanup
        self._apply_physics_and_cleanup(hp_deltas, newly_born_cells)

        return newly_born_cells

    def _calculate_hp_deltas(self) -> np.ndarray:
        """Calculates all passive HP changes for the step."""
        num_cells = len(self.cell_ids)
        adj_matrix_csr = self.adjacency_matrix.tocsr()
        hp_deltas = np.zeros_like(self.hp, dtype=np.float32)

        # Environmental effects (Sunlight, Night decay)
        cycle_position = self.time_step % self.day_length
        self.time_of_day = 'day' if cycle_position < self.day_length / 2 else 'night'
        if self.time_of_day == 'day':
            sun_node = self.wave_mechanics.kg_manager.get_node('sun')
            if sun_node:
                sunlight_hp_gain = sun_node.get('activation_energy', 3.0) # Plants get HP from sun
                life_mask = (self.element_types == 'life') & self.is_alive_mask
                hp_deltas[life_mask] += sunlight_hp_gain
        else: # Night
            hp_deltas[self.is_alive_mask] -= 0.2 # All living things lose a bit of HP at night
            animal_mask = (self.element_types == 'animal') & self.is_alive_mask
            hp_deltas[animal_mask] += 0.15 # Animals are more active at night

        # Nurturing for plants
        life_mask = (self.element_types == 'life') & self.is_alive_mask
        nurturing_mask = (self.element_types == 'water') | (self.element_types == 'earth')
        if np.any(life_mask) and np.any(nurturing_mask):
            life_to_nurture_connections = adj_matrix_csr[life_mask][:, nurturing_mask]
            nurturing_counts = np.array(life_to_nurture_connections.sum(axis=1)).flatten()
            hp_deltas[life_mask] += nurturing_counts * 0.5

        # Physiological effects (Aging, Injury)
        self.age[self.is_alive_mask] += 1
        old_age_mask = (self.age > self.max_age * 0.8) & self.is_alive_mask
        hp_deltas[old_age_mask] -= 0.5
        hp_deltas[self.is_injured] -= 1.0

        # Physical laws (Gravity)
        flying_mask = (self.positions[:, 2] > 0) & self.is_alive_mask
        altitude = self.positions[flying_mask, 2]
        hp_deltas[flying_mask] -= altitude * 0.02 # Flying costs HP

        # --- Survival System ---
        # Hunger decreases over time
        hunger_decay_rate = self.max_hunger / (self.day_length * 3) # 3 days to starve
        self.hunger -= (hunger_decay_rate * self.is_alive_mask) # Apply decay only to live cells
        self.hunger = np.maximum(0, self.hunger) # Hunger cannot be negative

        # Starvation effect
        starving_mask = (self.hunger == 0) & self.is_alive_mask
        hp_deltas[starving_mask] -= 1.0 # Lose HP when starving


        # --- Metaphysical Energy Conversion ---
        # Cultures can convert aether to MP, but cannot exceed max_mp
        can_convert_mask = self.is_alive_mask & (self.aether > 0) & (self.mp < self.max_mp)

        # Wuxia culture converts aether to mp (naegong)
        wuxia_mask = (self.culture == 'wuxia') & can_convert_mask
        potential_mp_gain_wuxia = np.minimum(self.aether[wuxia_mask], 0.1) * 1.1 # More efficient
        actual_mp_gain_wuxia = np.minimum(potential_mp_gain_wuxia, self.max_mp[wuxia_mask] - self.mp[wuxia_mask])
        aether_cost_wuxia = actual_mp_gain_wuxia / 1.1
        self.aether[wuxia_mask] -= aether_cost_wuxia
        self.mp[wuxia_mask] += actual_mp_gain_wuxia

        # Knight culture converts aether to mp (mana)
        knight_mask = (self.culture == 'knight') & can_convert_mask
        potential_mp_gain_knight = np.minimum(self.aether[knight_mask], 0.1)
        actual_mp_gain_knight = np.minimum(potential_mp_gain_knight, self.max_mp[knight_mask] - self.mp[knight_mask])
        aether_cost_knight = actual_mp_gain_knight
        self.aether[knight_mask] -= aether_cost_knight
        self.mp[knight_mask] += actual_mp_gain_knight

        # --- MP Regeneration ---
        mp_regeneration_rate = 0.1 # A small amount of MP regenerates each step
        can_regen_mp_mask = self.is_alive_mask & (self.mp < self.max_mp)
        self.mp[can_regen_mp_mask] = np.minimum(self.max_mp[can_regen_mp_mask], self.mp[can_regen_mp_mask] + mp_regeneration_rate)

        # --- Farmland Cycle ---
        # Water evaporates over time
        self.water_level = np.maximum(0, self.water_level - 0.5)

        # Crops grow if watered
        can_grow_crop_mask = (self.farmland_state == 'planted') & (self.water_level > 1.0) & (self.crop_growth_stage < 4) # 4 stages: seed, sprout, grown, harvestable
        self.crop_growth_stage[can_grow_crop_mask] += 1


        return hp_deltas

    def _process_animal_actions(self, hp_deltas: np.ndarray):
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
                self._execute_animal_action(i, target_idx, action_type, hp_deltas)

    def _select_animal_action(self, actor_idx: int, adj_matrix_csr: csr_matrix) -> Tuple[int, str]:
        """Selects the best action (hunt or protect) for an animal."""
        # Default action
        target_idx = -1
        action_type = 'hunt'

        # --- Human-specific Civilization Actions ---
        if self.labels[actor_idx] == 'human':
            all_connected = adj_matrix_csr[actor_idx].indices
            inventory = self.inventories[actor_idx]

            # 1. Craft tools if resources are available
            if inventory:
                # Check for stone axe ingredients
                if 'wood' in inventory.items and inventory.items['wood'].quantity >= 1 and \
                   'stone' in inventory.items and inventory.items['stone'].quantity >= 1 and \
                   'stone_axe' not in inventory.items: # Don't craft if they already have one
                   # Target is self for crafting action
                   return actor_idx, 'craft'

            # 2. Harvest grown crops
            harvestable_mask = (self.farmland_state[all_connected] == 'planted') & (self.crop_growth_stage[all_connected] >= 4)
            if np.any(harvestable_mask):
                harvestable_indices = all_connected[harvestable_mask]
                target_idx = random.choice(harvestable_indices)
                return target_idx, 'harvest'

            # 2. Plant seeds if there's tilled land and they have seeds
            if inventory and 'wheat_seed' in inventory.items:
                tilled_mask = (self.farmland_state[all_connected] == 'tilled')
                if np.any(tilled_mask):
                    tilled_indices = all_connected[tilled_mask]
                    target_idx = random.choice(tilled_indices)
                    return target_idx, 'plant_seed'

            # 3. Till land if there's un-tilled earth nearby
            if self.hunger[actor_idx] > 40:
                earth_mask = (self.element_types[all_connected] == 'earth') & (self.farmland_state[all_connected] == 'none')
                if np.any(earth_mask):
                    earth_indices = all_connected[earth_mask]
                    # Find the closest earth plot to till
                    earth_positions = self.positions[earth_indices]
                    actor_position = self.positions[actor_idx]
                    distances = np.linalg.norm(earth_positions - actor_position, axis=1)
                    closest_earth_idx = earth_indices[np.argmin(distances)]
                    return closest_earth_idx, 'till_land'

            # 4. Check for nearby resources to gather
            forest_mask = (self.labels[all_connected] == 'forest') & self.is_alive_mask[all_connected]
            if np.any(forest_mask) and self.hunger[actor_idx] > 30: # Only gather if not too hungry
                forest_indices = all_connected[forest_mask]
                forest_positions = self.positions[forest_indices]
                actor_position = self.positions[actor_idx]
                distances = np.linalg.norm(forest_positions - actor_position, axis=1)
                closest_forest_idx = forest_indices[np.argmin(distances)]
                return closest_forest_idx, 'gather_wood'


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
                                self_preservation = 0.5 + (self.hp[actor_idx] / self.max_hp[actor_idx]) # Use HP ratio
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
        kin_indices = set(kin_connections.indices[kin_connections.data >= 0.8])
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
            target_idx = prey_indices[np.argmin(self.hp[np.array(prey_indices)])]

        return target_idx, action_type

    def _execute_animal_action(self, actor_idx: int, target_idx: int, action_type: str, hp_deltas: np.ndarray):
        """Executes the chosen action (move, hunt, etc.)."""
        # Move towards target
        direction = self.positions[target_idx] - self.positions[actor_idx]
        distance = np.linalg.norm(direction)
        speed = 0.2
        if distance > 0:
            # Move by speed or the remaining distance, whichever is smaller, to prevent overshooting.
            move_distance = min(speed, distance)
            self.positions[actor_idx] += (direction / distance) * move_distance

        # Resolve combat if in range
        if action_type in ['hunt', 'protect'] and np.linalg.norm(self.positions[actor_idx] - self.positions[target_idx]) < 1.5:
            self.logger.info(f"COMBAT: '{self.cell_ids[actor_idx]}' engages '{self.cell_ids[target_idx]}'.")

            courage_multiplier = 1.0
            if action_type == 'protect':
                fear = max(0, self.prestige[target_idx] - self.prestige[actor_idx])
                courage_multiplier = 1.0 + (fear / 5.0)
                if courage_multiplier > 1.5:
                    # Burn life force (HP) for a power boost
                    hp_cost = self.hp[actor_idx] * 0.1
                    hp_deltas[actor_idx] -= hp_cost
                    boost = hp_cost * 2.0
                    hp_deltas[actor_idx] += boost # This is a temporary boost for this combat round
                    courage_multiplier += boost / 10.0
                    self.logger.info(f"TRANSCENDENCE: '{self.cell_ids[actor_idx]}' burns life force for power!")
                self.logger.info(f"COURAGE: '{self.cell_ids[actor_idx]}' fights with {courage_multiplier:.2f}x power.")

            hp_transfer = 10.0 * courage_multiplier # Damage dealt
            hp_deltas[actor_idx] += hp_transfer # Gain HP from successful hunt/defense
            hp_deltas[target_idx] -= hp_transfer # Lose HP from being attacked
            self.is_injured[target_idx] = True
            self.prestige[actor_idx] += 1 * courage_multiplier
            self.prestige[target_idx] -= 1

            # --- Stat Growth from Action ---
            # Combat increases strength
            strength_gain = 0.1 + (self.talent_strength[actor_idx] * 0.5) # Talent-based growth rate
            self.strength[actor_idx] = min(self.max_strength[actor_idx], self.strength[actor_idx] + strength_gain)


            # Post-combat emotions
            if self.is_alive_mask[actor_idx]: self.emotions[actor_idx] = 'joy'
            if action_type == 'protect': self.emotions[actor_idx] = 'relief'

            # Spiritual Resonance
            if action_type == 'protect':
                resonance = 0.5 * courage_multiplier
                self.wave_mechanics.inject_stimulus('love', resonance)
                self.wave_mechanics.inject_stimulus('protection', resonance)
                self.logger.info(f"Spiritual Resonance: Courageous act resonated (Strength: {resonance:.2f}).")

        elif action_type == 'gather_wood' and np.linalg.norm(self.positions[actor_idx] - self.positions[target_idx]) < 1.5:
            inventory = self.inventories[actor_idx]
            if inventory:
                wood_item = Item(name='wood', quantity=1, weight=2.0)
                if inventory.add_item(wood_item):
                    self.logger.info(f"GATHERING: '{self.cell_ids[actor_idx]}' gathered wood.")
                    # Action costs a little bit of hunger
                    self.hunger[actor_idx] = max(0, self.hunger[actor_idx] - 2.0)
                    # Action increases Strength
                    str_gain = 0.1 + (self.talent_strength[actor_idx] * 0.5)
                    self.strength[actor_idx] = min(self.max_strength[actor_idx], self.strength[actor_idx] + str_gain)
                else:
                    self.logger.info(f"GATHERING: '{self.cell_ids[actor_idx]}' failed to gather wood (overweight).")

        elif action_type == 'till_land' and np.linalg.norm(self.positions[actor_idx] - self.positions[target_idx]) < 1.5:
            if self.element_types[target_idx] == 'earth' and self.farmland_state[target_idx] == 'none':
                self.farmland_state[target_idx] = 'tilled'
                self.logger.info(f"FARMING: '{self.cell_ids[actor_idx]}' tilled the land '{self.cell_ids[target_idx]}'.")
                self.hunger[actor_idx] = max(0, self.hunger[actor_idx] - 5.0) # Tilling is hard work
                str_gain = 0.2 + (self.talent_strength[actor_idx] * 0.5) # More STR gain than gathering
                self.strength[actor_idx] = min(self.max_strength[actor_idx], self.strength[actor_idx] + str_gain)

        elif action_type == 'plant_seed' and np.linalg.norm(self.positions[actor_idx] - self.positions[target_idx]) < 1.5:
            inventory = self.inventories[actor_idx]
            if inventory and self.farmland_state[target_idx] == 'tilled':
                # For now, let's assume they have wheat seeds
                if inventory.remove_item('wheat_seed', 1):
                    self.farmland_state[target_idx] = 'planted'
                    self.crop_type[target_idx] = 'wheat'
                    self.crop_growth_stage[target_idx] = 1 # Stage 1: Sprout
                    self.logger.info(f"FARMING: '{self.cell_ids[actor_idx]}' planted wheat in '{self.cell_ids[target_idx]}'.")
                    self.hunger[actor_idx] = max(0, self.hunger[actor_idx] - 1.0)
                    # Planting is an intelligent action
                    int_gain = 0.1 + (self.talent_intelligence[actor_idx] * 0.5)
                    self.intelligence[actor_idx] = min(self.max_intelligence[actor_idx], self.intelligence[actor_idx] + int_gain)
                else:
                    self.logger.info(f"FARMING: '{self.cell_ids[actor_idx]}' tried to plant but had no seeds.")

        elif action_type == 'harvest' and np.linalg.norm(self.positions[actor_idx] - self.positions[target_idx]) < 1.5:
            inventory = self.inventories[actor_idx]
            if inventory and self.farmland_state[target_idx] == 'planted' and self.crop_growth_stage[target_idx] >= 4:
                crop = self.crop_type[target_idx]
                # Reset the farmland plot
                self.farmland_state[target_idx] = 'none' # Becomes fallow, needs tilling again
                self.crop_type[target_idx] = ''
                self.crop_growth_stage[target_idx] = 0

                # Add harvest to inventory
                harvest_item = Item(name=crop, quantity=3, weight=0.5) # 1 seed yields 3 food
                seed_item = Item(name=f"{crop}_seed", quantity=1, weight=0.1)
                inventory.add_item(harvest_item)
                inventory.add_item(seed_item)

                self.logger.info(f"FARMING: '{self.cell_ids[actor_idx]}' harvested {crop}.")
                self.hunger[actor_idx] = max(0, self.hunger[actor_idx] - 2.0)

                # Harvesting is a wise action
                wis_gain = 0.1 + (self.talent_wisdom[actor_idx] * 0.5)
                self.wisdom[actor_idx] = min(self.max_wisdom[actor_idx], self.wisdom[actor_idx] + wis_gain)

                # Wisdom bonus: chance for extra seed
                if random.random() < (self.wisdom[actor_idx] / 100.0):
                    bonus_seed = Item(name=f"{crop}_seed", quantity=1, weight=0.1)
                    inventory.add_item(bonus_seed)
                    self.logger.info(f"FARMING: Wisdom provided an extra seed to '{self.cell_ids[actor_idx]}'.")

        elif action_type == 'craft':
            # The 'target_idx' for crafting is the actor themselves
            inventory = self.inventories[actor_idx]
            if inventory:
                # AI Logic to decide what to craft will go here. For now, try to craft a stone axe.
                recipe_name = 'stone_axe'
                if recipe_name in self.recipes:
                    recipe = self.recipes[recipe_name]

                    # Check if all ingredients are available
                    can_craft = True
                    for ingredient, required_qty in recipe.ingredients.items():
                        if ingredient not in inventory.items or inventory.items[ingredient].quantity < required_qty:
                            can_craft = False
                            break

                    if can_craft:
                        # Consume ingredients
                        for ingredient, required_qty in recipe.ingredients.items():
                            inventory.remove_item(ingredient, required_qty)

                        # Add crafted item
                        inventory.add_item(recipe.output)
                        self.logger.info(f"CRAFTING: '{self.cell_ids[actor_idx]}' crafted a {recipe_name}.")
                        self.hunger[actor_idx] = max(0, self.hunger[actor_idx] - 3.0)

                        # Crafting is a highly intelligent action
                        int_gain = 0.5 + (self.talent_intelligence[actor_idx] * 0.8)
                        self.intelligence[actor_idx] = min(self.max_intelligence[actor_idx], self.intelligence[actor_idx] + int_gain)
                    #else:
                        #self.logger.info(f"CRAFTING: '{self.cell_ids[actor_idx]}' failed to craft {recipe_name} (missing ingredients).")


    def _process_life_cycles(self, hp_deltas: np.ndarray) -> List[Cell]:
        """Handles birth, growth, and reproduction for all entities."""
        newly_born_cells = []
        adj_matrix_csr = self.adjacency_matrix.tocsr()

        # --- Plant Life Cycle ---
        plant_mask = (self.element_types == 'life') & self.is_alive_mask
        plant_indices = np.where(plant_mask)[0]
        can_grow = (self.hp[plant_indices] > 10.0) & (self.growth_stages[plant_indices] < 3)
        self.growth_stages[plant_indices[can_grow]] += 1
        hp_deltas[plant_indices[can_grow]] -= 5
        fruiting_mask = (self.growth_stages[plant_indices] == 3) & (self.hp[plant_indices] > 20.0)
        fruiting_indices = plant_indices[fruiting_mask]
        for i in fruiting_indices:
            hp_deltas[i] -= 15
            new_seed_id = f"plant_{self.time_step}_{i}"
            # New cells are added via add_cell, which sets their initial HP
            new_cell = Cell(new_seed_id, self.primordial_dna, initial_properties={'element_type': 'life'})
            newly_born_cells.append(new_cell)
            self.growth_stages[i] = 1

        # --- Animal Mating and Reproduction ---
        animal_mask = (self.element_types == 'animal') & self.is_alive_mask
        animal_indices = np.where(animal_mask)[0]
        # Mating readiness increases if healthy (high HP percentage) and not hungry
        healthy_mask = (self.hp[animal_indices] / self.max_hp[animal_indices] > 0.8) & (self.hunger[animal_indices] > 50.0)
        self.mating_readiness[animal_indices[healthy_mask]] += 0.1
        # Poor health or hunger reduces readiness
        unhealthy_mask = (self.hp[animal_indices] / self.max_hp[animal_indices] < 0.5) | (self.hunger[animal_indices] < 20.0)
        self.mating_readiness[animal_indices[unhealthy_mask]] = 0

        female_mask = (self.genders[animal_indices] == 'female') & (self.mating_readiness[animal_indices] >= 1.0)
        fertile_female_indices = animal_indices[female_mask]
        for i in fertile_female_indices:
            connected_indices = adj_matrix_csr[i].indices
            male_mask = (self.genders[connected_indices] == 'male') & (self.mating_readiness[connected_indices] >= 1.0)
            potential_mates = connected_indices[male_mask]
            if potential_mates.size > 0:
                mate_idx = random.choice(potential_mates)
                hp_deltas[i] -= 15.0 # HP cost for giving birth
                new_animal_id = f"{self.labels[i]}_{self.time_step}"
                parent_cell = self.materialize_cell(self.cell_ids[i])
                child_props = parent_cell.organelles.copy() if parent_cell else {}
                child_props['gender'] = random.choice(['male', 'female'])
                new_cell = Cell(new_animal_id, self.primordial_dna, initial_properties=child_props)
                newly_born_cells.append(new_cell)
                self.mating_readiness[i] = 0.0
                self.mating_readiness[mate_idx] = 0.0
                self.logger.info(f"Mating: '{self.cell_ids[i]}' and '{self.cell_ids[mate_idx]}' produced '{new_animal_id}'.")
                break

        return newly_born_cells


    def _apply_physics_and_cleanup(self, hp_deltas: np.ndarray, newly_born_cells: List[Cell]):
        """Applies final HP changes, handles death, and integrates new cells."""
        adj_matrix_csr = self.adjacency_matrix.tocsr()
        num_cells = len(self.cell_ids)

        # Apply all HP deltas calculated during the step
        self.hp += hp_deltas
        # Ensure HP does not exceed max_hp or drop below zero
        self.hp = np.clip(self.hp, 0, self.max_hp)


        # Add newly born cells to the world
        for cell in newly_born_cells:
            if cell.id not in self.id_to_idx:
                self.add_cell(cell.id, dna=cell.nucleus['dna'], properties=cell.organelles)

        # Process death for cells with zero HP
        apoptosis_mask = (self.hp < 0.1) & self.is_alive_mask
        self.is_alive_mask[apoptosis_mask] = False
        self.hp[apoptosis_mask] = 0.0

        dead_cell_indices = np.where(apoptosis_mask)[0]
        for dead_idx in dead_cell_indices:
            cell_id = self.cell_ids[dead_idx]
            if cell_id in self.materialized_cells:
                dead_cell = self.materialized_cells[cell_id]

                # Organic matter cycling - dead cells' remaining essence returns to the earth
                connected_indices = adj_matrix_csr[dead_idx].indices
                for conn_idx in connected_indices:
                    if self.is_alive_mask[conn_idx] and self.element_types[conn_idx] == 'earth':
                        # Earth gains HP from the decomposed matter
                        self.hp[conn_idx] += self.max_hp[dead_idx] * 0.25

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

        # --- Power System Awakening ---
        # Check for strength-based awakening (Aura)
        strength_awakening_mask = (self.strength >= 50.0) & (self.power_system_awakened == False) & self.is_alive_mask
        if np.any(strength_awakening_mask):
            self.power_system[strength_awakening_mask] = 'aura'
            self.power_system_awakened[strength_awakening_mask] = True
            # Optional: Log the awakening event
            for idx in np.where(strength_awakening_mask)[0]:
                self.logger.info(f"AWAKENING: '{self.cell_ids[idx]}' has awakened the power of Aura!")

        # Check for intelligence-based awakening (Mana)
        intelligence_awakening_mask = (self.intelligence >= 50.0) & (self.power_system_awakened == False) & self.is_alive_mask
        if np.any(intelligence_awakening_mask):
            self.power_system[intelligence_awakening_mask] = 'mana'
            self.power_system_awakened[intelligence_awakening_mask] = True
            for idx in np.where(intelligence_awakening_mask)[0]:
                self.logger.info(f"AWAKENING: '{self.cell_ids[idx]}' has awakened the power of Mana!")


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

    def inject_stimulus(self, concept_id: str, hp_boost: float, _record_event: bool = True):
        if concept_id in self.id_to_idx:
            if self.chronicle and _record_event:
                details = {'concept_id': concept_id, 'hp_boost': hp_boost}
                scopes = [concept_id]
                event = self.chronicle.record_event('stimulus_injected', details, scopes, self.branch_id, self.parent_event_id)
                self.parent_event_id = event['id']
            idx = self.id_to_idx[concept_id]
            if self.is_alive_mask[idx]:
                self.hp[idx] += hp_boost

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
        print(f"Living Cells: {len(self.is_alive_mask[self.is_alive_mask].tolist())}, Dead Cells (Archived): {len(self.graveyard)}")

        # This is inefficient for large worlds, but invaluable for debugging small scenarios.
        for i, cell_id in enumerate(self.cell_ids):
            if not self.is_alive_mask[i]:
                continue

            label = self.labels[i]
            hp = self.hp[i]
            max_hp = self.max_hp[i]
            mp = self.mp[i]
            max_mp = self.max_mp[i]
            hunger = self.hunger[i]
            strength = self.strength[i]
            intelligence = self.intelligence[i]
            rank = self.rank[i]
            age = self.age[i]

            status_parts = [
                f"<Cell: {label} (Rank: {rank})",
                f"HP: {hp:.1f}/{max_hp:.1f}",
                f"MP: {mp:.1f}/{max_mp:.1f}",
                f"Hunger: {hunger:.1f}",
                f"STR: {strength:.1f}",
                f"INT: {intelligence:.1f}",
                f"WIS: {self.wisdom[i]:.1f}",
                f"Age: {age}",
                "Status: Alive>"
            ]
            print(f"  - {' | '.join(status_parts)}")

        print("-------------------------\n")
