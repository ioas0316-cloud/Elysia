
import random
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from .cell import Cell
from .chronicle import Chronicle
from .skills import MARTIAL_STYLES, Move
from .spells import SPELL_BOOK, cast_spell
from .world_event_logger import WorldEventLogger
from ..wave_mechanics import WaveMechanics
from .fields import FieldRegistry


class World:
    """Represents the universe where cells exist, interact, and evolve, optimized with NumPy."""

    def __init__(self, primordial_dna: Dict, wave_mechanics: WaveMechanics,
                 chronicle: Optional[Chronicle] = None, logger: Optional[logging.Logger] = None,
                 branch_id: str = "main", parent_event_id: Optional[str] = None):
        # --- Event Logger ---
        self.event_logger = WorldEventLogger()

        # --- Core Attributes ---
        self.primordial_dna = primordial_dna
        self.wave_mechanics = wave_mechanics
        self.chronicle = chronicle
        self.time_step = 0
        self.logger = logger or logging.getLogger(__name__)

        # --- Martial Arts / Spells ---
        self.martial_styles = MARTIAL_STYLES
        self.spells = SPELL_BOOK
        if self.logger:
            self.logger.info(f"Loaded {len(self.martial_styles)} martial art styles.")

        # --- Celestial Cycle ---
        # Time scaling follows the fractal principle: adjust minutes-per-tick and
        # the world recomputes day/year tick spans and continuous sunlight fields.
        self.minutes_per_tick: float = 10.0  # real minutes represented by one simulation tick
        self.day_length: int = int(max(1, round(24 * 60 / self.minutes_per_tick)))  # ticks per day
        self.year_length_days: float = 365.25
        self.time_of_day = 'day'
        self.axial_tilt_deg: float = 23.4
        # Computed each step
        self.sunlight_field = None  # np.ndarray (width x width) in [0,1]
        self.sun_intensity_global: float = 1.0
        self.ambient_temperature_c: float = 15.0
        # Lunar cycle
        self.month_length_days: float = 29.53
        self.moon_phase: float = 0.0
        self.moonlight_global: float = 0.0
        self.lunar_arousal: float = 0.0
        self.tide_level_global: float = 0.0
        self.lunar_tide_amplitude: float = 1.0

        # --- Atmosphere ---
        self.oxygen_level = 100.0
        self.cloud_cover = 0.2 # 0.0 (clear) to 1.0 (overcast)
        self.humidity = 0.5 # 0.0 to 1.0

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

        # --- Core Game System Attributes ---
        self.is_alive_mask = np.array([], dtype=bool)
        self.hp = np.array([], dtype=np.float32)
        self.max_hp = np.array([], dtype=np.float32)
        self.ki = np.array([], dtype=np.float32)
        self.max_ki = np.array([], dtype=np.float32)
        self.mana = np.array([], dtype=np.float32)
        self.max_mana = np.array([], dtype=np.float32)
        self.faith = np.array([], dtype=np.float32)
        self.max_faith = np.array([], dtype=np.float32)
        self.strength = np.array([], dtype=np.int32)
        self.agility = np.array([], dtype=np.int32)
        self.intelligence = np.array([], dtype=np.int32)
        self.vitality = np.array([], dtype=np.int32)
        self.wisdom = np.array([], dtype=np.int32)
        self.hunger = np.array([], dtype=np.float32)
        self.hydration = np.array([], dtype=np.float32)
        self.temperature = np.array([], dtype=np.float32)
        self.satisfaction = np.array([], dtype=np.float32)


        # --- General Simulation Attributes ---
        self.connection_counts = np.array([], dtype=np.int32)
        self.element_types = np.array([], dtype='<U10')
        self.diets = np.array([], dtype='<U10')
        self.growth_stages = np.array([], dtype=np.int8)
        self.genders = np.array([], dtype='<U6')  # male, female
        self.mating_readiness = np.array([], dtype=np.float32)
        self.age = np.array([], dtype=np.int32)
        self.max_age = np.array([], dtype=np.int32)
        self.is_injured = np.array([], dtype=bool)
        self.is_meditating = np.array([], dtype=bool) # For ?닿린議곗떇, 湲곕룄
        self.positions = np.zeros((0, 3), dtype=np.float32)
        self.labels = np.array([], dtype='<U20')
        self.insight = np.array([], dtype=np.float32)
        self.emotions = np.array([], dtype='<U10') # joy, sorrow, anger, fear

        # --- Civilization Attributes ---
        self.continent = np.array([], dtype='<U10') # e.g., 'East', 'West'
        self.culture = np.array([], dtype='<U10') # e.g., 'wuxia', 'knight'
        self.affiliation = np.array([], dtype='<U20') # e.g., 'Wudang', 'Shaolin'


        # --- SciPy Sparse Matrix for Connections ---
        self.adjacency_matrix = lil_matrix((0, 0), dtype=np.float32)

        # --- Geology (Grid-based) ---
        self.width = 256  # Default size, can be configured
        self.height_map = np.zeros((self.width, self.width), dtype=np.float32)
        self.soil_fertility = np.full((self.width, self.width), 0.5, dtype=np.float32)
        self.wetness = np.zeros((self.width, self.width), dtype=np.float32) # 0.0 (dry) to 1.0 (puddle)
        # Latitude map (y -> latitude radians, +north at top)
        y_coords = np.arange(self.width, dtype=np.float32)
        lat_norm = 0.5 - (y_coords / max(1, self.width))  # +0.5 at top, -0.5 at bottom
        self._lat_radians_row = lat_norm * np.pi  # [-pi/2, +pi/2]

        # --- Emergent Field Layers (Fractal Law Carriers) ---
        # Soft, continuous fields that influence but do not dictate behavior.
        self.threat_field = np.zeros((self.width, self.width), dtype=np.float32)
        self.hydration_field = np.zeros((self.width, self.width), dtype=np.float32)
        # Electromagnetic-like salience field (scalar carrier; E as grad)
        self.em_s = np.zeros((self.width, self.width), dtype=np.float32)
        self._threat_decay = 0.92  # memory of threat (EMA)
        self._threat_sigma = 7.0   # spatial spread of threat influence (grid units)
        self._threat_gain = 1.0    # base contribution gain
        self._cohesion_gain = 0.08 # social cohesion gain toward allies
        # EM field dynamics and coupling (defaults preserve behavior: weights=0)
        self._em_decay = 0.92
        self._em_sigma = 6.0
        self._em_weight_E = 0.0  # soft bias along E (grad of salience)
        self._em_weight_B = 0.0  # soft bias along perp(E) (rotation cue)

        # --- Field Registry (fractal carriers; fields-over-commands) ---
        # Register existing scalar fields so sampling/gradients go through one interface.
        self.fields = FieldRegistry()
        self.fields.register_scalar(
            name="threat",
            scale="micro",
            array_getter=lambda: self.threat_field,
            grad_func=lambda arr, fx, fy: self._sample_field_grad(arr, fx, fy),
        )
        self.fields.register_scalar(
            name="hydration",
            scale="micro",
            array_getter=lambda: self.hydration_field,
            grad_func=lambda arr, fx, fy: self._sample_field_grad(arr, fx, fy),
        )
        self.fields.register_scalar(
            name="em_s",
            scale="micro",
            array_getter=lambda: self.em_s,
            grad_func=lambda arr, fx, fy: self._sample_field_grad(arr, fx, fy),
        )


    def _resize_matrices(self, new_size: int):
        current_size = len(self.cell_ids)
        if new_size <= current_size:
            return

        # --- Core Game System Attributes ---
        self.is_alive_mask = np.pad(self.is_alive_mask, (0, new_size - current_size), 'constant', constant_values=False)
        self.hp = np.pad(self.hp, (0, new_size - current_size), 'constant')
        self.max_hp = np.pad(self.max_hp, (0, new_size - current_size), 'constant')
        self.ki = np.pad(self.ki, (0, new_size - current_size), 'constant')
        self.max_ki = np.pad(self.max_ki, (0, new_size - current_size), 'constant')
        self.mana = np.pad(self.mana, (0, new_size - current_size), 'constant')
        self.max_mana = np.pad(self.max_mana, (0, new_size - current_size), 'constant')
        self.faith = np.pad(self.faith, (0, new_size - current_size), 'constant')
        self.max_faith = np.pad(self.max_faith, (0, new_size - current_size), 'constant')
        self.strength = np.pad(self.strength, (0, new_size - current_size), 'constant')
        self.agility = np.pad(self.agility, (0, new_size - current_size), 'constant')
        self.intelligence = np.pad(self.intelligence, (0, new_size - current_size), 'constant')
        self.vitality = np.pad(self.vitality, (0, new_size - current_size), 'constant')
        self.wisdom = np.pad(self.wisdom, (0, new_size - current_size), 'constant')
        self.hunger = np.pad(self.hunger, (0, new_size - current_size), 'constant', constant_values=100.0)
        self.hydration = np.pad(self.hydration, (0, new_size - current_size), 'constant', constant_values=100.0)
        self.temperature = np.pad(self.temperature, (0, new_size - current_size), 'constant', constant_values=36.5)
        self.satisfaction = np.pad(self.satisfaction, (0, new_size - current_size), 'constant', constant_values=50.0)


        # --- General Simulation Attributes ---
        self.connection_counts = np.pad(self.connection_counts, (0, new_size - current_size), 'constant')
        self.element_types = np.pad(self.element_types, (0, new_size - current_size), 'constant')
        self.diets = np.pad(self.diets, (0, new_size - current_size), 'constant', constant_values='omnivore')
        self.growth_stages = np.pad(self.growth_stages, (0, new_size - current_size), 'constant', constant_values=0)
        self.genders = np.pad(self.genders, (0, new_size - current_size), 'constant', constant_values='')
        self.mating_readiness = np.pad(self.mating_readiness, (0, new_size - current_size), 'constant', constant_values=0.0)
        self.age = np.pad(self.age, (0, new_size - current_size), 'constant', constant_values=0)
        self.max_age = np.pad(self.max_age, (0, new_size - current_size), 'constant', constant_values=100)
        self.is_injured = np.pad(self.is_injured, (0, new_size - current_size), 'constant', constant_values=False)
        self.is_meditating = np.pad(self.is_meditating, (0, new_size - current_size), 'constant', constant_values=False)
        self.labels = np.pad(self.labels, (0, new_size - current_size), 'constant', constant_values='')
        self.insight = np.pad(self.insight, (0, new_size - current_size), 'constant', constant_values=0.0)
        self.emotions = np.pad(self.emotions, (0, new_size - current_size), 'constant', constant_values='neutral')


        # --- Civilization Attributes ---
        self.continent = np.pad(self.continent, (0, new_size - current_size), 'constant', constant_values='')
        self.culture = np.pad(self.culture, (0, new_size - current_size), 'constant', constant_values='')
        self.affiliation = np.pad(self.affiliation, (0, new_size - current_size), 'constant', constant_values='')

        # --- Positional and Connection Data ---
        new_positions = np.zeros((new_size, 3), dtype=np.float32)
        if current_size > 0:
            new_positions[:current_size, :] = self.positions
        self.positions = new_positions

        new_adj = lil_matrix((new_size, new_size), dtype=np.float32)
        if self.adjacency_matrix.shape[0] > 0:
            new_adj[:current_size, :current_size] = self.adjacency_matrix
        self.adjacency_matrix = new_adj

    # -------------------------
    # Time/Season utilities
    # -------------------------
    def set_time_scale(self, minutes_per_tick: float) -> None:
        """Adjust the time fractal scale. Recomputes day_length accordingly."""
        self.minutes_per_tick = max(0.001, float(minutes_per_tick))
        self.day_length = int(max(1, round(24 * 60 / self.minutes_per_tick)))

    def _year_length_ticks(self) -> int:
        return int(max(1, round(self.day_length * self.year_length_days)))

    def _month_length_ticks(self) -> int:
        return int(max(1, round(self.day_length * self.month_length_days)))

    def get_day_phase(self) -> float:
        if self.day_length <= 0:
            return 0.0
        return (self.time_step % self.day_length) / float(self.day_length)

    def get_year_phase(self) -> float:
        yl = self._year_length_ticks()
        if yl <= 0:
            return 0.0
        return (self.time_step % yl) / float(yl)
    def get_season_name(self) -> str:
        p = self.get_year_phase()
        # 4 equal seasons for now
        if p < 0.25:
            return '봄'
        elif p < 0.5:
            return '여름'
        elif p < 0.75:
            return '가을'
        else:
            return '겨울'

    def get_clock_hm(self) -> Tuple[int, int]:
        dp = self.get_day_phase()
        total_minutes = 24 * 60
        m = int(dp * total_minutes)
        return (m // 60) % 24, m % 60

    def get_date_ymd(self) -> Tuple[int, int, int]:
        """(year, month, day) 반환. 연도는 0년부터 시작. 월은 12개월 균등 분할 근사."""
        yl = self._year_length_ticks()
        if yl <= 0 or self.day_length <= 0:
            return 0, 1, 1
        year = int(self.time_step // yl)
        ticks_into_year = int(self.time_step % yl)
        day_of_year = int(ticks_into_year // self.day_length)
        days_in_year = max(1, int(round(self.year_length_days)))
        days_per_month = max(1, int(round(days_in_year / 12.0)))
        month = int(day_of_year // days_per_month) + 1
        if month > 12:
            month = 12
        day = int(day_of_year % days_per_month) + 1
        return year, month, day

    def get_month_phase(self) -> float:
        ml = self._month_length_ticks()
        if ml <= 0:
            return 0.0
        return (self.time_step % ml) / float(ml)

    def get_season_name(self) -> str:
        p = self.get_year_phase()
        # 4 equal seasons for now
        if p < 0.25:
            return '봄'
        elif p < 0.5:
            return '여름'
        elif p < 0.75:
            return '가을'
        else:
            return '겨울'

    def get_clock_hm(self) -> Tuple[int, int]:
        dp = self.get_day_phase()
        total_minutes = 24 * 60
        m = int(dp * total_minutes)
        return (m // 60) % 24, m % 60

    def add_cell(self, concept_id: str, dna: Optional[Dict] = None, properties: Optional[Dict] = None, initial_energy: float = 0.0, _record_event: bool = True):
        if concept_id in self.quantum_states:
            return

        # Ensure properties is a dictionary to prevent AttributeErrors on .get()
        if properties is None:
            properties = {}

        # The concept of 'initial_energy' is deprecated, but kept for legacy compatibility during transition.
        # We will map it to HP for now.
        if self.chronicle and _record_event:
            details = {'concept_id': concept_id, 'initial_hp': initial_energy, 'properties': properties or {}}
            scopes = [concept_id]
            event = self.chronicle.record_event('cell_added', details, scopes, self.branch_id, self.parent_event_id)
            self.parent_event_id = event['id']

        self.quantum_states[concept_id] = {'existence_probability': 1.0, 'age': 0}

        idx = len(self.cell_ids)
        if idx >= self.adjacency_matrix.shape[0]:
            self._resize_matrices(max(idx + 1, 100))

        self.cell_ids.append(concept_id)
        self.id_to_idx[concept_id] = idx

        # --- Initialize Core Attributes ---
        self.is_alive_mask[idx] = True
        self.connection_counts[idx] = 0
        self.growth_stages[idx] = 0 # Start at seed stage
        self.age[idx] = 0
        self.is_injured[idx] = False
        self.is_meditating[idx] = False
        # max_age???섏쨷???쇰꺼/醫낆쓣 ?뚯븙????'???⑥쐞'濡?寃곗젙?섍퀬, ?대? ?€?μ? '?? ?⑥쐞濡?蹂€?섑븳??
        self.insight[idx] = 0.0
        self.emotions[idx] = 'neutral'

        # --- Initialize Game System Stats ---
        # Base stats are set first
        self.strength[idx] = properties.get('strength', 5)
        self.agility[idx] = properties.get('agility', 5)
        self.intelligence[idx] = properties.get('intelligence', 5)
        self.vitality[idx] = properties.get('vitality', 5)
        self.wisdom[idx] = properties.get('wisdom', 5)

        # Derived stats (HP/Ki/Mana/Faith) are calculated from base stats
        self.max_hp[idx] = self.vitality[idx] * 10
        self.hp[idx] = self.max_hp[idx]

        # --- Anti-Hybrid Protocol ---
        # A cell's power system is determined by its culture at birth.
        culture = properties.get('culture', '')
        if culture == 'wuxia':
            self.max_ki[idx] = self.wisdom[idx] * 10
            self.ki[idx] = self.max_ki[idx]
            self.max_mana[idx] = 0
            self.mana[idx] = 0
            self.max_faith[idx] = 0
            self.faith[idx] = 0
        elif culture == 'knight':
            self.max_mana[idx] = self.wisdom[idx] * 10
            self.mana[idx] = self.max_mana[idx]
            self.max_ki[idx] = 0
            self.ki[idx] = 0
            self.max_faith[idx] = 0
            self.faith[idx] = 0
        else: # Default for non-magical entities
            self.max_ki[idx] = 0
            self.ki[idx] = 0
            self.max_mana[idx] = 0
            self.mana[idx] = 0
            self.max_faith[idx] = 0
            self.faith[idx] = 0


        self.hunger[idx] = 100.0
        self.hydration[idx] = 100.0
        self.temperature[idx] = 36.5
        self.satisfaction[idx] = 50.0

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

        # --- Lifespan selection (years -> ticks) based on label/type ---
        def _lifespan_years(label: str, element_type: str) -> int:
            label = (label or '').lower()
            if label == 'human':
                return random.randint(70, 100)
            if label in ('wolf', 'deer'):
                return random.randint(8, 16)
            if label in ('tree',):
                return random.randint(50, 200)
            if label in ('bush', 'plant'):
                return random.randint(3, 12)
            if element_type == 'animal':
                return random.randint(6, 20)
            if element_type == 'life':
                return random.randint(5, 30)
            return random.randint(40, 100)

        current_label = self.labels[idx] if self.labels[idx] else properties.get('label', concept_id)
        current_type = self.element_types[idx]
        years = _lifespan_years(current_label, current_type)
        self.max_age[idx] = max(1, int(years * self._year_length_ticks()))

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
            if 'continent' in properties:
                self.continent[idx] = properties['continent']
            if 'culture' in properties:
                self.culture[idx] = properties['culture']
            if 'affiliation' in properties:
                self.affiliation[idx] = properties['affiliation']

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
            # Pass the current HP from the numpy array to the Cell object
            cell = Cell(concept_id, self.primordial_dna, initial_properties=initial_properties)
            cell.age = state.get('age', 0)
            cell.is_alive = self.is_alive_mask[idx]

            # Sync all game stats to the cell object upon materialization
            cell.hp = self.hp[idx]
            cell.max_hp = self.max_hp[idx]
            cell.ki = self.ki[idx]
            cell.max_ki = self.max_ki[idx]
            cell.mana = self.mana[idx]
            cell.max_mana = self.max_mana[idx]
            cell.faith = self.faith[idx]
            cell.max_faith = self.max_faith[idx]
            cell.strength = self.strength[idx]
            cell.agility = self.agility[idx]
            cell.intelligence = self.intelligence[idx]
            cell.vitality = self.vitality[idx]
            cell.wisdom = self.wisdom[idx]


            # Crucially, update the numpy array with the correct element type upon materialization
            self.element_types[idx] = cell.element_type

            self.materialized_cells[concept_id] = cell
            return cell
        return None

    def _sync_states_to_objects(self):
        """Syncs the critical numpy array states back to any materialized cell objects."""
        for i, cell_id in enumerate(self.cell_ids):
            if cell_id in self.materialized_cells:
                cell = self.materialized_cells[cell_id]
                cell.is_alive = self.is_alive_mask[i]
                cell.hp = self.hp[i]
                cell.ki = self.ki[i]
                cell.mana = self.mana[i]
                cell.faith = self.faith[i]
                # Sync other stats if they can change during simulation
                cell.strength = self.strength[i]
                cell.agility = self.agility[i]
                cell.intelligence = self.intelligence[i]
                cell.vitality = self.vitality[i]
                cell.wisdom = self.wisdom[i]


    def run_simulation_step(self) -> List[Cell]:
        if self.chronicle:
            event = self.chronicle.record_event('simulation_step_run', {}, [], self.branch_id, self.parent_event_id)
            self.parent_event_id = event['id']
        self.time_step += 1

        if len(self.cell_ids) == 0:
            return []

        # Update celestial cycle & weather
        self._update_celestial_fields()
        self._update_weather()

        # Update emergent continuous fields (e.g., threat)
        self._update_emergent_fields()
        self._update_hydration_field()
        self._update_em_perception_field()

        # Update passive resources (MP regen, hunger, starvation)
        self._update_passive_resources()

        # Process major state changes and actions
        newly_born_cells = []
        self._process_animal_actions()
        newly_born_cells.extend(self._process_life_cycles())

        # Apply final physics and cleanup
        self._apply_physics_and_cleanup(newly_born_cells)

        return newly_born_cells

    def _update_weather(self):
        """Updates the global weather conditions and handles weather events like lightning."""
        # Simple random walk for cloud cover and humidity
        self.cloud_cover += random.uniform(-0.05, 0.05)
        self.humidity += random.uniform(-0.05, 0.05)
        self.cloud_cover = np.clip(self.cloud_cover, 0, 1)
        self.humidity = np.clip(self.humidity, 0, 1)

        # --- Rain Event ---
        is_raining = self.cloud_cover > 0.6 and self.humidity > 0.5 and random.random() < 0.2
        if is_raining:
            self.wetness += 0.2
            self.wetness = np.clip(self.wetness, 0, 1)

            # Plants get hydrated by rain
            plant_mask = (self.element_types == 'life') & self.is_alive_mask
            self.hydration[plant_mask] = np.minimum(100, self.hydration[plant_mask] + 10)

            self.logger.info("PROVIDENCE: It has started to rain.")
            self.event_logger.log('RAIN_START', self.time_step)
        else:
            # Dry out
            self.wetness -= 0.01
            self.wetness = np.clip(self.wetness, 0, 1)


        # --- Lightning Event ---
        if self.cloud_cover > 0.8 and self.humidity > 0.7 and random.random() < 0.1:
            if len(self.cell_ids) > 0:
                strike_idx = random.randint(0, len(self.cell_ids) - 1)

                # Strike at the cell's position on the grid
                pos = self.positions[strike_idx]
                x, y = int(pos[0]) % self.width, int(pos[1]) % self.width

                # Boost soil fertility at the strike location
                self.soil_fertility[x, y] = min(1.0, self.soil_fertility[x, y] + 0.5)

                # Damage the cell at the strike location
                if self.is_alive_mask[strike_idx]:
                    damage = random.uniform(20, 50)
                    self.hp[strike_idx] -= damage
                    self.is_injured[strike_idx] = True

                    cell_id = self.cell_ids[strike_idx]
                    self.logger.info(f"PROVIDENCE: Lightning strikes '{cell_id}', dealing {damage:.1f} damage and enriching the soil.")
                    self.event_logger.log('LIGHTNING_STRIKE', self.time_step, cell_id=cell_id, damage=damage)
        
        
    def _update_celestial_fields(self) -> None:
        """Compute sunlight and ambient temperature fields as soft influences.
        Laws as fields: no commands, only continuous context values.
        """
        # Phases
        day_p = self.get_day_phase()
        year_p = self.get_year_phase()
        month_p = self.get_month_phase()
        self.moon_phase = month_p
        # Solar geometry (approximate)
        tilt = np.deg2rad(self.axial_tilt_deg)
        decl = tilt * np.sin(2.0 * np.pi * year_p)  # solar declination
        hour_angle = 2.0 * np.pi * day_p - np.pi  # -pi at midnight, 0 at noon
        # Latitude per row; broadcast over x dimension
        phi = self._lat_radians_row.reshape(1, -1)  # shape (1, width)
        # Sun altitude: sin(alt) = sin ? sin 灌 + cos ? cos 灌 cos h
        sin_alt = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
        intensity_row = np.clip(sin_alt, 0.0, 1.0)  # (1, width)
        # Build field as same along x-axis for simplicity
        field = np.repeat(intensity_row, self.width, axis=0)  # shape (width, width)
        # Clouds attenuate sunlight
        field *= (1.0 - 0.7 * float(self.cloud_cover))
        self.sunlight_field = field.astype(np.float32)
        self.sun_intensity_global = float(np.mean(self.sunlight_field))
        # Moonlight (simple model): bright near full moon, visible at night
        bright = 0.5 - 0.5 * np.cos(2.0 * np.pi * month_p)  # 0=new, 1=full
        night_factor = max(0.0, 1.0 - self.sun_intensity_global)
        self.moonlight_global = float(night_factor * bright * (1.0 - 0.6 * float(self.cloud_cover)))
        self.lunar_arousal = self.moonlight_global
        # Time of day label
        if self.sun_intensity_global < 0.05:
            self.time_of_day = 'night'
        elif self.sun_intensity_global < 0.15:
            self.time_of_day = 'dawn'
        elif self.sun_intensity_global > 0.75:
            self.time_of_day = 'noon'
        else:
            self.time_of_day = 'day'
        season_term = 10.0 * np.sin(2.0 * np.pi * (year_p - 0.20))
        daily_term = 5.0 * np.sin(2.0 * np.pi * (day_p - 0.25))
        cloud_cool = -5.0 * float(self.cloud_cover)
        self.ambient_temperature_c = float(15.0 + season_term + daily_term + cloud_cool)

        # Tide level (global scalar): semi-diurnal + monthly modulation
        # Two highs per day (simplified) and stronger around full/new moon
        semi_diurnal = np.sin(4.0 * np.pi * day_p)  # two cycles per day
        spring_factor = 0.5 + 0.5 * np.cos(2.0 * np.pi * month_p)  # new/full stronger
        self.tide_level_global = float(self.lunar_tide_amplitude * semi_diurnal * spring_factor)

        # Optional ambient events: wolves howl under bright full moon (no physics effect)
        try:
            if self.moonlight_global > 0.75:
                wolf_mask = (self.labels == 'wolf') & self.is_alive_mask
                wolf_indices = np.where(wolf_mask)[0]
                for i in wolf_indices:
                    if random.random() < 0.02:
                        self.event_logger.log('HOWL', self.time_step, cell_id=self.cell_ids[i])
        except Exception:
            pass

    def _update_passive_resources(self):
        """Updates passive resource changes for all living cells (Ki, Mana, Hunger, HP)."""
        # --- Ki Regeneration (very slow) ---
        ki_regen_mask = self.is_alive_mask & (self.ki < self.max_ki)
        self.ki[ki_regen_mask] = np.minimum(self.max_ki[ki_regen_mask], self.ki[ki_regen_mask] + 0.1)

        # --- Mana Regeneration (moderate) ---
        mana_regen_mask = self.is_alive_mask & (self.mana < self.max_mana)
        self.mana[mana_regen_mask] = np.minimum(self.max_mana[mana_regen_mask], self.mana[mana_regen_mask] + 1.0)

        # --- Faith does not regenerate passively ---

        # --- Hunger Depletion ---
        # All living things get hungrier over time.
        self.hunger[self.is_alive_mask] = np.maximum(0, self.hunger[self.is_alive_mask] - 0.5)

        # --- Starvation ---
        # If hunger is 0, the cell starts losing HP.
        starvation_mask = self.is_alive_mask & (self.hunger <= 0)
        self.hp[starvation_mask] -= 2.0 # Penalty for starvation

        # --- Hydration Depletion ---
        self.hydration[self.is_alive_mask] = np.maximum(0, self.hydration[self.is_alive_mask] - 0.7)

        # --- Dehydration ---
        dehydration_mask = self.is_alive_mask & (self.hydration <= 0)
        self.hp[dehydration_mask] -= 2.5 # Penalty for dehydration is slightly higher

        # --- Aging ---
        self.age[self.is_alive_mask] += 1
        old_age_mask = (self.age > self.max_age * 0.8) & self.is_alive_mask
        self.hp[old_age_mask] -= 0.5 # HP decay from old age

        # --- Law of Mortality: Death from old age ---
        death_by_age_mask = (self.age >= self.max_age) & self.is_alive_mask
        self.hp[death_by_age_mask] = 0

        dead_from_age_indices = np.where(death_by_age_mask)[0]
        for idx in dead_from_age_indices:
             self.logger.info(f"PROVIDENCE: '{self.cell_ids[idx]}' has died of old age.")
             self.event_logger.log('DEATH_BY_OLD_AGE', self.time_step, cell_id=self.cell_ids[idx])

    def _update_emergent_fields(self):
        """Updates soft fields (e.g., threat) from distributed sources.
        The field is not a command; it is a context carrier that agents can sense.
        """
        if len(self.cell_ids) == 0:
            return

        # Build a fresh threat imprint from predators and recent injuries
        new_threat = np.zeros_like(self.threat_field)
        try:
            predator_mask = (self.element_types == 'animal') & (self.diets == 'carnivore') & self.is_alive_mask
            predator_indices = np.where(predator_mask)[0]
            if predator_indices.size > 0:
                sigma = self._threat_sigma
                rad = int(max(2, sigma * 3))
                for i in predator_indices:
                    px, py = int(self.positions[i][0]) % self.width, int(self.positions[i][1]) % self.width
                    x0, x1 = max(0, px - rad), min(self.width, px + rad + 1)
                    y0, y1 = max(0, py - rad), min(self.width, py + rad + 1)
                    xs = np.arange(x0, x1) - px
                    ys = np.arange(y0, y1) - py
                    gx = np.exp(-(xs**2) / (2 * sigma * sigma))
                    gy = np.exp(-(ys**2) / (2 * sigma * sigma))
                    patch = (gy[:, None] * gx[None, :]).astype(np.float32)
                    # hunger amplifies perceived threat; strength also contributes
                    amp = self._threat_gain * float(max(0.5, 1.5 - (self.hunger[i] / 100.0))) * float(max(1.0, self.strength[i] / 10.0))
                    new_threat[y0:y1, x0:x1] += amp * patch

            if new_threat.max() > 0:
                new_threat = new_threat / float(new_threat.max())
            self.threat_field = (self._threat_decay * self.threat_field) + ((1.0 - self._threat_decay) * new_threat)
        except Exception:
            # Field updates should never break the sim
            pass

    def _update_hydration_field(self):
        """Updates the hydration field based on wetness, creating a 'water aura'."""
        try:
            # A simple way to spread the wetness is using a convolution,
            # but for simplicity and to avoid adding more dependencies,
            # we can simulate a simple diffusion or blur.
            # A more robust implementation would use scipy.ndimage.gaussian_filter
            from scipy.ndimage import gaussian_filter
            self.hydration_field = gaussian_filter(self.wetness, sigma=10)
        except ImportError:
            # Fallback if scipy is not available
            self.hydration_field = self.wetness

    def _update_em_perception_field(self):
        """Update electromagnetic-like salience field em_s (soft, non-command carrier).
        Sources: presence of alive agents; diffusion+decay; normalized; EMA.
        """
        try:
            if len(self.cell_ids) == 0:
                self.em_s *= self._em_decay
                return

            presence = np.zeros_like(self.em_s)
            alive_idx = np.where(self.is_alive_mask)[0]
            if alive_idx.size > 0:
                px = np.clip(self.positions[alive_idx, 0].astype(np.int32), 0, self.width - 1)
                py = np.clip(self.positions[alive_idx, 1].astype(np.int32), 0, self.width - 1)
                presence[py, px] += 1.0

            try:
                from scipy.ndimage import gaussian_filter
                sal = gaussian_filter(presence, sigma=self._em_sigma)
            except Exception:
                sal = presence

            if sal.max() > 0:
                sal = sal / float(sal.max())
            self.em_s = (self._em_decay * self.em_s) + ((1.0 - self._em_decay) * sal.astype(np.float32))
        except Exception:
            # Perception field must never break the simulation
            pass

    def _sample_field_grad(self, field: np.ndarray, fx: float, fy: float) -> Tuple[float, float]:
        """Returns a finite-difference gradient of a field at world coords (fx, fy)."""
        x = int(np.clip(fx, 1, self.width - 2))
        y = int(np.clip(fy, 1, self.width - 2))
        gx = float(field[y, x + 1] - field[y, x - 1]) * 0.5
        gy = float(field[y + 1, x] - field[y - 1, x]) * 0.5
        return gx, gy

    def _process_animal_actions(self):
        """Handles all AI, decision-making, and actions for animals."""
        adj_matrix_csr = self.adjacency_matrix.tocsr()
        animal_mask = (self.element_types == 'animal') & self.is_alive_mask
        animal_indices = np.where(animal_mask)[0]

        # Basic random movement & cohesion (agility influences pace)
        n_animals = int(np.sum(animal_mask))
        if n_animals > 0:
            movement_vectors = np.random.randn(n_animals, 3) * 0.05
            # Humans: apply soft influences from fields and allies (no hard commands)
            for local_idx, i in enumerate(animal_indices):
                if not self.is_alive_mask[i]:
                    continue
                # is_human = (self.labels[i] == 'human') or (self.culture[i] in ['wuxia', 'knight'])
                # if is_human:
                px, py = float(self.positions[i][0]), float(self.positions[i][1])
                gx, gy = self.fields.grad("threat", px, py)
                avoid = np.array([-gx, -gy, 0.0], dtype=np.float32)
                coh = np.zeros(3, dtype=np.float32)
                kin_attraction = np.zeros(3, dtype=np.float32)
                hydration_seeking = np.zeros(3, dtype=np.float32)
                em_bias = np.zeros(3, dtype=np.float32)

                # --- Magnetoreception for Water ---
                if self.hydration[i] < 70:  # Start seeking water when hydration drops below 70
                    thirst_factor = (1.0 - self.hydration[i] / 100.0) ** 2  # Urgency increases non-linearly
                    hx, hy = self.fields.grad("hydration", px, py)
                    hydration_seeking = np.array([hx, hy, 0.0], dtype=np.float32) * thirst_factor * 0.5

                # --- Electromagnetic-like Perception Bias (soft, defaults off) ---
                if (self._em_weight_E != 0.0) or (self._em_weight_B != 0.0):
                    ex, ey = self.fields.grad("em_s", px, py)
                    e_vec = np.array([ex, ey, 0.0], dtype=np.float32)
                    # Perpendicular for B-like cue
                    b_vec = np.array([-ey, ex, 0.0], dtype=np.float32)
                    em_bias = (self._em_weight_E * e_vec) + (self._em_weight_B * b_vec)

                neigh = adj_matrix_csr[i].indices
                if neigh.size > 0:
                    # General cohesion towards same culture
                    same_culture = [j for j in neigh if self.is_alive_mask[j] and self.culture[j] == self.culture[i]]
                    if len(same_culture) > 0:
                        center = self.positions[same_culture].mean(axis=0)
                        coh = (center - self.positions[i]) * self._cohesion_gain

                    # Strong attraction to kin (Will Field)
                    kin = [j for j in neigh if self.is_alive_mask[j] and adj_matrix_csr[i, j] >= 0.8]
                    if len(kin) > 0:
                        kin_center = self.positions[kin].mean(axis=0)
                        kin_attraction = (kin_center - self.positions[i]) * (self._cohesion_gain * 5.0) # 5x stronger than normal cohesion

                movement_vectors[local_idx] += avoid + coh + kin_attraction + hydration_seeking + em_bias

            speeds = (0.08 + (self.agility[animal_indices] / 100.0) * 0.04).reshape(-1, 1)
            self.positions[animal_mask] += movement_vectors * speeds
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

            # If meditating, the cell is vulnerable and can do nothing else.
            if self.is_meditating[i]:
                self.logger.info(f"'{self.cell_ids[i]}' is meditating and is vulnerable.")
                # While meditating, the action is to continue meditating.
                self._execute_animal_action(i, -1, 'meditate', None)
                continue

            if self.emotions[i] == 'sorrow' and random.random() < 0.5:
                self.logger.info(f"EMOTION: '{self.cell_ids[i]}' is paralyzed by sorrow.")
                continue

            target_idx, action, move = self._select_animal_action(i, adj_matrix_csr)

            # Pass -1 for target_idx if the action doesn't require a target (e.g., meditation)
            self._execute_animal_action(i, target_idx if target_idx is not None else -1, action, move)

    def _select_animal_action(self, actor_idx: int, adj_matrix_csr: csr_matrix) -> Tuple[Optional[int], str, Optional[Move]]:
        """Orchestrates animal action selection by checking needs in priority order."""
        # 1. High-priority survival actions
        survival_action = self._decide_survival_action(actor_idx, adj_matrix_csr)
        if survival_action:
            return survival_action

        # 2. Social, combat, or hunger-driven actions
        social_combat_action = self._decide_social_or_combat_action(actor_idx, adj_matrix_csr)
        if social_combat_action:
            return social_combat_action

        # 3. Default to idle if no other action is taken
        return None, 'idle', None

    def _decide_survival_action(self, actor_idx: int, adj_matrix_csr: csr_matrix) -> Optional[Tuple[Optional[int], str, Optional[Move]]]:
        """Handles immediate, critical survival needs."""
        # Need 1: Drink if thirsty and at a water source
        if self.hydration[actor_idx] < 40:
            actor_pos = self.positions[actor_idx]
            # Check if the animal is currently standing on a 'wet' grid cell
            if self.wetness[int(actor_pos[1]) % self.width, int(actor_pos[0]) % self.width] > 0.5:
                return None, 'drink', None
            # If not at water, the movement logic will handle getting there, so no action needed here.

        # Need 2: Meditate (Ki Circulation) if Ki is low and it's safe to do so
        is_wuxia = self.culture[actor_idx] == 'wuxia'
        if is_wuxia and self.ki[actor_idx] < self.max_ki[actor_idx] * 0.3:
            connected_indices = adj_matrix_csr[actor_idx].indices
            # Check for nearby threats before meditating
            is_safe = not np.any((self.element_types[connected_indices] == 'animal') & self.is_alive_mask[connected_indices] & (connected_indices != actor_idx))
            if is_safe:
                return None, 'meditate', None

        # Add other high-priority actions here (e.g., flee from overwhelming threat)

        return None # No urgent survival action needed

    def _decide_social_or_combat_action(self, actor_idx: int, adj_matrix_csr: csr_matrix) -> Optional[Tuple[Optional[int], str, Optional[Move]]]:
        """Handles hunting, fighting, skill use, and other social behaviors."""
        connected_indices = adj_matrix_csr[actor_idx].indices
        if connected_indices.size == 0:
            return None # No one nearby to interact with

        # Determine if hungry
        is_hungry = self.hunger[actor_idx] < 60
        kin_indices = set(connected_indices[adj_matrix_csr[actor_idx, connected_indices].toarray().flatten() >= 0.8])

        # Find potential targets based on hunger and diet
        potential_targets = []
        actor_diet = self.diets[actor_idx]

        if is_hungry:
            # Find food: non-kin animals for carnivores, plants for herbivores
            if actor_diet in ['carnivore', 'omnivore']:
                mask = (self.element_types[connected_indices] == 'animal') & self.is_alive_mask[connected_indices]
                non_kin_prey = [p_idx for p_idx in connected_indices[mask] if p_idx not in kin_indices]
                potential_targets.extend(non_kin_prey)
            if actor_diet in ['herbivore', 'omnivore']:
                mask = (self.element_types[connected_indices] == 'life') & self.is_alive_mask[connected_indices]
                potential_targets.extend(connected_indices[mask])
        else:
            # If not hungry, might pick a fight with non-kin rivals
            mask = (self.element_types[connected_indices] == 'animal') & self.is_alive_mask[connected_indices]
            non_kin_rivals = [r_idx for r_idx in connected_indices[mask] if r_idx not in kin_indices]
            potential_targets.extend(non_kin_rivals)

        if not potential_targets:
            return None # No suitable targets found

        # Select the weakest target
        target_idx = potential_targets[np.argmin(self.hp[np.array(potential_targets)])]

        # --- Tactical Decision Logic ---
        target_is_animal = self.element_types[target_idx] == 'animal'
        if not target_is_animal:
            return target_idx, 'eat', None # Action is to eat the plant

        # If target is an animal, decide on combat style
        action = 'attack'
        selected_move = None
        culture = self.culture[actor_idx]

        # Wuxia: Use Ki-based martial arts
        if culture == 'wuxia':
            actor_affiliation = self.affiliation[actor_idx]
            if actor_affiliation and actor_affiliation in self.martial_styles:
                style = self.martial_styles[actor_affiliation]
                # Try to use the best available move
                for move in sorted(style.moves, key=lambda m: m.ki_cost, reverse=True):
                    if self.ki[actor_idx] >= move.ki_cost:
                        # Check if stats are sufficient for the move
                        if all(getattr(self, stat)[actor_idx] >= value for stat, value in move.min_stats.items()):
                            selected_move = move
                            break # Found a suitable move

        # Knight: Use Mana-based spells
        elif culture == 'knight' and self.mana[actor_idx] >= 8:
            # Prioritize healing if HP is low
            if self.hp[actor_idx] < self.max_hp[actor_idx] * 0.6:
                action, target_idx = 'cast_heal', None
            # Otherwise, cast an offensive spell if mana is sufficient
            elif self.mana[actor_idx] >= 10:
                action = 'cast_firebolt'

        return target_idx, action, selected_move


    def _execute_animal_action(self, actor_idx: int, target_idx: int, action: str, move: Optional[Move]):
        """Executes the chosen action, including non-target actions like meditation."""

        # --- Handle non-target actions ---
        if action == 'drink':
            self.hydration[actor_idx] = min(100, self.hydration[actor_idx] + 50)
            self.logger.info(f"ACTION: '{self.cell_ids[actor_idx]}' drinks water.")
            self.event_logger.log('DRINK', self.time_step, cell_id=self.cell_ids[actor_idx])
            return

        if action == 'meditate':
            self.is_meditating[actor_idx] = True
            self.logger.info(f"ACTION: '{self.cell_ids[actor_idx]}' begins ?닿린議곗떇.")
            # Rapid Ki regeneration
            self.ki[actor_idx] = min(self.max_ki[actor_idx], self.ki[actor_idx] + 10)
            # HP regeneration and healing effect
            if self.hp[actor_idx] < self.max_hp[actor_idx]:
                self.hp[actor_idx] = min(self.max_hp[actor_idx], self.hp[actor_idx] + 2)
            if self.is_injured[actor_idx] and random.random() < 0.2: # 20% chance to heal injury
                self.is_injured[actor_idx] = False
                self.logger.info(f"HEALING: '{self.cell_ids[actor_idx]}' has healed their internal injuries.")
            # Stop meditating if ki is full
            if self.ki[actor_idx] >= self.max_ki[actor_idx]:
                self.is_meditating[actor_idx] = False
            return

        # Stop meditating if any other action is taken
        self.is_meditating[actor_idx] = False

        if target_idx == -1: return # No target for other actions

        # --- Handle target-based actions ---
        direction = self.positions[target_idx] - self.positions[actor_idx]
        if np.linalg.norm(direction) > 0:
            step = 0.1 + (float(self.agility[actor_idx]) * 0.002)
            self.positions[actor_idx] += (direction / np.linalg.norm(direction)) * step

        if np.linalg.norm(self.positions[actor_idx] - self.positions[target_idx]) < 1.5:
            if action == 'eat' and self.element_types[target_idx] == 'life':
                 self.logger.info(f"ACTION: '{self.cell_ids[actor_idx]}' eats '{self.cell_ids[target_idx]}'.")
                 self.event_logger.log('EAT', self.time_step, actor_id=self.cell_ids[actor_idx], target_id=self.cell_ids[target_idx])
                 self.hp[target_idx] = 0 # Eating kills the plant
                 food_value = 20
                 self.hunger[actor_idx] = min(100, self.hunger[actor_idx] + food_value)
                 return

            damage_multiplier = 1.0
            if move:
                self.logger.info(f"SKILL: '{self.cell_ids[actor_idx]}' uses [{move.name}] on '{self.cell_ids[target_idx]}'.")
                self.ki[actor_idx] -= move.ki_cost
                damage_multiplier = move.apply_effect(self, actor_idx, target_idx, self.hp)
            elif action == 'attack':
                self.logger.info(f"ACTION: '{self.cell_ids[actor_idx]}' attacks '{self.cell_ids[target_idx]}'.")
            elif action == 'cast_firebolt':
                if self.spells.get('firebolt'):
                    # Evasion check
                    evade_chance = min(0.4, float(self.agility[target_idx]) / 100.0)
                    if random.random() < evade_chance:
                        self.logger.info(f"EVADE: '{self.cell_ids[target_idx]}' evaded Firebolt.")
                        self.event_logger.log('EVADE', self.time_step, cell_id=self.cell_ids[target_idx])
                        return
                    result = cast_spell(self, 'firebolt', actor_idx, target_idx)
                    dmg = float(result.get('damage', 0.0))
                    if dmg > 0:
                        self.logger.info(f"SPELL: '{self.cell_ids[actor_idx]}' casts Firebolt on '{self.cell_ids[target_idx]}' ({dmg:.1f}).")
                        self.event_logger.log('SPELL', self.time_step, caster_id=self.cell_ids[actor_idx], spell='firebolt', target_id=self.cell_ids[target_idx], damage=dmg)
                    return
            elif action == 'cast_heal':
                if self.spells.get('heal'):
                    result = cast_spell(self, 'heal', actor_idx, None)
                    healed = float(result.get('heal', 0.0))
                    if healed > 0:
                        self.logger.info(f"SPELL: '{self.cell_ids[actor_idx]}' casts Heal (+{healed:.1f}).")
                        self.event_logger.log('SPELL', self.time_step, caster_id=self.cell_ids[actor_idx], spell='heal', heal=healed)
                    return

            # Evasion chance based on target agility
            evade_chance = min(0.4, float(self.agility[target_idx]) / 100.0)
            if random.random() < evade_chance:
                self.logger.info(f"EVADE: '{self.cell_ids[target_idx]}' evaded the attack.")
                self.event_logger.log('EVADE', self.time_step, cell_id=self.cell_ids[target_idx])
                return

            base_damage = self.strength[actor_idx]
            # Lunar arousal softly increases damage, especially for wolves
            lunar_bonus = 1.0 + 0.25 * float(getattr(self, 'lunar_arousal', 0.0))
            if self.labels[actor_idx] == 'wolf':
                lunar_bonus = 1.0 + 0.5 * float(getattr(self, 'lunar_arousal', 0.0))
            final_damage = max(0, (base_damage + random.randint(-2, 2)) * damage_multiplier * lunar_bonus)
            self.hp[target_idx] -= final_damage
            self.is_injured[target_idx] = True
            self.logger.info(f"COMBAT: Damage dealt: {final_damage:.2f}.")

            if self.hp[target_idx] <= 0:
                self.logger.info(f"HUNT: '{self.cell_ids[actor_idx]}' killed '{self.cell_ids[target_idx]}'.")
                food_value = 50
                self.hunger[actor_idx] = min(100, self.hunger[actor_idx] + food_value)
                self.satisfaction[actor_idx] += 20
                self.emotions[actor_idx] = 'joy'


    def _process_life_cycles(self) -> List[Cell]:
        """Handles birth, growth, and reproduction for all entities."""
        newly_born_cells = []
        adj_matrix_csr = self.adjacency_matrix.tocsr()

        # --- Plant Life Cycle (Simplified, as they don't have HP in the same way) ---
        plant_mask = (self.element_types == 'life') & self.is_alive_mask
        plant_indices = np.where(plant_mask)[0]
        # Plants can fruit if they are mature
        fruiting_mask = (self.growth_stages[plant_indices] == 3)
        fruiting_indices = plant_indices[fruiting_mask]
        for i in fruiting_indices:
            if random.random() < 0.1: # 10% chance to create a seed each turn
                new_seed_id = f"plant_{self.time_step}_{i}"
                parent_cell = self.materialize_cell(self.cell_ids[i])
                child_props = parent_cell.organelles.copy() if parent_cell else {'element_type': 'life'}

                # --- Seed of Will Field: Plant Colonization ---
                # New seed falls near the parent, forming groves and forests.
                parent_pos = self.positions[i]
                spread_radius = 2.0 # How far the seed can fall
                new_pos = {
                    'x': parent_pos[0] + random.uniform(-spread_radius, spread_radius),
                    'y': parent_pos[1] + random.uniform(-spread_radius, spread_radius),
                    'z': parent_pos[2]
                }
                child_props['position'] = new_pos

                new_cell = Cell(new_seed_id, self.primordial_dna, initial_properties=child_props)
                newly_born_cells.append(new_cell)
                self.growth_stages[i] = 1 # Reset to growing stage


        # --- Animal Mating and Reproduction ---
        animal_mask = (self.element_types == 'animal') & self.is_alive_mask
        animal_indices = np.where(animal_mask)[0]
        # Mating readiness increases if not hungry and healthy
        ready_mask = (self.hunger[animal_indices] > 70) & (self.hp[animal_indices] > self.max_hp[animal_indices] * 0.8)
        self.mating_readiness[animal_indices[ready_mask]] = np.minimum(1.0, self.mating_readiness[animal_indices[ready_mask]] + 0.1)

        # Hunger or injury reduces readiness
        not_ready_mask = (self.hunger[animal_indices] < 50) | (self.is_injured[animal_indices])
        self.mating_readiness[animal_indices[not_ready_mask]] = 0

        female_mask = (self.genders[animal_indices] == 'female') & (self.mating_readiness[animal_indices] >= 1.0)
        fertile_female_indices = animal_indices[female_mask]
        for i in fertile_female_indices:
            connected_indices = adj_matrix_csr[i].indices
            male_mask = (self.genders[connected_indices] == 'male') & (self.mating_readiness[connected_indices] >= 1.0)
            potential_mates = connected_indices[male_mask]

            if potential_mates.size > 0:
                mate_idx = random.choice(potential_mates)
                # Gestation costs resources
                self.hp[i] -= 20
                self.hunger[i] -= 30

                new_animal_id = f"{self.labels[i]}_{self.time_step}"
                parent_cell = self.materialize_cell(self.cell_ids[i])
                child_props = parent_cell.organelles.copy()
                child_props['gender'] = random.choice(['male', 'female'])
                new_cell = Cell(new_animal_id, self.primordial_dna, initial_properties=child_props)
                newly_born_cells.append(new_cell)

                # Reset readiness after procreation
                self.mating_readiness[i] = 0.0
                self.mating_readiness[mate_idx] = 0.0
                self.logger.info(f"Mating: '{self.cell_ids[i]}' and '{self.cell_ids[mate_idx]}' produced '{new_animal_id}'.")

                # --- Seed of Will Field: Forge family bonds ---
                # Add the new cell to the world immediately to get its index
                if new_animal_id not in self.id_to_idx:
                    self.add_cell(new_animal_id, dna=new_cell.nucleus['dna'], properties=new_cell.organelles)

                new_animal_idx = self.id_to_idx.get(new_animal_id)

                if new_animal_idx is not None:
                    # Parent-child bonds
                    self.add_connection(self.cell_ids[i], new_animal_id, strength=0.9) # Mother-child
                    self.add_connection(new_animal_id, self.cell_ids[i], strength=0.9)
                    self.add_connection(self.cell_ids[mate_idx], new_animal_id, strength=0.9) # Father-child
                    self.add_connection(new_animal_id, self.cell_ids[mate_idx], strength=0.9)

                    # Sibling bonds (if any) - conceptual placeholder for now
                    self.logger.info(f"WILL_FIELD: Forged family bonds for '{new_animal_id}'.")

                break # Only one birth per turn

        return newly_born_cells


    def _apply_physics_and_cleanup(self, newly_born_cells: List[Cell]):
        """Applies final state changes, handles death, and integrates new cells."""
        adj_matrix_csr = self.adjacency_matrix.tocsr()

        # Add newly born cells to the world
        for cell in newly_born_cells:
            if cell.id not in self.id_to_idx:
                self.add_cell(cell.id, dna=cell.nucleus['dna'], properties=cell.organelles)

        # Process death for cells with zero or less HP
        apoptosis_mask = (self.hp <= 0) & self.is_alive_mask
        self.is_alive_mask[apoptosis_mask] = False
        self.hp[apoptosis_mask] = 0.0

        dead_cell_indices = np.where(apoptosis_mask)[0]
        for dead_idx in dead_cell_indices:
            cell_id = self.cell_ids[dead_idx]
            self.event_logger.log('DEATH', self.time_step, cell_id=cell_id) # Log the death event
            if cell_id in self.materialized_cells:
                dead_cell = self.materialized_cells[cell_id]

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

    def get_population_summary(self) -> Dict[str, int]:
        """Returns a dictionary with the count of living cells for each label."""
        summary = {}
        living_indices = np.where(self.is_alive_mask)[0]
        if living_indices.size == 0:
            return {}

        living_labels = self.labels[living_indices]
        unique_labels, counts = np.unique(living_labels, return_counts=True)

        for label, count in zip(unique_labels, counts):
            if label:  # 鍮??쇰꺼?€ ?쒖쇅
                summary[label] = int(count)
        return summary

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
                details = {'concept_id': concept_id, 'hp_boost': energy_boost} # Renamed for clarity
                scopes = [concept_id]
                event = self.chronicle.record_event('stimulus_injected', details, scopes, self.branch_id, self.parent_event_id)
                self.parent_event_id = event['id']
            idx = self.id_to_idx[concept_id]
            if self.is_alive_mask[idx]:
                self.hp[idx] = min(self.max_hp[idx], self.hp[idx] + energy_boost)


    def print_world_summary(self):
        """Prints a detailed summary of the current world state for debugging."""
        self._sync_states_to_objects()
        print(f"\n--- World State (Time: {self.time_step}) ---")
        living_cell_count = np.sum(self.is_alive_mask)
        print(f"Living Cells: {living_cell_count}, Dead Cells (Archived): {len(self.graveyard)}")

        # This is inefficient for large worlds, but invaluable for debugging small scenarios.
        for i, cell_id in enumerate(self.cell_ids):
            if not self.is_alive_mask[i]:
                continue

            label = self.labels[i]
            hp = self.hp[i]
            max_hp = self.max_hp[i]
            ki = self.ki[i]
            max_ki = self.max_ki[i]
            mana = self.mana[i]
            max_mana = self.max_mana[i]
            hunger = self.hunger[i]
            age = self.age[i]
            strength = self.strength[i]
            agility = self.agility[i]
            intelligence = self.intelligence[i]
            vitality = self.vitality[i]
            wisdom = self.wisdom[i]


            status_parts = [
                f"<Cell: {label} ({cell_id})",
                f"HP: {hp:.1f}/{max_hp:.1f}",
                f"Ki: {ki:.1f}/{max_ki:.1f}",
                f"Mana: {mana:.1f}/{max_mana:.1f}",
                f"Hunger: {hunger:.1f}",
                f"Age: {age}",
                f"Stats: [S:{strength} A:{agility} I:{intelligence} V:{vitality} W:{wisdom}]",
                "Status: Alive>"
            ]
            print(f"  - {' | '.join(status_parts)}")

        print("-------------------------\n")



