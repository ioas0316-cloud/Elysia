
import random
import logging
from typing import List, Dict, Optional, Tuple, NamedTuple

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from .cell import Cell
from .chronicle import Chronicle
from .skills import MARTIAL_STYLES, Move
from .spells import SPELL_BOOK, cast_spell
from .world_event_logger import WorldEventLogger
from ..wave_mechanics import WaveMechanics
from .fields import FieldRegistry
from .dialogue_kr import get_line as kr_dialogue


class AwakeningEvent(NamedTuple):
    cell_id: str
    e_value: float
    r_value: int


class AwakeningEvent(NamedTuple):
    cell_id: str
    e_value: float
    r_value: int


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

        # --- Simulation Modes ---
        # peaceful_mode: when True, disables lethal combat and catastrophic weather
        # for baseline ecology/civilization survivability experiments.
        self.peaceful_mode: bool = False
        # macro_food_model_enabled: when True, enables background nourishment tuned
        # for long-horizon, macro-scale survival experiments (not used in unit tests).
        self.macro_food_model_enabled: bool = False
        # When True, WORLD will read macro_* attributes (war, surplus, unrest, etc.)
        # and emit soft, narrative-scale events (war eras, famine, bounty, disasters,
        # omens) as logs/flags. Default is False so unit tests remain untouched.
        self.enable_macro_disaster_events: bool = False

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
        self.is_awakened = np.array([], dtype=bool) # For the Law of Existential Change

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
        # Value/Will fields (scalar potentials; directions via gradients)
        self.value_mass_field = np.zeros((self.width, self.width), dtype=np.float32)
        self.will_field = np.zeros((self.width, self.width), dtype=np.float32)
        # Historical imprint, Norms, Prestige fields (skeletons)
        self.h_imprint = np.zeros((self.width, self.width), dtype=np.float32)
        self.norms_field = np.zeros((self.width, self.width), dtype=np.float32)
        self.prestige_field = np.zeros((self.width, self.width), dtype=np.float32)
        self._threat_decay = 0.92  # memory of threat (EMA)
        self._threat_sigma = 7.0   # spatial spread of threat influence (grid units)
        self._threat_gain = 1.0    # base contribution gain
        self._cohesion_gain = 0.08 # social cohesion gain toward allies
        # EM field dynamics and coupling (defaults preserve behavior: weights=0)
        self._em_decay = 0.92
        self._em_sigma = 6.0
        self._em_weight_E = 0.0  # soft bias along E (grad of salience)
        self._em_weight_B = 0.0  # soft bias along perp(E) (rotation cue)
        # H/N/P dynamics (defaults: passive; no coupling)
        self._h_decay = 0.95
        self._h_sigma = 8.0
        self._n_decay = 0.95
        self._n_sigma = 12.0
        self._p_decay = 0.95
        self._p_sigma = 10.0
        # Value/Will dynamics (defaults conservative)
        self._vm_decay = 0.95
        self._vm_sigma = 10.0
        self._will_decay = 0.95
        self._will_sigma = 10.0
        # Movement bias (defaults off)
        self._vm_weight_E = 0.0
        self._will_weight_E = 0.0

        # --- Coherence (mind-like soft measure) ---
        self.coherence_field = np.zeros((self.width, self.width), dtype=np.float32)
        self._coh_alpha = 0.6  # EWMA for temporal smoothing


        # --- Coherence (mind-like soft measure) ---
        self.coherence_field = np.zeros((self.width, self.width), dtype=np.float32)
        self._coh_alpha = 0.6  # EWMA for temporal smoothing
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
        self.fields.register_scalar(
            name="value_mass",
            scale="meso",
            array_getter=lambda: self.value_mass_field,
            grad_func=lambda arr, fx, fy: self._sample_field_grad(arr, fx, fy),
        )
        self.fields.register_scalar(
            name="will",
            scale="macro",
            array_getter=lambda: self.will_field,
            grad_func=lambda arr, fx, fy: self._sample_field_grad(arr, fx, fy),
        )
        self.fields.register_scalar(
            name="h_imprint",
            scale="meso",
            array_getter=lambda: self.h_imprint,
            grad_func=lambda arr, fx, fy: self._sample_field_grad(arr, fx, fy),
        )
        self.fields.register_scalar(
            name="norms",
            scale="meso",
            array_getter=lambda: self.norms_field,
            grad_func=lambda arr, fx, fy: self._sample_field_grad(arr, fx, fy),
        )
        self.fields.register_scalar(
            name="prestige",
            scale="meso",
            array_getter=lambda: self.prestige_field,
            grad_func=lambda arr, fx, fy: self._sample_field_grad(arr, fx, fy),
        )

        # Vector field for the Law of Intention
        self.intentional_field = np.zeros((self.width, self.width, 2), dtype=np.float32)
        self.fields.register_vector(
            name="intention",
            scale="macro",
            array_getter=lambda: self.intentional_field
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
        self.is_awakened = np.pad(self.is_awakened, (0, new_size - current_size), 'constant', constant_values=False)


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

    def get_world_snapshot(self) -> Dict[str, float]:
        """Return a coarse snapshot of the world's living state for diagnostics.

        This is designed to help external observers understand what is happening
        without inspecting every internal array.
        """
        # Time
        yl = self._year_length_ticks()
        approx_years = (self.time_step / float(yl)) if yl > 0 else 0.0

        alive = self.is_alive_mask
        labels = self.labels

        humans = (labels == 'human') & alive
        plants = (self.element_types == 'life') & alive
        animals = (self.element_types == 'animal') & alive

        human_count = int(humans.sum())
        plant_count = int(plants.sum())
        animal_count = int(animals.sum())

        def _avg(arr, mask) -> float:
            if mask.sum() == 0:
                return 0.0
            return float(arr[mask].mean())

        snapshot: Dict[str, float] = {
            "time_step": float(self.time_step),
            "approx_years": float(approx_years),
            "humans": float(human_count),
            "animals": float(animal_count),
            "plants": float(plant_count),
            "avg_human_hunger": _avg(self.hunger, humans),
            "avg_human_hydration": _avg(self.hydration, humans),
            "avg_human_age_ticks": _avg(self.age, humans),
        }

        # Very coarse ecology hints
        if plant_count == 0 and human_count > 0:
            snapshot["hint_no_plants"] = 1.0
        if human_count == 0 and (animal_count > 0 or plant_count > 0):
            snapshot["hint_no_humans"] = 1.0

        return snapshot

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
        self.is_awakened[idx] = False
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
                # Shorten human lifespan in simulation years so that
                # generational and civilization-scale change is visible
                # within a reasonable number of ticks.
                return random.randint(25, 40)
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

        # Optional age override in "years" for initial seeding (e.g., start as 16살 성인).
        age_years = properties.get('age_years')
        if age_years is not None:
            try:
                age_years_f = float(age_years)
                self.age[idx] = max(0, int(age_years_f * self._year_length_ticks()))
            except (TypeError, ValueError):
                # Fallback: keep default 0 if parsing fails.
                pass

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


    def run_simulation_step(self) -> Tuple[List[Cell], List[AwakeningEvent]]:
        if self.chronicle:
            event = self.chronicle.record_event('simulation_step_run', {}, [], self.branch_id, self.parent_event_id)
            self.parent_event_id = event['id']
        self.time_step += 1

        if len(self.cell_ids) == 0:
            return [], []

        # --- Experience snapshot (HP delta from previous step) ---
        try:
            if not hasattr(self, '_prev_hp') or self._prev_hp.shape != self.hp.shape:
                self._prev_hp = self.hp.copy()
            hp_delta = (self.hp - self._prev_hp).astype(np.float32)
            self._last_hp_delta = hp_delta
            # Log coarse experience delta if significant
            thr = 5.0
            pos = hp_delta[hp_delta > thr]
            neg = hp_delta[hp_delta < -thr]
            if pos.size or neg.size:
                self.event_logger.log('EXPERIENCE_DELTA', self.time_step,
                                      total_pos=float(pos.sum()) if pos.size else 0.0,
                                      total_neg=float(neg.sum()) if neg.size else 0.0,
                                      count_pos=int(pos.size), count_neg=int(neg.size))
        except Exception:
            pass

        # Update celestial cycle & weather
        self._update_celestial_fields()
        self._update_weather()

        # Update emergent continuous fields (e.g., threat)
        self._update_emergent_fields()
        self._update_hydration_field()
        self._update_em_perception_field()
        self._update_historical_imprint_field()
        self._update_norms_field()
        self._update_prestige_field()
        self._update_value_will_fields()
        self._update_intentional_field()

        # Update passive resources (MP regen, hunger, starvation)
        self._update_passive_resources()

        # Macro-scale narrative/disaster hooks (war/famine/bounty/plague/storm/omens).
        # These are soft, opt-in overlays driven by macro_* attributes and do not
        # run unless enable_macro_disaster_events is True.
        self._apply_macro_disaster_events()

        # Process major state changes and actions
        newly_born_cells = []
        self._process_animal_actions()
        newly_born_cells.extend(self._process_life_cycles())

        # Apply final physics and cleanup
        self._apply_physics_and_cleanup(newly_born_cells)

        # --- Law of Existential Change (e > r) ---
        awakening_events = self._apply_law_of_awakening()

        # Prepare next-step snapshot
        try:
            self._prev_hp = self.hp.copy()
        except Exception:
            pass

        return newly_born_cells, awakening_events

    def _apply_macro_disaster_events(self) -> None:
        """
        Soft macro-level events (war eras, famine/bounty, disasters, omens).

        - Reads macro_* attributes attached via scripts.world_macro_bridge.
        - Emits coarse events via event_logger and toggles a few boolean flags.
        - Does NOT directly kill units; any concrete effects should be mediated
          through higher-level laws/fields.

        This function is opt-in: it only runs when enable_macro_disaster_events
        is True. This keeps unit tests and small sandboxes free of surprise
        catastrophes.
        """
        if not getattr(self, "enable_macro_disaster_events", False):
            return

        try:
            war = float(getattr(self, "macro_war_pressure", 0.0))
            unrest = float(getattr(self, "macro_unrest", 0.0))
            power = float(getattr(self, "macro_power_concentration", 0.0))
            surplus = float(getattr(self, "macro_surplus_years", 0.0))
            pop = float(getattr(self, "macro_population", 0.0))
            monster = float(getattr(self, "macro_monster_threat", 0.0))
            trade = float(getattr(self, "macro_trade_index", 0.0))
            literacy = float(getattr(self, "macro_literacy", 0.0))
            culture = float(getattr(self, "macro_culture_index", 0.0))
        except Exception:
            # If macro attributes are not present, skip gracefully.
            return

        # --- War / Civil War era flags --------------------------------------
        prev_war_state = getattr(self, "_macro_war_state", "peace")
        war_state = "peace"
        if war > 0.6:
            war_state = "war"
        if war > 0.7 and unrest > 0.4 and power > 0.5:
            war_state = "civil_war"

        if war_state != prev_war_state:
            self._macro_war_state = war_state
            if war_state == "peace":
                self.event_logger.log(
                    "ERA_PEACE",
                    self.time_step,
                    war=war,
                    unrest=unrest,
                    power_concentration=power,
                )
            elif war_state == "war":
                self.event_logger.log(
                    "ERA_WAR",
                    self.time_step,
                    war=war,
                    unrest=unrest,
                    power_concentration=power,
                )
            elif war_state == "civil_war":
                self.event_logger.log(
                    "ERA_CIVIL_WAR",
                    self.time_step,
                    war=war,
                    unrest=unrest,
                    power_concentration=power,
                )

        # --- Famine / Bountiful harvest eras --------------------------------
        prev_famine = bool(getattr(self, "_macro_famine_active", False))
        famine_active = (surplus < 0.3) and (pop > 0.0)
        if famine_active != prev_famine:
            self._macro_famine_active = famine_active
            if famine_active:
                self.event_logger.log(
                    "ERA_FAMINE_START",
                    self.time_step,
                    surplus_years=surplus,
                    population=pop,
                )
            else:
                self.event_logger.log(
                    "ERA_FAMINE_END",
                    self.time_step,
                    surplus_years=surplus,
                    population=pop,
                )

        prev_bounty = bool(getattr(self, "_macro_bounty_active", False))
        bounty_active = (surplus > 2.5) and (not famine_active) and (pop > 0.0)
        if bounty_active != prev_bounty:
            self._macro_bounty_active = bounty_active
            if bounty_active:
                self.event_logger.log(
                    "ERA_BOUNTIFUL_HARVEST_START",
                    self.time_step,
                    surplus_years=surplus,
                    population=pop,
                )
            else:
                self.event_logger.log(
                    "ERA_BOUNTIFUL_HARVEST_END",
                    self.time_step,
                    surplus_years=surplus,
                    population=pop,
                )

        # --- Hydrology-driven disasters (floods / tempests) -----------------
        try:
            mean_wetness = float(self.wetness.mean()) if self.wetness.size > 0 else 0.0
        except Exception:
            mean_wetness = 0.0

        prev_flood = bool(getattr(self, "_macro_flood_active", False))
        flood_active = (mean_wetness > 0.75) and (float(self.cloud_cover) > 0.7)
        if flood_active != prev_flood:
            self._macro_flood_active = flood_active
            if flood_active:
                self.event_logger.log(
                    "DISASTER_FLOOD",
                    self.time_step,
                    mean_wetness=mean_wetness,
                    cloud=float(self.cloud_cover),
                )
            else:
                self.event_logger.log(
                    "DISASTER_FLOOD_RECEDES",
                    self.time_step,
                    mean_wetness=mean_wetness,
                    cloud=float(self.cloud_cover),
                )

        prev_storm = bool(getattr(self, "_macro_storm_active", False))
        storm_active = (float(self.cloud_cover) > 0.85) and (float(self.humidity) > 0.8)
        if storm_active != prev_storm:
            self._macro_storm_active = storm_active
            if storm_active:
                self.event_logger.log(
                    "DISASTER_TEMPEST",
                    self.time_step,
                    cloud=float(self.cloud_cover),
                    humidity=float(self.humidity),
                )
            else:
                self.event_logger.log(
                    "DISASTER_TEMPEST_END",
                    self.time_step,
                    cloud=float(self.cloud_cover),
                    humidity=float(self.humidity),
                )

        # --- Plague pressure (trade + population + humidity/wetness) --------
        prev_plague = bool(getattr(self, "_macro_plague_active", False))
        plague_pressure = trade * (pop / (pop + 1000.0)) * (0.5 + mean_wetness)
        plague_active = plague_pressure > 1.0
        if plague_active != prev_plague:
            self._macro_plague_active = plague_active
            if plague_active:
                self.event_logger.log(
                    "DISASTER_PLAGUE_OUTBREAK",
                    self.time_step,
                    trade_index=trade,
                    population=pop,
                    mean_wetness=mean_wetness,
                )
            else:
                self.event_logger.log(
                    "DISASTER_PLAGUE_RECOVER",
                    self.time_step,
                    trade_index=trade,
                    population=pop,
                    mean_wetness=mean_wetness,
                )

        # --- Demonic / angelic omens (avatars, demon lord, etc.) ------------
        # Demonic pressure: war + unrest + monster threat (0..3+), softly squashed.
        demonic_index = max(0.0, war + unrest + monster)
        demonic_norm = min(1.0, demonic_index / 2.5)

        # Angelic pressure: literacy + culture (0..2), softly squashed.
        angelic_index = max(0.0, literacy + culture)
        angelic_norm = min(1.0, angelic_index / 2.0)

        if demonic_norm > 0.8 and not getattr(self, "_macro_demon_omen_emitted", False):
            self._macro_demon_omen_emitted = True
            self.event_logger.log(
                "OMEN_DEMON_LORD",
                self.time_step,
                demonic_index=demonic_norm,
            )
            # Soft manifestation: spawn a single demon-lord avatar cell if not present.
            try:
                if "DemonLord" not in self.id_to_idx:
                    cx = float(self.width // 2)
                    cy = float(self.width // 2)
                    self.add_cell(
                        "DemonLord",
                        properties={
                            "label": "마왕",
                            "element_type": "animal",
                            "culture": "demon",
                            "continent": "Deep",
                            "strength": 60,
                            "vitality": 80,
                            "wisdom": 40,
                            "position": {"x": cx, "y": cy, "z": 0.0},
                        },
                        _record_event=False,
                    )
            except Exception:
                # World must not break even if spawning fails.
                pass

        if angelic_norm > 0.8 and not getattr(self, "_macro_angel_omen_emitted", False):
            self._macro_angel_omen_emitted = True
            self.event_logger.log(
                "OMEN_ANGEL_AVATAR",
                self.time_step,
                angelic_index=angelic_norm,
            )
            # Soft manifestation: spawn a single angelic avatar cell if not present.
            try:
                if "AngelAvatar" not in self.id_to_idx:
                    cx = float(self.width // 2)
                    cy = float(self.width // 2) - 10.0
                    self.add_cell(
                        "AngelAvatar",
                        properties={
                            "label": "천사",
                            "element_type": "animal",
                            "culture": "knight",
                            "continent": "Sky",
                            "strength": 40,
                            "vitality": 50,
                            "wisdom": 70,
                            "position": {"x": cx, "y": cy, "z": 0.0},
                        },
                        _record_event=False,
                    )
            except Exception:
                pass

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
        # In peaceful_mode, skip catastrophic lightning strikes to focus on ecology.
        if (not self.peaceful_mode) and self.cloud_cover > 0.8 and self.humidity > 0.7 and random.random() < 0.1:
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
                    self.event_logger.log('LIGHTNING_STRIKE', self.time_step, cell_id=cell_id, damage=damage, x=x, y=y)
                    # Historical imprint at strike location (soft)
                    self._imprint_gaussian(self.h_imprint, x, y, sigma=self._h_sigma, amplitude=1.0)
        
        
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
        """Updates passive resource changes for all living cells (Ki, Mana, Hunger, HP).

        NOTE: Rates are intentionally gentle so that, at typical time scales
        (10–60 minutes per tick), cells can survive 여러 날 without eating or
        drinking, allowing civilization patterns to emerge before total collapse.
        """
        # --- Ki Regeneration (very slow) ---
        ki_regen_mask = self.is_alive_mask & (self.ki < self.max_ki)
        self.ki[ki_regen_mask] = np.minimum(self.max_ki[ki_regen_mask], self.ki[ki_regen_mask] + 0.1)

        # --- Mana Regeneration (moderate) ---
        mana_regen_mask = self.is_alive_mask & (self.mana < self.max_mana)
        self.mana[mana_regen_mask] = np.minimum(self.max_mana[mana_regen_mask], self.mana[mana_regen_mask] + 1.0)

        # --- Faith does not regenerate passively ---

        # --- Hunger Depletion ---
        # All living things get hungrier over time.
        if self.peaceful_mode:
            # In peaceful ecology tests, hunger still moves but more slowly.
            self.hunger[self.is_alive_mask] = np.maximum(0, self.hunger[self.is_alive_mask] - 0.05)
        else:
            # Default law for production worlds (aligned with unit tests).
            # Hunger depletion is 0.5 per step so that a cell starting at 1.0
            # reaches 0 after two steps in non‑peaceful mode.
            self.hunger[self.is_alive_mask] = np.maximum(0, self.hunger[self.is_alive_mask] - 0.5)

        # --- Starvation ---
        # If hunger is 0, the cell starts losing HP.
        if not self.peaceful_mode:
            starvation_mask = self.is_alive_mask & (self.hunger <= 0)
            # Starvation damage tuned to 2.0 HP per step to match
            # expectations in world_simulation tests while remaining
            # a soft field effect at simulation time scales.
            self.hp[starvation_mask] -= 2.0

        # --- Hydration Depletion ---
        if self.peaceful_mode:
            self.hydration[self.is_alive_mask] = np.maximum(0, self.hydration[self.is_alive_mask] - 0.1)
        else:
            # Water loss is still faster than food, but reduced so that cells do not
            # instantly die in sparse-resource worlds.
            self.hydration[self.is_alive_mask] = np.maximum(0, self.hydration[self.is_alive_mask] - 0.3)

        # --- Dehydration ---
        if not self.peaceful_mode:
            dehydration_mask = self.is_alive_mask & (self.hydration <= 0)
            self.hp[dehydration_mask] -= 1.0  # gentler penalty for dehydration

        # --- Macro-scale nourishment (optional) ---
        # Used only when macro_food_model_enabled is True to approximate a world
        # where food/water capacity is safely above population demand. This keeps
        # long-horizon tests from collapsing instantly without modeling full
        # agriculture/economy details.
        if getattr(self, "macro_food_model_enabled", False):
            alive_mask = self.is_alive_mask
            if np.any(alive_mask):
                # Gently push hunger/hydration toward a comfortable band (around 70)
                target = 70.0
                h = self.hunger[alive_mask]
                d = self.hydration[alive_mask]
                self.hunger[alive_mask] = np.minimum(
                    100.0,
                    h + 0.5 * np.clip((target - h) / 100.0, 0.0, 1.0),
                )
                self.hydration[alive_mask] = np.minimum(
                    100.0,
                    d + 0.5 * np.clip((target - d) / 100.0, 0.0, 1.0),
                )

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
            # Historical imprint at place of death
            dx = int(self.positions[idx, 0]) % self.width
            dy = int(self.positions[idx, 1]) % self.width
            self._imprint_gaussian(self.h_imprint, dx, dy, sigma=self._h_sigma, amplitude=0.6)

    def _update_emergent_fields(self):
        """Updates soft fields (e.g., threat) from distributed sources.
        The field is not a command; it is a context carrier that agents can sense.
        """
        if len(self.cell_ids) == 0:
            return

        # Read macro-scale modifiers (default to 0 if not set).
        macro_war = float(getattr(self, "macro_war_pressure", 0.0))
        macro_mon = float(getattr(self, "macro_monster_threat", 0.0))
        macro_unrest = float(getattr(self, "macro_unrest", 0.0))
        # Soft scaling factor: higher war/monster/unrest strengthens threat imprint,
        # but stays within a gentle range (~0.5x..2x).
        threat_scale = 1.0 + 0.5 * macro_war + 0.5 * macro_mon + 0.3 * macro_unrest
        threat_scale = max(0.5, min(2.0, threat_scale))

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
            # Apply macro-scale threat scaling as a soft law.
            self.threat_field = (self._threat_decay * self.threat_field) + ((1.0 - self._threat_decay) * (new_threat * threat_scale))
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

    def _update_value_will_fields(self):
        """Update value_mass_field and will_field from recent experience signals (HP delta).
        Positive HP deltas reinforce value_mass; negative deltas reinforce will (tension/striving).
        """
        try:
            if not hasattr(self, '_last_hp_delta') or self._last_hp_delta.shape != self.hp.shape:
                return
            vm_src = np.zeros_like(self.value_mass_field)
            will_src = np.zeros_like(self.will_field)
            idxs = np.where(self.is_alive_mask)[0]
            if idxs.size > 0:
                thr = 2.0
                for i in idxs:
                    d = float(self._last_hp_delta[i])
                    if abs(d) < thr:
                        continue
                    x = int(self.positions[i][0]) % self.width
                    y = int(self.positions[i][1]) % self.width
                    if d > 0:
                        vm_src[y, x] += d
                    else:
                        will_src[y, x] += (-d)
            # Diffuse + normalize
            try:
                from scipy.ndimage import gaussian_filter
                vm = gaussian_filter(vm_src, sigma=self._vm_sigma)
                wl = gaussian_filter(will_src, sigma=self._will_sigma)
            except Exception:
                vm, wl = vm_src, will_src
            if vm.max() > 0:
                vm = vm / float(vm.max())
            if wl.max() > 0:
                wl = wl / float(wl.max())
            self.value_mass_field = (self._vm_decay * self.value_mass_field) + ((1.0 - self._vm_decay) * vm.astype(np.float32))
            self.will_field = (self._will_decay * self.will_field) + ((1.0 - self._will_decay) * wl.astype(np.float32))
        except Exception:
            pass

    def _update_intentional_field(self):
        """
        Calculates the gradient of the value_mass_field to create the intentional_field.
        This field directs entities towards areas of higher meaning.
        """
        try:
            # The gradient returns derivatives along each axis (dy, dx)
            grad_y, grad_x = np.gradient(self.value_mass_field)

            # Combine into a (width, width, 2) vector field
            self.intentional_field[..., 0] = grad_x
            self.intentional_field[..., 1] = grad_y

            # Normalize the field to prevent extreme forces, but preserve direction
            magnitudes = np.sqrt(grad_x**2 + grad_y**2) + 1e-9 # Add epsilon to avoid division by zero
            max_mag = np.max(magnitudes)
            if max_mag > 0:
                self.intentional_field /= max_mag

        except Exception as e:
            self.logger.error(f"Error updating intentional field: {e}", exc_info=True)

    def _update_coherence_field(self) -> None:
        """Lightweight coherence map from value/will gradient alignment (0..1).

        Coherence intuition: when the spatial directions of value and will fields align,
        and their magnitudes are non-trivial, we mark high coherence. This is a proxy for
        "여러 신호가 같은 문장으로 정렬되는 순간".
        """
        try:
            vm = np.asarray(self.value_mass_field, dtype=np.float32)
            wl = np.asarray(self.will_field, dtype=np.float32)
            if vm.size == 0 or wl.size == 0:
                return
            gv_y, gv_x = np.gradient(vm)
            gw_y, gw_x = np.gradient(wl)
            mv = np.sqrt(gv_x*gv_x + gv_y*gv_y) + 1e-6
            mw = np.sqrt(gw_x*gw_x + gw_y*gw_y) + 1e-6
            ux_v, uy_v = gv_x/mv, gv_y/mv
            ux_w, uy_w = gw_x/mw, gw_y/mw
            cosang = (ux_v*ux_w + uy_v*uy_w)
            cos01 = (cosang + 1.0) * 0.5
            mag = np.tanh((mv + mw) * 0.25)
            coh = np.clip(cos01 * mag, 0.0, 1.0).astype(np.float32)
            self.coherence_field = (self._coh_alpha * self.coherence_field) + ((1.0 - self._coh_alpha) * coh)
        except Exception:
            pass
    def _imprint_gaussian(self, target: np.ndarray, x: int, y: int, sigma: float, amplitude: float = 1.0):
        try:
            rad = int(max(2, sigma * 3))
            x0, x1 = max(0, x - rad), min(self.width, x + rad + 1)
            y0, y1 = max(0, y - rad), min(self.width, y + rad + 1)
            xs = np.arange(x0, x1) - x
            ys = np.arange(y0, y1) - y
            gx = np.exp(-(xs**2) / (2 * sigma * sigma))
            gy = np.exp(-(ys**2) / (2 * sigma * sigma))
            patch = (gy[:, None] * gx[None, :]).astype(np.float32)
            target[y0:y1, x0:x1] += amplitude * patch
        except Exception:
            pass

    def _update_historical_imprint_field(self):
        """Decay-only tick; event handlers add local imprints when events occur."""
        try:
            self.h_imprint *= self._h_decay
            # optional normalization to keep in [0,1]
            m = float(self.h_imprint.max())
            if m > 0:
                self.h_imprint = (self.h_imprint / m).astype(np.float32)
        except Exception:
            pass

    def _update_norms_field(self):
        """Skeleton: diffuse alive presence as proxy for norms cohesion (to be refined)."""
        try:
            presence = np.zeros_like(self.norms_field)
            alive_idx = np.where(self.is_alive_mask)[0]
            if alive_idx.size > 0:
                px = np.clip(self.positions[alive_idx, 0].astype(np.int32), 0, self.width - 1)
                py = np.clip(self.positions[alive_idx, 1].astype(np.int32), 0, self.width - 1)
                presence[py, px] += 1.0
            try:
                from scipy.ndimage import gaussian_filter
                diff = gaussian_filter(presence, sigma=self._n_sigma)
            except Exception:
                diff = presence
            if diff.max() > 0:
                diff = diff / float(diff.max())
            self.norms_field = (self._n_decay * self.norms_field) + ((1.0 - self._n_decay) * diff.astype(np.float32))
        except Exception:
            pass

    def _update_prestige_field(self):
        """Skeleton: prestige from local connectivity/strength; diffuse+decay."""
        try:
            prestige_src = np.zeros_like(self.prestige_field)
            alive_idx = np.where(self.is_alive_mask)[0]
            if alive_idx.size > 0:
                # degree via adjacency
                adj = self.adjacency_matrix.tocsr()
                deg = np.array(adj.getnnz(axis=1)).reshape(-1)
                score = np.sqrt(deg + 1.0) + (self.strength.astype(np.float32) / 100.0)
                score = np.clip(score, 0.0, None)
                px = np.clip(self.positions[alive_idx, 0].astype(np.int32), 0, self.width - 1)
                py = np.clip(self.positions[alive_idx, 1].astype(np.int32), 0, self.width - 1)
                for idx in alive_idx:
                    prestige_src[int(self.positions[idx,1]) % self.width, int(self.positions[idx,0]) % self.width] += float(score[idx])
            try:
                from scipy.ndimage import gaussian_filter
                diff = gaussian_filter(prestige_src, sigma=self._p_sigma)
            except Exception:
                diff = prestige_src
            if diff.max() > 0:
                diff = diff / float(diff.max())
            self.prestige_field = (self._p_decay * self.prestige_field) + ((1.0 - self._p_decay) * diff.astype(np.float32))
        except Exception:
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
                vm_bias = np.zeros(3, dtype=np.float32)
                will_bias = np.zeros(3, dtype=np.float32)
                intention_force = np.zeros(3, dtype=np.float32)

                # --- The Law of Intention: Follow the gradient of meaning ---
                wisdom_factor = self.wisdom[i] / 50.0 # Normalize wisdom, 50 is a high value
                if wisdom_factor > 0:
                    ix, iy = self.fields.vector("intention", px, py)
                    intention_force = np.array([ix, iy, 0.0], dtype=np.float32) * wisdom_factor * 0.2 # 0.2 is a weighting factor

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
                # --- Value Mass / Will Bias (soft, defaults off) ---
                if self._vm_weight_E != 0.0:
                    vx, vy = self.fields.grad("value_mass", px, py)
                    vm_bias = np.array([vx, vy, 0.0], dtype=np.float32) * self._vm_weight_E
                if self._will_weight_E != 0.0:
                    wx_, wy_ = self.fields.grad("will", px, py)
                    will_bias = np.array([wx_, wy_, 0.0], dtype=np.float32) * self._will_weight_E

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

                other_forces = avoid + coh + kin_attraction + hydration_seeking + em_bias + vm_bias + will_bias
                movement_vectors[local_idx] += other_forces + intention_force

                # --- Observe and Record Meaningful Choice ---
                if np.linalg.norm(intention_force) > np.linalg.norm(other_forces):
                    self.logger.info(f"CHOICE: Cell '{self.cell_ids[i]}' made a meaningful choice, driven by intention.")
                    self.event_logger.log('INTENTION_DRIVEN_ACTION', self.time_step, cell_id=self.cell_ids[i])

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
            return None  # No one nearby to interact with

        # Determine if hungry
        is_hungry = self.hunger[actor_idx] < 60
        kin_indices = set(connected_indices[adj_matrix_csr[actor_idx, connected_indices].toarray().flatten() >= 0.8])

        # Trinity forces (Body/Soul/Spirit) for this actor – used as soft biases.
        body_p, soul_p, spirit_p = self.get_trinity_for_actor(actor_idx)

        # Find potential targets based on hunger and diet
        potential_targets = []
        actor_diet = self.diets[actor_idx]

        if is_hungry:
            # Find food: non-kin animals for carnivores, plants for herbivores
            # In peaceful_mode, skip hunting other animals and rely on plants.
            if not self.peaceful_mode and actor_diet in ['carnivore', 'omnivore']:
                mask = (self.element_types[connected_indices] == 'animal') & self.is_alive_mask[connected_indices]
                non_kin_prey = [p_idx for p_idx in connected_indices[mask] if p_idx not in kin_indices]
                potential_targets.extend(non_kin_prey)
            if actor_diet in ['herbivore', 'omnivore']:
                mask = (self.element_types[connected_indices] == 'life') & self.is_alive_mask[connected_indices]
                potential_targets.extend(connected_indices[mask])
        else:
            # If not hungry, might pick a fight with non-kin rivals
            if not self.peaceful_mode:
                mask = (self.element_types[connected_indices] == 'animal') & self.is_alive_mask[connected_indices]
                non_kin_rivals = [r_idx for r_idx in connected_indices[mask] if r_idx not in kin_indices]
                potential_targets.extend(non_kin_rivals)

        if not potential_targets:
            # When there are no obvious targets and body is not under strong pressure,
            # a stronger spirit can bias the actor toward non-instrumental behaviour.
            if not is_hungry and body_p < 0.3 and spirit_p > 0.5 and np.random.random() < spirit_p * 0.4:
                return None, 'idle', None
            return None  # No suitable targets found

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
        # In peaceful_mode, suppress explicitly lethal combat actions against animals.
        if self.peaceful_mode and action in ('attack', 'cast_firebolt') and target_idx >= 0:
            try:
                if self.element_types[target_idx] == 'animal':
                    return
            except Exception:
                # If anything goes wrong, fall back to normal handling.
                pass

        # --- Handle non-target actions ---
        if action == 'idle':
            # Occasionally treat idle as playful, non-instrumental behavior.
            if random.random() < 0.1:
                pos = self.positions[actor_idx]
                self.event_logger.log(
                    'IDLE_PLAY',
                    self.time_step,
                    cell_id=self.cell_ids[actor_idx],
                    x=float(pos[0]),
                    y=float(pos[1]),
                )
                self._speak(actor_idx, "IDLE_PLAY")
            return

        if action == 'drink':
            self.hydration[actor_idx] = min(100, self.hydration[actor_idx] + 50)
            self.logger.info(f"ACTION: '{self.cell_ids[actor_idx]}' drinks water.")
            self.event_logger.log('DRINK', self.time_step, cell_id=self.cell_ids[actor_idx])
            self._speak(actor_idx, "DRINK")
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

        # Compute distance once for melee / proximity checks.
        dist = float(np.linalg.norm(self.positions[actor_idx] - self.positions[target_idx]))
        if dist < 1.5:
            if action == 'eat' and self.element_types[target_idx] == 'life':
                self.logger.info(f"ACTION: '{self.cell_ids[actor_idx]}' eats '{self.cell_ids[target_idx]}'.")
                self.event_logger.log('EAT', self.time_step, actor_id=self.cell_ids[actor_idx], target_id=self.cell_ids[target_idx])
                self.hp[target_idx] = 0  # Eating kills the plant
                food_value = 20
                self.hunger[actor_idx] = min(100, self.hunger[actor_idx] + food_value)
                self._speak(actor_idx, "EAT_PLANT")
                return

            damage_multiplier = 1.0
            if move:
                self.logger.info(f"SKILL: '{self.cell_ids[actor_idx]}' uses [{move.name}] on '{self.cell_ids[target_idx]}'.")
                self.ki[actor_idx] -= move.ki_cost
                damage_multiplier = move.apply_effect(self, actor_idx, target_idx, self.hp)
                self._speak(actor_idx, "SKILL_ATTACK")
            elif action == 'attack':
                self.logger.info(f"ACTION: '{self.cell_ids[actor_idx]}' attacks '{self.cell_ids[target_idx]}'.")
                self._speak(actor_idx, "ATTACK")
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
                        self._speak(actor_idx, "SPELL_FIRE")
                    return
            elif action == 'cast_heal':
                if self.spells.get('heal'):
                    result = cast_spell(self, 'heal', actor_idx, None)
                    healed = float(result.get('heal', 0.0))
                    if healed > 0:
                        self.logger.info(f"SPELL: '{self.cell_ids[actor_idx]}' casts Heal (+{healed:.1f}).")
                        self.event_logger.log('SPELL', self.time_step, caster_id=self.cell_ids[actor_idx], spell='heal', heal=healed)
                        self._speak(actor_idx, "SPELL_HEAL")

                        # --- First Ignition: The Birth of Meaning (E) ---
                        # A healing act creates positive meaning energy in the world.
                        delta_e_local = healed * 0.1 # The amount of meaning is proportional to the heal amount
                        actor_pos = self.positions[actor_idx]
                        x, y = int(actor_pos[0]) % self.width, int(actor_pos[1]) % self.width
                        self._imprint_gaussian(self.value_mass_field, x, y, sigma=self._vm_sigma, amplitude=delta_e_local)
                        self.logger.info(f"SPIRIT: A healing act created delta_E_local={delta_e_local:.2f} at ({x}, {y}).")
                        self.event_logger.log('MEANING_CREATED', self.time_step, type='healing', magnitude=delta_e_local, x=x, y=y)

                    return

            # Evasion chance based on target agility and distance.
            # At true grappling range (very close), dodging is effectively impossible.
            evade_chance = min(0.4, float(self.agility[target_idx]) / 100.0)
            if dist < 0.5:
                evade_chance = 0.0
            elif dist < 1.0:
                evade_chance *= 0.5
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

    def _apply_law_of_awakening(self) -> List[AwakeningEvent]:
        """
        Applies the Law of Existential Change (e > r) and returns a list of awakening events.
        This is a core physical law of the world.
        """
        events = []
        alive_indices = np.where(self.is_alive_mask)[0]
        if alive_indices.size == 0:
            return events

        e = self.insight[alive_indices] * 100
        r = self.age[alive_indices]

        awakening_mask = (e > r) & ~self.is_awakened[alive_indices]
        awakened_indices = alive_indices[awakening_mask]

        if awakened_indices.size > 0:
            for idx in awakened_indices:
                e_val = e[np.where(alive_indices==idx)[0][0]]
                r_val = r[np.where(alive_indices==idx)[0][0]]

                event = AwakeningEvent(
                    cell_id=self.cell_ids[idx],
                    e_value=e_val,
                    r_value=r_val
                )
                events.append(event)
                self.logger.info(f"LAW OF CHANGE: Cell '{event.cell_id}' has met conditions for awakening! (e={event.e_value:.2f} > r={event.r_value})")
                self.event_logger.log('AWAKENING_EVENT', self.time_step, cell_id=event.cell_id)

            # Enact physical consequences
            self.is_awakened[awakened_indices] = True # Mark as awakened to prevent immediate re-awakening
            self.age[awakened_indices] = 0
            self.insight[awakened_indices] = 0

        return events


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

    def get_pfe_for_actor(self, actor_idx: int) -> Tuple[float, float, float]:
        """Compute coarse P/F/E scalars (Body/Soul/Spirit) for a single actor.

        P (inertia of the past / Body): normalized age (0..1).
        F (force of the present / Soul): aggregated need pressure (hunger + hydration) in 0..1.
        E (meaning energy for the future / Spirit): local value_mass_field sample at the actor position.
        """
        if actor_idx < 0 or actor_idx >= self.age.size:
            raise IndexError(f"actor_idx out of range: {actor_idx}")

        age = float(self.age[actor_idx]) if self.age.size > 0 else 0.0
        max_age = float(self.max_age[actor_idx]) if self.max_age.size > 0 and self.max_age[actor_idx] > 0 else 1.0
        p_inertia = max(0.0, min(1.0, age / max_age))

        hunger_val = float(self.hunger[actor_idx]) if self.hunger.size > 0 else 100.0
        hydration_val = float(self.hydration[actor_idx]) if self.hydration.size > 0 else 100.0
        hunger_deficit = max(0.0, (70.0 - hunger_val) / 70.0)
        hydration_deficit = max(0.0, (70.0 - hydration_val) / 70.0)
        f_force = max(0.0, min(1.0, hunger_deficit + hydration_deficit))

        if self.positions.size > 0 and self.value_mass_field.size > 0:
            pos = self.positions[actor_idx]
            x = int(np.clip(pos[0], 0, self.width - 1))
            y = int(np.clip(pos[1], 0, self.width - 1))
            e_energy = float(self.value_mass_field[y, x])
        else:
            e_energy = 0.0

        return p_inertia, f_force, e_energy

    def _speak(self, actor_idx: int, key: str, **kwargs) -> None:
        """Emit a Korean utterance for the given actor and key, if available."""
        if actor_idx < 0 or actor_idx >= len(self.cell_ids):
            return
        text = kr_dialogue(key, **kwargs)
        if not text:
            return
        cell_id = self.cell_ids[actor_idx]
        # Log as a SAY event and also via the logger for visibility.
        self.logger.info(f"SAY: '{cell_id}' {text}")
        self.event_logger.log('SAY', self.time_step, cell_id=cell_id, text=text)

    def get_trinity_for_actor(self, actor_idx: int) -> Tuple[float, float, float]:
        """Compute Body/Soul/Spirit pressure scalars (0..1) for a single actor.

        Body  : physical survival pressure (HP, hunger/hydration, local threat).
        Soul  : relational/identity pressure (degree/connection_counts, culture/affiliation).
        Spirit: meaning/purpose pressure (local value_mass / will / coherence if available).
        """
        if actor_idx < 0 or actor_idx >= self.age.size:
            raise IndexError(f"actor_idx out of range: {actor_idx}")

        # Body pressure: low HP, high deficits, or high threat increase this term.
        hp = float(self.hp[actor_idx]) if self.hp.size > 0 else 100.0
        max_hp = float(self.max_hp[actor_idx]) if self.max_hp.size > 0 and self.max_hp[actor_idx] > 0 else 100.0
        hp_deficit = max(0.0, 1.0 - (hp / max_hp))
        hunger_val = float(self.hunger[actor_idx]) if self.hunger.size > 0 else 100.0
        hydration_val = float(self.hydration[actor_idx]) if self.hydration.size > 0 else 100.0
        hunger_deficit = max(0.0, (70.0 - hunger_val) / 70.0)
        hydration_deficit = max(0.0, (70.0 - hydration_val) / 70.0)

        local_threat = 0.0
        if self.positions.size > 0 and self.threat_field.size > 0:
            pos = self.positions[actor_idx]
            tx = int(np.clip(pos[0], 0, self.width - 1))
            ty = int(np.clip(pos[1], 0, self.width - 1))
            local_threat = float(self.threat_field[ty, tx])

        body_pressure = hp_deficit + hunger_deficit + hydration_deficit + (local_threat * 0.5)
        body_pressure = max(0.0, min(1.0, body_pressure))

        # Soul pressure: how "entangled" this actor is with others (degree, simple culture/affiliation presence).
        degree = float(self.connection_counts[actor_idx]) if self.connection_counts.size > 0 else 0.0
        soul_from_degree = 1.0 - 1.0 / (1.0 + degree)  # saturating 0..1
        has_affiliation = 1.0 if (self.affiliation.size > 0 and bool(self.affiliation[actor_idx])) else 0.0
        has_culture = 1.0 if (self.culture.size > 0 and bool(self.culture[actor_idx])) else 0.0
        soul_pressure = soul_from_degree * 0.6 + (has_affiliation + has_culture) * 0.2
        soul_pressure = max(0.0, min(1.0, soul_pressure))

        # Spirit pressure: strength of meaning/purpose field at this location.
        spirit_pressure = 0.0
        if self.positions.size > 0 and self.value_mass_field.size > 0:
            pos = self.positions[actor_idx]
            vx = int(np.clip(pos[0], 0, self.width - 1))
            vy = int(np.clip(pos[1], 0, self.width - 1))
            local_vm = float(self.value_mass_field[vy, vx])
            # Normalise softly using a tanh squash to keep it in [0,1).
            spirit_pressure = float(np.tanh(max(0.0, local_vm)))

        return body_pressure, soul_pressure, spirit_pressure

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



