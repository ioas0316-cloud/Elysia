
from __future__ import annotations

import random
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable, NamedTuple
import json
import os

from pyquaternion import Quaternion

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from Core.Abstractions.Cell import Cell

# TODO: Integrate with Core modules
# from Project_Elysia.core.photon import PhotonEntity
# from Project_Elysia.core.spectrum import value_to_hue
from Core.Staging.chronicle import Chronicle
from Core.Life.Systems.skills import MARTIAL_STYLES, Move
from Core.Life.Systems.spells import SPELL_BOOK, cast_spell
from Core.Staging.world_event_logger import WorldEventLogger
from Core.Staging.genesis_engine import GenesisEngine
from Core.Staging.neural_eye import NeuralEye
# from ..wave_mechanics import WaveMechanics # This is in a higher level directory
from Core.Staging.fields import FieldRegistry
from Core.Staging.dialogue_kr import get_line as kr_dialogue
from Core.Staging.tensor_wave import Tensor3D, SoulTensor


# --- Cosmic Axis Constants: The 7 Directions of Ascension ---
ASCENSION_LIFE = 0       # Vitariael
ASCENSION_CREATION = 1   # Emetriel
ASCENSION_REFLECTION = 2 # Sophiel
ASCENSION_TRUTH = 3      # Gavriel
ASCENSION_SACRIFICE = 4  # Sarakhiel
ASCENSION_LOVE = 5       # Rafamiel
ASCENSION_LIBERATION = 6 # Lumiel

# --- Cosmic Axis Constants: The 7 Stages of Descent ---
DESCENT_DEATH = 0        # Motus
DESCENT_DISSOLUTION = 1  # Solvaris
DESCENT_IGNORANCE = 2    # Obscure
DESCENT_DISTORTION = 3   # Diabolos
DESCENT_SELF_OBSESSION = 4 # Lucifel
DESCENT_CONSUMPTION = 5  # Mammon
DESCENT_BONDAGE = 6      # Asmodeus


LawAction = Optional[Tuple[Optional[int], str, Optional[Move]]]


@dataclass
class LawPolicy:
    name: str
    priority: float
    condition: Callable[[int, csr_matrix, np.ndarray], bool]
    action: Callable[[int, csr_matrix, np.ndarray], LawAction]
    insight: Callable[[int, csr_matrix, np.ndarray], List[str]]


@dataclass
class LawEvaluationResult:
    action: LawAction
    reflections: List[str]
    policy_name: str


class LawPriorityManager:
    def __init__(self, world: "World", policies: List[LawPolicy]):
        self.world = world
        self.policies = policies

    def evaluate(
        self,
        actor_idx: int,
        adj_matrix_csr: csr_matrix,
        connected_indices: np.ndarray,
    ) -> Optional[LawEvaluationResult]:
        candidates: List[Tuple[float, str, LawAction, List[str]]] = []
        for policy in self.policies:
            if not policy.condition(actor_idx, adj_matrix_csr, connected_indices):
                continue
            action = policy.action(actor_idx, adj_matrix_csr, connected_indices)
            bonus = self.world.get_meta_priority_bonus(policy.name)
            reflections = policy.insight(actor_idx, adj_matrix_csr, connected_indices)
            candidates.append((policy.priority + bonus, policy.name, action, reflections))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        priority, name, action, reflections = candidates[0]
        return LawEvaluationResult(action=action, reflections=reflections, policy_name=name)


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

        # Language memetics (simple pattern weighting)
        self.meme_bank = defaultdict(float)  # key: (speech_act, obj) -> weight

        # --- Lexicon Physics Mastery (Elysia's Experience) ---
        # Tracks the 'mass' or 'weight' of each speech act based on its historical success.
        # Heavier words create stronger field ripples.
        self.lexicon_mastery = defaultdict(lambda: 1.0)

        # Gravity pulse (pressure cooker) controls
        self.gravity_pulse_period = 200         # ticks per full cycle (crunch + release)
        self.gravity_pulse_on_duration = 100    # ticks the pulse is active (crunch)
        self.gravity_pulse_strength = 80.0      # pull magnitude toward the center

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
        # --- Asymptotic Safety (Love Fixed Point) ---
        # If systemic energies/threats explode, clamp to a "safe" fixed point.
        self.asymptotic_safety_enabled: bool = True
        self.asymptotic_threat_cap: float = 50.0  # cap for threat field amplitude
        self.asymptotic_energy_cap: float = 500.0 # cap for per-cell energy/hp mirroring
        self.asymptotic_love_gain: float = 5.0    # coherence/value boost when clamped
        self.asymptotic_cooldown_ticks: int = 10  # duration of soft damp after clamp
        self.asymptotic_threat_damp: float = 0.85 # multiplier during cooldown
        self.asymptotic_coherence_boost: float = 0.5 # per-step boost during cooldown
        self._asymptotic_cooldown: int = 0
        # --- Micro-layer controls ---
        self.micro_layer_enabled: bool = False
        self.micro_tick_interval: int = 100
        self.micro_roi: List[Tuple[int, int, int]] = []
        self.micro_state: Dict[int, Dict[str, float]] = {}
        # --- Band-split fields (low/high frequency) ---
        self.band_split_enabled: bool = False
        self.band_low_resolution: int = 64  # low-res grid
        self.band_low_decay: float = 0.98
        self.band_high_decay: float = 0.90
        self.band_low_threat = None
        self.band_high_threat = None
        # --- Free-will collapse triggers ---
        self.free_will_threat_threshold: float = 80.0
        self.free_will_value_threshold: float = 120.0
        self.last_free_will_collapse: Optional[Dict[str, float]] = None
        # --- Entropy/Love dynamics ---
        self.entropy_decay: float = 0.995
        self.love_injection_gain: float = 0.05
        # Width must be defined before allocating entropy_field
        self.width = 256  # default; can be overridden after init
        self.entropy_field = np.zeros((self.width, self.width), dtype=np.float32)
        # --- Spectrum snapshot logging ---
        self.spectrum_log_interval: int = 50
        # --- Trust scenario (scarcity/cooperation) ---
        self.trust_scarcity: float = 0.4  # regen per tick
        self.trust_hunger_drain: float = 0.2

        # --- Policy Stack ---
        self.law_manager = self._build_law_manager()

        # --- Genesis Engine (Data-Driven Physics) ---
        self.genesis_engine = GenesisEngine(self)

        # --- Quantum State Management ---
        self.quantum_states: Dict[str, Dict[str, float]] = {}
        self.last_reflections: Dict[int, List[str]] = {}

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
        self.shields = np.array([], dtype=np.float32) # Protoss Shield
        self.max_shields = np.array([], dtype=np.float32)
        self.tech_level = np.array([], dtype=np.float32) # Terran Tech
        self.energy = np.array([], dtype=np.float32)
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
        # Simple economy signals
        self.wealth = np.array([], dtype=np.float32)
        self.prestige = np.array([], dtype=np.float32)


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
        self.experience_scars = np.array([], dtype=np.uint8) # Bitmask for experiences

        # --- Mental / Reflective Channels ---
        self.memory_strength = np.array([], dtype=np.float32)
        self.imagination_brightness = np.array([], dtype=np.float32)
        self.emotion_intensity = np.array([], dtype=np.float32)
        self.vision_awareness = np.array([], dtype=np.float32)
        self.auditory_clarity = np.array([], dtype=np.float32)
        self.gustatory_imbue = np.array([], dtype=np.float32)
        self.olfactory_sensitivity = np.array([], dtype=np.float32)
        self.tactile_feedback = np.array([], dtype=np.float32)
        self.meta_focus: str = "baseline"
        self.meta_focus_history: List[str] = []

        # --- Civilization Attributes ---
        self.continent = np.array([], dtype='<U10') # e.g., 'East', 'West'
        self.culture = np.array([], dtype='<U10') # e.g., 'wuxia', 'knight'
        self.affiliation = np.array([], dtype='<U20') # e.g., 'Wudang', 'Shaolin'


        # --- SciPy Sparse Matrix for Connections ---
        self.adjacency_matrix = lil_matrix((0, 0), dtype=np.float32)

        # --- Geology (Grid-based) ---
        self.width = 256  # Default size, can be configured

        # --- Neural Eye (Intuition Engine) ---
        # Must be initialized after self.width is defined
        self.neural_eye = NeuralEye(width=self.width)
        self._last_intuition_tick = 0

        self.height_map = np.zeros((self.width, self.width), dtype=np.float32)
        self.soil_fertility = np.full((self.width, self.width), 0.5, dtype=np.float32)
        self.wetness = np.zeros((self.width, self.width), dtype=np.float32) # 0.0 (dry) to 1.0 (puddle)

        # --- Spirit Layer: 3D Fields (The "Space" of Imagination & Future) ---
        # These fields represent the 'Atmosphere' or 'Will' that inclines events before they happen.
        self.threat_field = np.zeros((self.width, self.width), dtype=np.float32)
        self.coherence_field = np.zeros((self.width, self.width), dtype=np.float32)
        self.will_field = np.zeros((self.width, self.width), dtype=np.float32)
        self.value_mass_field = np.zeros((self.width, self.width), dtype=np.float32)
        self.norms_field = np.zeros((self.width, self.width), dtype=np.float32)
        self.hydration_field = np.zeros((self.width, self.width), dtype=np.float32)
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
        # Derived tensor field (structure/emotion/identity) and its spatial gradients
        self.tensor_field = np.zeros((self.width, self.width, 3), dtype=np.float32)
        self.tensor_field_grad_x = np.zeros_like(self.tensor_field)
        self.tensor_field_grad_y = np.zeros_like(self.tensor_field)
        # Historical imprint, Norms, Prestige fields (skeletons)
        self.h_imprint = np.zeros((self.width, self.width), dtype=np.float32)
        self.norms_field = np.zeros((self.width, self.width), dtype=np.float32)
        self.prestige_field = np.zeros((self.width, self.width), dtype=np.float32)

        # --- Cosmic Axis Fields (Ascension and Descent) ---
        self.ascension_field = np.zeros((self.width, self.width, 7), dtype=np.float32)
        self.descent_field = np.zeros((self.width, self.width, 7), dtype=np.float32)

        self._threat_decay = 0.92  # memory of threat (EMA)
        self._threat_sigma = 7.0   # spatial spread of threat influence (grid units)
        self._threat_gain = 1.0    # base contribution gain
        # Fast-path diffusion toggle: when True, uses a single gaussian_filter over
        # a weighted predator presence map instead of per-predator kernel stamping.
        self.use_fast_field_diffusion: bool = True
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

        # --- Incarnation Status ---
        self.demon_lord_status = 'sealed'  # sealed, awakening, unleashed
        self.angel_status = 'slumbering'   # slumbering, watching, manifested

        # --- Spiritual Event Queue ---
        self.spiritual_events = []

        # --- Khala Network (The Psionic Web) ---
        self.khala_connected_mask = np.array([], dtype=bool)
        self.delta_synchronization_factor = 0.0 # 0.0 = Chaos, 1.0 = Khala Unity


    def _resize_matrices(self, new_size: int):
        current_size = len(self.cell_ids)
        if new_size <= current_size:
            return

        # --- Core Game System Attributes ---
        self.is_alive_mask = np.pad(self.is_alive_mask, (0, new_size - current_size), 'constant', constant_values=False)
        self.hp = np.pad(self.hp, (0, new_size - current_size), 'constant')
        self.max_hp = np.pad(self.max_hp, (0, new_size - current_size), 'constant')
        self.energy = np.pad(self.energy, (0, new_size - current_size), 'constant')
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
        self.khala_connected_mask = np.pad(self.khala_connected_mask, (0, new_size - current_size), 'constant', constant_values=False)
        self.experience_scars = np.pad(self.experience_scars, (0, new_size - current_size), 'constant', constant_values=0)

        # Xel'Naga Attributes
        # Shield (Protoss): Regenerating HP buffer
        self.shields = np.pad(self.shields, (0, new_size - current_size), 'constant')
        self.max_shields = np.pad(self.max_shields, (0, new_size - current_size), 'constant')
        # Tech Level (Terran): Tool usage modifier
        self.tech_level = np.pad(self.tech_level, (0, new_size - current_size), 'constant')

        self.memory_strength = np.pad(self.memory_strength, (0, new_size - current_size), 'constant')
        self.imagination_brightness = np.pad(self.imagination_brightness, (0, new_size - current_size), 'constant')
        self.emotion_intensity = np.pad(self.emotion_intensity, (0, new_size - current_size), 'constant')
        self.vision_awareness = np.pad(self.vision_awareness, (0, new_size - current_size), 'constant')
        self.auditory_clarity = np.pad(self.auditory_clarity, (0, new_size - current_size), 'constant')
        self.gustatory_imbue = np.pad(self.gustatory_imbue, (0, new_size - current_size), 'constant')
        self.olfactory_sensitivity = np.pad(self.olfactory_sensitivity, (0, new_size - current_size), 'constant')
        self.tactile_feedback = np.pad(self.tactile_feedback, (0, new_size - current_size), 'constant')
        self.wealth = np.pad(self.wealth, (0, new_size - current_size), 'constant', constant_values=0.0)
        self.prestige = np.pad(self.prestige, (0, new_size - current_size), 'constant', constant_values=0.0)


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

    def get_field_tensor(self, x: int, y: int) -> Tensor3D:
        """
        Synthesizes a Tensor3D at the given coordinates by sampling environmental fields.
        This creates a 'Fractal Field' from scalar components.

        Mapping:
        - Structure (X): Norms (Social Order) + Prestige (Power)
        - Emotion (Y): Coherence (Positive) - Threat (Negative)
        - Identity (Z): Value Mass (Meaning) + Will (Direction)
        """
        x, y = int(np.clip(x, 0, self.width-1)), int(np.clip(y, 0, self.width-1))

        # Sample fields
        norms = float(self.norms_field[y, x])
        prestige = float(self.prestige_field[y, x])
        coherence = float(self.coherence_field[y, x])
        threat = float(self.threat_field[y, x])
        value_mass = float(self.value_mass_field[y, x])
        will = float(self.will_field[y, x])

        # Synthesize Tensor Components
        structure = (norms + prestige) * 0.5
        emotion = coherence - threat # Can be negative
        identity = (value_mass + will) * 0.5

        return Tensor3D(structure, emotion, identity)

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
        base_str = properties.get('strength', 5)
        base_agi = properties.get('agility', 5)
        base_int = properties.get('intelligence', 5)
        base_vit = properties.get('vitality', 5)
        base_wis = properties.get('wisdom', 5)
        gender = properties.get('gender', self.genders[idx] if idx < len(self.genders) else '')
        # Simple gender tilt: males lean physical, females lean sustain/social
        if gender == 'male':
            base_str += 2
            base_vit += 1
        elif gender == 'female':
            base_wis += 2
            base_int += 1

        self.strength[idx] = base_str
        self.agility[idx] = base_agi
        self.intelligence[idx] = base_int
        self.vitality[idx] = base_vit
        self.wisdom[idx] = base_wis
        # Economy seeds
        self.wealth[idx] = float(properties.get('wealth', random.uniform(5.0, 15.0)))
        self.prestige[idx] = float(properties.get('prestige', random.uniform(0.0, 2.0)))

        # Derived stats (HP/Ki/Mana/Faith) are calculated from base stats
        self.max_hp[idx] = self.vitality[idx] * 10
        self.hp[idx] = self.max_hp[idx]
        base_energy = properties.get('energy', properties.get('hp', float(self.hp[idx])))
        try:
            self.energy[idx] = float(base_energy)
        except (TypeError, ValueError):
            self.energy[idx] = float(self.hp[idx])

        # --- Xel'Naga Protocol: Racial Traits ---
        culture = properties.get('culture', '')
        if culture == 'protoss':
            # Protoss: High Shields, Psionic (Mana/Wisdom)
            self.max_shields[idx] = self.max_hp[idx] * 0.5
            self.shields[idx] = self.max_shields[idx]
            self.max_mana[idx] = self.wisdom[idx] * 20 # High psionic potential
            self.mana[idx] = self.max_mana[idx]
            self.khala_connected_mask[idx] = True # Born into the Khala
            self.max_ki[idx] = 0
            self.ki[idx] = 0
            self.max_faith[idx] = 0
            self.faith[idx] = 0

        elif culture == 'terran':
            # Terran: High Tech, Adaptive
            self.tech_level[idx] = 1.0
            self.max_hp[idx] *= 1.2 # Stimpack/Armor
            self.hp[idx] = self.max_hp[idx]
            self.max_ki[idx] = 0
            self.ki[idx] = 0
            self.max_mana[idx] = 0
            self.mana[idx] = 0
            self.max_faith[idx] = 0
            self.faith[idx] = 0

        elif culture == 'zerg':
            # Zerg: High Regen, Swarm (Low individual stats but fast growth)
            self.max_hp[idx] *= 0.8
            self.hp[idx] = self.max_hp[idx]
            # Zerg regen is handled in passive update
            self.max_ki[idx] = 0
            self.ki[idx] = 0
            self.max_mana[idx] = 0
            self.mana[idx] = 0
            self.max_faith[idx] = 0
            self.faith[idx] = 0

        # Legacy Support (Wuxia/Knight map to Protoss/Terran archetypes loosely)
        elif culture == 'wuxia':
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


        try:
            self.hunger[idx] = float(properties.get('hunger', 100.0))
        except (TypeError, ValueError):
            self.hunger[idx] = 100.0
        try:
            self.hydration[idx] = float(properties.get('hydration', 100.0))
        except (TypeError, ValueError):
            self.hydration[idx] = 100.0
        try:
            self.temperature[idx] = float(properties.get('temperature', 36.5))
        except (TypeError, ValueError):
            self.temperature[idx] = 36.5
        try:
            self.satisfaction[idx] = float(properties.get('satisfaction', 50.0))
        except (TypeError, ValueError):
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
        pos_dict = temp_cell.properties.get('position', {'x': random.uniform(-10, 10), 'y': random.uniform(-10, 10), 'z': random.uniform(-10, 10)})
        self.positions[idx] = [pos_dict.get('x', 0), pos_dict.get('y', 0), pos_dict.get('z', 0)]
        if temp_cell:
            # Now that the cell is materialized and has its properties, update the numpy arrays
            self.element_types[idx] = temp_cell.element_type
            self.diets[idx] = temp_cell.properties.get('diet', 'omnivore')
            self.genders[idx] = temp_cell.properties.get('gender', '')
            self.labels[idx] = temp_cell.properties.get('label', concept_id)
        else:
            # Fallback if materialization fails, but we've already set element_type above
            self.diets[idx] = 'omnivore'
            self.genders[idx] = ''
            self.labels[idx] = concept_id

        # --- Lifespan selection (years -> ticks) based on label/type + external (mana/ki/faith) bonus ---
        def _lifespan_years(label: str, element_type: str) -> float:
            label = (label or '').lower()
            if label in ('human', 'villager', 'monk', 'knight', 'wizard', 'mage', 'cleric', 'priest', 'warrior'):
                base = random.randint(70, 90)
            elif label in ('wolf', 'deer'):
                base = random.randint(12, 20)
            elif label in ('tree',):
                base = random.randint(80, 200)
            elif label in ('bush', 'plant'):
                base = random.randint(5, 15)
            elif element_type == 'animal':
                base = random.randint(18, 30)
            elif element_type == 'life':
                base = random.randint(8, 40)
            else:
                base = random.randint(60, 120)

            # External power bonus (mana/ki/faith) extends lifespan slightly.
            ext_power = float(self.max_mana[idx] + self.max_ki[idx] + self.max_faith[idx])
            bonus = min(20.0, ext_power / 30.0)
            return base + bonus

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


        # Avoid instant old-age deaths when seeded age exceeds sampled lifespan.
        if self.age[idx] >= self.max_age[idx]:
            safety_pad = max(self._year_length_ticks(), int(5 * self._year_length_ticks()))
            self.max_age[idx] = int(self.age[idx] + safety_pad)
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
            if 'max_hp' in properties:
                try:
                    self.max_hp[idx] = float(properties['max_hp'])
                except (TypeError, ValueError):
                    pass
            if 'hp' in properties:
                try:
                    self.hp[idx] = float(properties['hp'])
                except (TypeError, ValueError):
                    pass
            if 'energy' in properties:
                try:
                    self.energy[idx] = float(properties['energy'])
                except (TypeError, ValueError):
                    pass
            elif 'hp' in properties:
                # Mirror hp override into energy when explicit energy not provided
                self.energy[idx] = float(self.hp[idx])
            # Ensure hp never exceeds max_hp after overrides
        if self.hp[idx] > self.max_hp[idx]:
            self.hp[idx] = self.max_hp[idx]


    def materialize_cell(self, concept_id: str, force_materialize: bool = False, explicit_properties: Optional[Dict] = None) -> Optional[Cell]:
        """
        Materializes a Cell from the 'Quantum State' (KG or memory).
        Upgraded to fully hydrate the SoulTensor (Physics State).
        """
        if not force_materialize and concept_id in self.materialized_cells:
            return self.materialized_cells[concept_id]
        if concept_id in self.quantum_states:
            idx = self.id_to_idx.get(concept_id)
            if idx is None:
                self.logger.error(f"Quantum state for '{concept_id}' exists, but it has no index in the world.")
                return None

            # Fetch node properties from KG (tests may inject mocks without kg_manager)
            node_data = None
            kg_manager = getattr(self.wave_mechanics, "kg_manager", None)

            # --- Quantum Thawing: Load Physics State ---
            soul_tensor = None
            if hasattr(self.wave_mechanics, "get_node_tensor"):
                # Use the new entanglement-aware loader
                soul_tensor = self.wave_mechanics.get_node_tensor(concept_id)

            if kg_manager and hasattr(kg_manager, "get_node"):
                try:
                    node_data = kg_manager.get_node(concept_id)
                except Exception:
                    node_data = None
            initial_properties = node_data.copy() if isinstance(node_data, dict) else {}

            # Merge with explicit properties, which take precedence
            if explicit_properties:
                initial_properties.update(explicit_properties)

        # Create a new lightweight Cell dataclass instance.
        # All mutable state is in World's numpy arrays, not on the cell object.
        cell = Cell(
            id=concept_id,
            dna=self.primordial_dna,
            properties=initial_properties
        )

        # --- Soul/Tensor Hydration (Future) ---
        # If a soul_tensor is available, we can attach it.
        # if soul_tensor:
        #     cell.tensor = soul_tensor

        # Crucially, update the numpy array with the correct element type upon materialization
        self.element_types[idx] = cell.element_type

        self.materialized_cells[concept_id] = cell
        return cell
        return None

    def crystallize_cell(self, cell: Cell):
        """
        [The Ice Star Protocol]
        Freezes the 'Fire' (Living Cell State) back into 'Ice' (KG Node).
        Preserves the SoulTensor state so experience is not lost.
        """
        if not self.wave_mechanics:
            return

    # In the new model, there's nothing to sync from the Cell object itself.
    # Instead, we would read the state from the World's NumPy arrays for that cell's index
    # and update the Knowledge Graph based on that. This logic will be implemented later.

    # TODO: (SOUL FUSION) Implement logic to read state from NumPy arrays and update KG.
    # e.g., cell.sync_soul_to_body() is now an external process.

        # Use the WaveMechanics update method which handles entanglement
        # If this cell is entangled, its shared state gets updated for everyone.
    # if hasattr(self.wave_mechanics, 'update_node_tensor'):
    #     self.wave_mechanics.update_node_tensor(cell.id, cell.tensor)

        self.logger.info(f"CRYSTALLIZE: Preserved SoulTensor state for '{cell.id}' back to the Cosmos.")

    def _sync_states_to_objects(self):
        """
        DEPRECATED. With the flyweight Cell, there is no state to sync back to the objects.
        State lives in the World's NumPy arrays. This method is kept as a placeholder
        during transition and can be removed later.
        """
        pass


    def run_simulation_step(self) -> Tuple[List[Cell], List[AwakeningEvent]]:
        if self.chronicle:
            event = self.chronicle.record_event('simulation_step_run', {}, [], self.branch_id, self.parent_event_id)
            self.parent_event_id = event['id']
        self.time_step += 1
        self.spiritual_events.clear() # Clear events at the start of the step

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
        self._update_coherence_field()
        self._update_intentional_field()
        self._update_tensor_field()
        # Band-split refinement
        self._update_band_split_fields()
        # Free-will collapse trigger based on threat/value peaks
        self._maybe_trigger_free_will_collapse()
        # Entropy field decay
        self._decay_entropy_field()
        # Trust/Betrayal dynamics (cooperation incentives under scarcity)
        self._apply_trust_scenario_tick()
        # Spectrum snapshot logging (value/threat/coherence -> photon)
        if self.time_step % max(1, self.spectrum_log_interval) == 0:
            self._log_spectrum_snapshot()
        # Safety rail: prevent runaway threat/energy from collapsing the simulation.
        triggered = self._apply_asymptotic_safety_guard()
        # Apply cooldown smoothing if triggered recently.
        self._apply_asymptotic_cooldown_effects(triggered)
        # Optional micro-layer (ROI-limited) update
        self._update_micro_layer()

        # Update passive resources (MP regen, hunger, starvation)
        self._update_passive_resources()
        # Survival pain/drive signals (hunger, isolation)
        self._apply_survival_pain()
        # Cold world: body-heat law (forces clustering for survival)
        self._apply_body_heat_law()
        # Gravity pulse: periodically crunch everyone toward the center to force encounters.
        self._apply_gravity_pulse()
        # Elders teach nearby youngsters (knowledge transfer)
        self._apply_teaching()
        # Light economy/authority loop (trade/tribute)
        self._apply_economy()
        # Keep connection graph light to avoid combinatorial explosion
        if self.time_step % 10 == 0:
            self._prune_connections()

        # Macro-scale narrative/disaster hooks (war/famine/bounty/plague/storm/omens).
        # These are soft, opt-in overlays driven by macro_* attributes and do not
        # run unless enable_macro_disaster_events is True.
        self._apply_macro_disaster_events()

        # Process major state changes and actions
        newly_born_cells = []
        self._process_animal_actions()
        newly_born_cells.extend(self._process_life_cycles())

        # --- Apply Cosmic Laws ---
        self._apply_cosmic_laws()

        # Apply final physics and cleanup
        self._apply_physics_and_cleanup(newly_born_cells)

        # --- Law of Existential Change (e > r) ---
        awakening_events = self._apply_law_of_awakening()

        # --- The Khala Synchronization (Delta One) ---
        self._synchronize_khala()

        # --- Neural Eye Perception Cycle ---
        # Run every 10 ticks to save compute and simulate "subconscious processing" time
        if self.time_step % 10 == 0:
            self._process_neural_intuition()

        # --- Fractal Soul Growth Cycle (New) ---
        # Process the inner soul of every materialized cell.
        # This allows the soul to grow (resonate) even if the body is idle.
        self._process_soul_cycles()
        # Prepare next-step snapshot
        try:
            self._prev_hp = self.hp.copy()
        except Exception:
            pass

        return newly_born_cells, awakening_events

    def _apply_asymptotic_safety_guard(self) -> bool:
        """
        Asymptotic Safety guard: when threat/energy diverge, clamp to a 'love' fixed point.
        - Threat field clipped to cap; coherence/value_mass nudged upward as stabilizer.
        - Energy (and mirrored HP) clipped softly.
        """
        if not getattr(self, "asymptotic_safety_enabled", True):
            return False

        try:
            threat_max = float(self.threat_field.max()) if self.threat_field.size else 0.0
            energy_max = float(self.energy.max()) if self.energy.size else 0.0
            triggered = False

            if threat_max > self.asymptotic_threat_cap:
                self.threat_field = np.clip(self.threat_field, 0.0, self.asymptotic_threat_cap)
                # Love/coherence injection proportional to overshoot (softened).
                bleed = (threat_max - self.asymptotic_threat_cap) * 0.02
                if self.coherence_field.size:
                    self.coherence_field = self.coherence_field + bleed
                if self.value_mass_field.size:
                    self.value_mass_field = self.value_mass_field + bleed
                triggered = True

            if energy_max > self.asymptotic_energy_cap and self.energy.size:
                self.energy = np.minimum(self.energy, self.asymptotic_energy_cap)
                # Mirror into HP to keep body/energy consistent
                if self.hp.size:
                    self.hp = np.minimum(self.hp, self.asymptotic_energy_cap)
                triggered = True

            if triggered:
                self.event_logger.log(
                    "ASYMPTOTIC_SAFETY",
                    self.time_step,
                    threat_max=threat_max,
                    energy_max=energy_max,
                    cap_threat=self.asymptotic_threat_cap,
                    cap_energy=self.asymptotic_energy_cap,
                )
                if self.logger:
                    self.logger.warning(
                        f"ASYMPTOTIC SAFETY: threat_max={threat_max:.2f}, energy_max={energy_max:.2f} -> clamped to love fixed point."
                    )
                self._asymptotic_cooldown = max(self._asymptotic_cooldown, self.asymptotic_cooldown_ticks)
            return triggered
        except Exception:
            # Never let safety guard crash the sim.
            return False

    def _apply_asymptotic_cooldown_effects(self, triggered: bool) -> None:
        """
        After a clamp, gently damp threat and lift coherence/value_mass for a few ticks.
        """
        if self._asymptotic_cooldown <= 0:
            return
        try:
            if self.threat_field.size:
                self.threat_field *= float(self.asymptotic_threat_damp)
            boost = float(self.asymptotic_coherence_boost)
            if self.coherence_field.size:
                self.coherence_field = self.coherence_field + boost
            if self.value_mass_field.size:
                self.value_mass_field = self.value_mass_field + boost
            self._asymptotic_cooldown -= 1
        except Exception:
            self._asymptotic_cooldown = max(0, self._asymptotic_cooldown - 1)

    def _ensure_band_buffers(self) -> None:
        """
        Initializes low/high frequency buffers if needed.
        """
        if not self.band_split_enabled:
            return
        low_res = max(1, int(self.band_low_resolution))
        if self.band_low_threat is None or self.band_low_threat.shape != (low_res, low_res):
            self.band_low_threat = np.zeros((low_res, low_res), dtype=np.float32)
        if self.band_high_threat is None or self.band_high_threat.shape != self.threat_field.shape:
            self.band_high_threat = np.zeros_like(self.threat_field)

    def _update_band_split_fields(self) -> None:
        """
        Splits threat_field into low/high frequency components:
        - Low: downsampled + decay (captures broad flow)
        - High: residual after subtracting upsampled low (captures local spikes)
        """
        if not getattr(self, "band_split_enabled", False):
            return
        if self.threat_field.size == 0:
            return

        self._ensure_band_buffers()
        low_res = self.band_low_threat.shape[0]

        try:
            # Downsample threat to low-res (mean pooling)
            factor = max(1, self.width // low_res)
            reshaped = self.threat_field[:low_res * factor, :low_res * factor].reshape(
                low_res, factor, low_res, factor
            )
            pooled = reshaped.mean(axis=(1, 3))

            # Decay + blend
            self.band_low_threat = (self.band_low_threat * self.band_low_decay) + (pooled * (1.0 - self.band_low_decay))

            # Upsample low to full res (nearest)
            low_up = np.repeat(np.repeat(self.band_low_threat, factor, axis=0), factor, axis=1)
            # Pad to match width if needed
            if low_up.shape != self.threat_field.shape:
                pad_y = self.threat_field.shape[0] - low_up.shape[0]
                pad_x = self.threat_field.shape[1] - low_up.shape[1]
                low_up = np.pad(low_up, ((0, pad_y), (0, pad_x)), mode='edge')

            residual = self.threat_field - low_up
            self.band_high_threat = (self.band_high_threat * self.band_high_decay) + (residual * (1.0 - self.band_high_decay))
        except Exception:
            # If anything fails, keep threat_field as-is
            return

    def _maybe_trigger_free_will_collapse(self) -> None:
        """
        Trigger a 'free-will' collapse event if threat/value peaks exceed thresholds.
        Logs the event; downstream consumers can hook this to drive HyperQubit collapse/decisions.
        """
        try:
            threat_peak = float(self.threat_field.max()) if self.threat_field.size else 0.0
            value_peak = float(self.value_mass_field.max()) if self.value_mass_field.size else 0.0
            triggered = False
            reason = []
            if threat_peak >= self.free_will_threat_threshold:
                triggered = True
                reason.append("threat")
            if value_peak >= self.free_will_value_threshold:
                triggered = True
                reason.append("value")

            if triggered:
                payload = {
                    "time_step": self.time_step,
                    "threat_peak": threat_peak,
                    "value_peak": value_peak,
                    "reason": ",".join(reason),
                }
                self.last_free_will_collapse = payload
                self.event_logger.log("FREE_WILL_COLLAPSE", self.time_step, **payload)
                if self.logger:
                    self.logger.warning(
                        f"FREE_WILL_COLLAPSE: reason={payload['reason']} threat={threat_peak:.2f} value={value_peak:.2f}"
                    )
        except Exception:
            return

    def _decay_entropy_field(self) -> None:
        """
        Applies entropy decay and optionally couples to value/coherence fields to simulate forgetting pressure.
        """
        try:
            self.entropy_field *= self.entropy_decay
            # Couple: higher entropy pulls down value/coherence slightly
            if self.value_mass_field.size:
                self.value_mass_field -= (self.entropy_field * 0.001)
                np.maximum(self.value_mass_field, 0.0, out=self.value_mass_field)
            if self.coherence_field.size:
                self.coherence_field -= (self.entropy_field * 0.001)
                np.maximum(self.coherence_field, 0.0, out=self.coherence_field)
        except Exception:
            return

    def _apply_trust_scenario_tick(self) -> None:
        """
        Lightweight trust/betrayal dynamics:
        - Agents lose hunger, gain scarcity regen.
        - Random requester/target interaction; share/reciprocate boosts coherence and trust; betrayal drops coherence.
        Simplified inline version from tools/trust_scenario.
        """
        try:
            if len(self.cell_ids) < 2:
                return
            # Hunger drain + scarcity regen
            self.hunger = np.maximum(0.0, self.hunger - self.trust_hunger_drain)
            self.hunger = np.minimum(100.0, self.hunger + (self.trust_scarcity * 100.0))

            # Pick two alive agents
            alive_indices = np.where(self.is_alive_mask)[0]
            if alive_indices.size < 2:
                return
            requester_idx, target_idx = np.random.choice(alive_indices, size=2, replace=False)
            req_id = self.cell_ids[requester_idx]
            tgt_id = self.cell_ids[target_idx]

            # Proxy for "food": use hydration as resource
            req_food = self.hydration[requester_idx]
            tgt_food = self.hydration[target_idx]
            need = req_food < 60.0
            if not need:
                return

            # Trust proxy: use norms_field at positions if available else baseline
            trust_level = 0.7
            try:
                px = int(self.positions[target_idx, 0]) % self.width
                py = int(self.positions[target_idx, 1]) % self.width
                trust_level = min(1.0, max(0.0, float(self.norms_field[py, px])))
            except Exception:
                pass

            share_prob = min(1.0, 0.5 + 0.4 * trust_level)
            share = (np.random.random() < share_prob) and (tgt_food > 20.0)

            if share:
                amt = min(20.0, tgt_food * 0.5)
                self.hydration[target_idx] = max(0.0, tgt_food - amt)
                self.hydration[requester_idx] = min(100.0, req_food + amt)

                repay = np.random.random() < 0.75
                if repay and self.hydration[requester_idx] > 30.0:
                    repay_amt = 10.0
                    self.hydration[requester_idx] = max(0.0, self.hydration[requester_idx] - repay_amt)
                    self.hydration[target_idx] = min(100.0, self.hydration[target_idx] + repay_amt)
                    # Coherence boost around both
                    if self.coherence_field.size:
                        try:
                            rx, ry = int(self.positions[requester_idx, 0]), int(self.positions[requester_idx, 1])
                            tx, ty = int(self.positions[target_idx, 0]), int(self.positions[target_idx, 1])
                            self.coherence_field[ry % self.width, rx % self.width] += 0.05
                            self.coherence_field[ty % self.width, tx % self.width] += 0.05
                        except Exception:
                            pass
                else:
                    # Mild coherence bump for sharing
                    if self.coherence_field.size:
                        try:
                            tx, ty = int(self.positions[target_idx, 0]), int(self.positions[target_idx, 1])
                            self.coherence_field[ty % self.width, tx % self.width] += 0.02
                        except Exception:
                            pass
            else:
                # Betrayal: local coherence drop at target
                if self.coherence_field.size:
                    try:
                        tx, ty = int(self.positions[target_idx, 0]), int(self.positions[target_idx, 1])
                        self.coherence_field[ty % self.width, tx % self.width] = max(
                            0.0, self.coherence_field[ty % self.width, tx % self.width] - 0.05
                        )
                    except Exception:
                        pass
        except Exception:
            return

    def _log_spectrum_snapshot(self) -> None:
        """
        Convert current field peaks into a photon-like snapshot for spectrum-aware monitoring.
        """
        try:
            value_peak = float(self.value_mass_field.max()) if self.value_mass_field.size else 0.0
            coherence_peak = float(self.coherence_field.max()) if self.coherence_field.size else 0.0
            threat_peak = float(self.threat_field.max()) if self.threat_field.size else 0.0
            # Normalize to 0..1 for hue mapping (rough heuristic)
            norm_value = min(1.0, value_peak / 200.0)
            hue = value_to_hue(norm_value, value_range=(0.0, 1.0))
            photon = PhotonEntity(
                hue=hue,
                intensity=min(1.0, norm_value),
                polarization=(0.0, 0.0, 1.0),
                payload=f"value={value_peak:.2f},coh={coherence_peak:.2f},threat={threat_peak:.2f}",
            )
            self.event_logger.log("SPECTRUM_SNAPSHOT", self.time_step, photon=photon.as_dict())
        except Exception:
            return

    # --- Micro-layer (ROI-limited) scaffolding -------------------------------
    def set_micro_roi(self, roi_list: List[Tuple[int, int, int]]) -> None:
        """
        ROI 리스트 설정. 각 항목은 (x, y, r) 정수 좌표/반경.
        """
        self.micro_roi = roi_list

    def _update_micro_layer(self) -> None:
        """
        최소한의 미시 레이어 업데이트:
        - ROI가 없거나 비활성화면 건너뜀
        - tick 간격에 맞춰 샘플링
        - ROI 내 threat/value_mass/coherence 평균을 계산해 micro_state에 기록
        - 선택적으로 value_mass_field에 미세한 보정
        """
        if not getattr(self, "micro_layer_enabled", False):
            return
        if getattr(self, "micro_tick_interval", 50) <= 0:
            return
        if self.time_step % int(self.micro_tick_interval) != 0:
            return
        if not hasattr(self, "micro_roi") or not self.micro_roi:
            return

        if not hasattr(self, "micro_state"):
            self.micro_state = {}

        for idx, (cx, cy, r) in enumerate(self.micro_roi):
            x0 = max(0, int(cx - r))
            x1 = min(self.width, int(cx + r))
            y0 = max(0, int(cy - r))
            y1 = min(self.width, int(cy + r))
            if x0 >= x1 or y0 >= y1:
                continue

            threat = float(self.threat_field[y0:y1, x0:x1].mean()) if self.threat_field.size else 0.0
            value_mass = float(self.value_mass_field[y0:y1, x0:x1].mean()) if self.value_mass_field.size else 0.0
            coherence = float(self.coherence_field[y0:y1, x0:x1].mean()) if self.coherence_field.size else 0.0

            self.micro_state[idx] = {
                "center": (cx, cy),
                "r": r,
                "threat": threat,
                "value_mass": value_mass,
                "coherence": coherence,
                "tick": self.time_step,
            }

            if coherence < 0.1 and self.value_mass_field.size:
                self.value_mass_field[y0:y1, x0:x1] = self.value_mass_field[y0:y1, x0:x1] + 0.01

    def _process_neural_intuition(self):
        """
        Uses the Neural Eye to scan the world for high-level patterns.
        """
        try:
            intuitions = self.neural_eye.perceive(self)
            for insight in intuitions:
                # Log the intuition
                self.logger.info(f"INTUITION: {insight['description']} (Intensity: {insight['intensity']:.2f})")

                # Map intuition to event log
                if insight['type'] == 'intuition_conflict':
                    self.event_logger.log('INTUITION_CONFLICT', self.time_step,
                                          intensity=insight['intensity'],
                                          location=insight.get('location'))
                elif insight['type'] == 'intuition_harmony':
                    self.event_logger.log('INTUITION_HARMONY', self.time_step,
                                          intensity=insight['intensity'])

        except Exception as e:
            self.logger.error(f"Neural Eye Blinked (Error): {e}")

    def _process_soul_cycles(self):
        """
        Iterates through all materialized cells and advances their Soul State.
        The Soul (SelfFractalCell) processes internal waves and generates resonance.
        This resonance (Phase Complexity) then feeds back into the Body (Insight/Mana).
        """
        for cell_id, cell in self.materialized_cells.items():
            idx = self.id_to_idx.get(cell_id)
            if idx is None or not self.is_alive_mask[idx]:
                continue

            # 1. Grow the Soul (Wave Propagation)
            # TODO: (SOUL FUSION) Properly initialize cell.soul. For now, skip if None.
            if cell.soul is None:
                continue
            # Returns: active_nodes (breadth), harmonic_richness (depth/complexity)
            active_nodes, richness = cell.soul.autonomous_grow()

            # 2. Soul-Body Feedback Loop
            # High resonance (richness) grants spiritual insight and mana regen.
            if richness > 5.0:
                idx = self.id_to_idx.get(cell_id)
                if idx is not None:
                    # Grant Insight (Understanding the complexity of self)
                    self.insight[idx] += richness * 0.01

                    # Regenerate Mana/Ki (Spiritual Energy)
                    # Deep soul resonance acts as a source of power
                    if self.max_mana[idx] > 0:
                        self.mana[idx] = min(self.max_mana[idx], self.mana[idx] + richness * 0.5)
                    if self.max_ki[idx] > 0:
                        self.ki[idx] = min(self.max_ki[idx], self.ki[idx] + richness * 0.5)

                    # Log significant spiritual breakthroughs
                    if richness > 50.0 and self.time_step % 10 == 0:
                         self.logger.info(f"SOUL RESONANCE: '{cell_id}' is experiencing deep internal harmony (Richness: {richness:.1f}).")
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

    def imprint_cozy_kitchen_scene(self, center_x: int = 128, center_y: int = 128, radius: int = 20):
        """
        Softly imprint a 'cozy kitchen' sensory scene onto the world fields.

        This does not force any behavior; it only nudges environmental fields
        so that dream observers and cells can sense a warm, safe place.
        """
        r = max(1, int(radius))
        x0 = max(0, center_x - r)
        x1 = min(self.width, center_x + r)
        y0 = max(0, center_y - r)
        y1 = min(self.width, center_y + r)

        if x0 >= x1 or y0 >= y1:
            return

        region = (slice(y0, y1), slice(x0, x1))

        # Slightly warmer, comfortable humidity and wetness.
        self.ambient_temperature_c = max(self.ambient_temperature_c, 20.0)
        self.humidity = min(1.0, max(self.humidity, 0.6))
        self.wetness[region] = np.maximum(self.wetness[region], 0.3)

        # Fertile, well‑tended soil (for an earthy, food-like scent).
        self.soil_fertility[region] = np.minimum(1.0, np.maximum(self.soil_fertility[region], 0.8))

        # Lower threat and a little more value mass to suggest safety and meaning.
        self.threat_field[region] *= 0.3
        self.value_mass_field[region] += 0.5

        # Log a soft scene event for observers.
        try:
            self.event_logger.log(
                "SCENE_COZY_KITCHEN",
                self.time_step,
                region={"center_x": center_x, "center_y": center_y, "radius": r},
            )
        except Exception:
            pass

    def imprint_rainy_street_scene(self, center_x: int = 64, center_y: int = 192, radius: int = 24):
        """
        Softly imprint a 'rainy street' scene: cool air, wet ground, dim light.
        """
        r = max(1, int(radius))
        x0 = max(0, center_x - r)
        x1 = min(self.width, center_x + r)
        y0 = max(0, center_y - r)
        y1 = min(self.width, center_y + r)

        if x0 >= x1 or y0 >= y1:
            return

        region = (slice(y0, y1), slice(x0, x1))

        # Cooler temperature and higher humidity, wet pavement.
        self.ambient_temperature_c = min(self.ambient_temperature_c, 14.0)
        self.humidity = min(1.0, max(self.humidity, 0.7))
        self.wetness[region] = np.maximum(self.wetness[region], 0.7)

        # Slightly reduced soil fertility (stone/road), but increased norms/prestige hints (city life).
        self.soil_fertility[region] = np.minimum(self.soil_fertility[region], 0.4)
        self.norms_field[region] += 0.3
        self.prestige_field[region] += 0.2

        # Threat stays low; value mass neutral so it feels melancholic but safe.
        self.threat_field[region] *= 0.7

        try:
            self.event_logger.log(
                "SCENE_RAINY_STREET",
                self.time_step,
                region={"center_x": center_x, "center_y": center_y, "radius": r},
            )
        except Exception:
            pass

    def imprint_dawn_forest_scene(self, center_x: int = 192, center_y: int = 64, radius: int = 28):
        """
        Softly imprint a 'dawn forest' scene: soft light, cool air, earthy scent.
        """
        r = max(1, int(radius))
        x0 = max(0, center_x - r)
        x1 = min(self.width, center_x + r)
        y0 = max(0, center_y - r)
        y1 = min(self.width, center_y + r)

        if x0 >= x1 or y0 >= y1:
            return

        region = (slice(y0, y1), slice(x0, x1))

        # Gentle temperature and humidity, slightly moist forest floor.
        self.ambient_temperature_c = min(max(self.ambient_temperature_c, 10.0), 18.0)
        self.humidity = min(1.0, max(self.humidity, 0.6))
        self.wetness[region] = np.maximum(self.wetness[region], 0.4)

        # High soil fertility for forest, low norms/prestige.
        self.soil_fertility[region] = np.minimum(1.0, np.maximum(self.soil_fertility[region], 0.9))
        self.norms_field[region] *= 0.5
        self.prestige_field[region] *= 0.5

        # Threat slightly reduced; value mass nudged up for awe/beauty.
        self.threat_field[region] *= 0.6
        self.value_mass_field[region] += 0.3

        try:
            self.event_logger.log(
                "SCENE_DAWN_FOREST",
                self.time_step,
                region={"center_x": center_x, "center_y": center_y, "radius": r},
            )
        except Exception:
            pass

    def imprint_starry_hill_scene(self, center_x: int = 128, center_y: int = 32, radius: int = 22):
        """
        Softly imprint a 'starry hill' scene: cold, clear air under a bright night sky.
        """
        r = max(1, int(radius))
        x0 = max(0, center_x - r)
        x1 = min(self.width, center_x + r)
        y0 = max(0, center_y - r)
        y1 = min(self.width, center_y + r)

        if x0 >= x1 or y0 >= y1:
            return

        region = (slice(y0, y1), slice(x0, x1))

        # Colder, very dry and clear.
        self.ambient_temperature_c = min(self.ambient_temperature_c, 5.0)
        self.humidity = min(self.humidity, 0.3)
        self.wetness[region] = np.minimum(self.wetness[region], 0.1)

        # Slightly elevated will/value toward contemplation.
        self.will_field[region] += 0.4
        self.value_mass_field[region] += 0.2

        # Threat very low; coherence slightly higher for a quiet, reflective place.
        self.threat_field[region] *= 0.2
        self.coherence_field[region] += 0.5

        try:
            self.event_logger.log(
                "SCENE_STARRY_HILL",
                self.time_step,
                region={"center_x": center_x, "center_y": center_y, "radius": r},
            )
        except Exception:
            pass

    def imprint_spiral_coil_field(self, center_x: int = 128, center_y: int = 128, radius: float = 80.0, turns: float = 3.0, strength: float = 1.0) -> None:
        """
        텐서 코일 감각의 나선 흐름을 의도/가치/의지 필드에 한 번에 인프린트한다.
        흐름 위에 올라타기만 하면 되도록 vector field(의도)와 scalar field(가치/의지)를 동시에 깐다.
        """
        rad = max(1.0, float(radius))
        width = self.width
        xs = np.arange(width, dtype=np.float32)
        ys = np.arange(width, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        dx = gx - float(center_x)
        dy = gy - float(center_y)
        r = np.sqrt(dx * dx + dy * dy) + 1e-6

        # Spiral phase grows with radius; arctan2 anchors direction.
        phase = (turns * (r / rad) * 2.0 * np.pi) + np.arctan2(dy, dx)
        vx = -np.sin(phase)  # tangential component
        vy = np.cos(phase)

        # Soft falloff so 코일 가장자리가 부드럽게 사라짐.
        decay = np.clip(1.0 - (r / rad), 0.0, 1.0) ** 1.5
        amp = (strength * decay).astype(np.float32)
        mask = (r <= rad)
        amp_mask = amp * mask

        # Intentional vector field gains a swirl; agents 샘플 후 그대로 이동.
        self.intentional_field[..., 0] += (vx * amp_mask).astype(np.float32)
        self.intentional_field[..., 1] += (vy * amp_mask).astype(np.float32)

        # Scalar carriers: value/will/coherence boosted along the coil.
        self.value_mass_field += amp_mask * 0.6
        self.will_field += amp_mask * 0.4
        self.coherence_field = (self._coh_alpha * self.coherence_field) + ((1.0 - self._coh_alpha) * amp_mask)

        try:
            self.event_logger.log(
                "SCENE_SPIRAL_COIL",
                self.time_step,
                params={"center_x": center_x, "center_y": center_y, "radius": rad, "turns": turns, "strength": strength},
            )
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

        # --- Protoss Shield Regeneration (Fast out of combat) ---
        shield_regen_mask = self.is_alive_mask & (self.shields < self.max_shields)
        self.shields[shield_regen_mask] = np.minimum(self.max_shields[shield_regen_mask], self.shields[shield_regen_mask] + 2.0)

        # --- Zerg HP Regeneration (Biological) ---
        zerg_mask = (self.culture == 'zerg') & self.is_alive_mask & (self.hp < self.max_hp)
        self.hp[zerg_mask] = np.minimum(self.max_hp[zerg_mask], self.hp[zerg_mask] + 1.5)

        # --- Faith does not regenerate passively ---

        # --- Hunger Depletion ---
        # All living things get hungrier over time.
        if self.peaceful_mode:
            # In peaceful ecology tests, hunger still moves but more slowly.
            self.hunger[self.is_alive_mask] = np.maximum(0, self.hunger[self.is_alive_mask] - 0.05)
        else:
            # Softer baseline depletion to allow time for growth/reproduction (further relaxed).
            self.hunger[self.is_alive_mask] = np.maximum(0, self.hunger[self.is_alive_mask] - 0.15)

        # --- Starvation ---
        # If hunger is 0, the cell starts losing HP.
        if not self.peaceful_mode:
            starvation_mask = self.is_alive_mask & (self.hunger <= 0)
            if np.any(starvation_mask):
                # Starvation damage tuned to 2.0 HP per step to match
                # expectations in world_simulation tests while remaining
                # a soft field effect at simulation time scales.
                self.hp[starvation_mask] -= 2.0
                # --- Leave an experience scar ---
                self.experience_scars[starvation_mask] |= 1 # Set the first bit for starvation

        # --- Hydration Depletion ---
        if self.peaceful_mode:
            self.hydration[self.is_alive_mask] = np.maximum(0, self.hydration[self.is_alive_mask] - 0.1)
        else:
            # Softer baseline water loss (further relaxed).
            self.hydration[self.is_alive_mask] = np.maximum(0, self.hydration[self.is_alive_mask] - 0.1)

        # --- Dehydration ---
        if not self.peaceful_mode:
            dehydration_mask = self.is_alive_mask & (self.hydration <= 0)
            self.hp[dehydration_mask] -= 0.5  # lighter penalty

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

    def _apply_survival_pain(self) -> None:
        """Inject pain signals: hunger drives aggression, isolation erodes will."""
        if len(self.cell_ids) == 0:
            return
        alive_idx = np.where(self.is_alive_mask)[0]
        if alive_idx.size == 0:
            return

        # Hunger pain: low hunger reduces coherence and boosts threat imprint around self.
        low_hunger = alive_idx[self.hunger[alive_idx] < 40]
        if low_hunger.size > 0:
            px = np.clip(self.positions[low_hunger, 0].astype(np.int32), 0, self.width - 1)
            py = np.clip(self.positions[low_hunger, 1].astype(np.int32), 0, self.width - 1)
            try:
                self.coherence_field[py, px] *= 0.7
                self.threat_field[py, px] += 0.6
            except Exception:
                pass

    def _apply_body_heat_law(self) -> None:
        """
        Cold-world rule: alone = freeze (burn resources fast), together = share heat/heal.
        This nudges emergent clustering/cooperation without hard alliances.
        """
        if len(self.cell_ids) == 0:
            return
        alive_idx = np.where(self.is_alive_mask)[0]
        if alive_idx.size == 0:
            return

        # Bin positions to avoid O(N^2) neighbor search.
        pos = self.positions[alive_idx, :2]
        cell_size = 8.0
        bins = np.floor(pos / cell_size).astype(int)
        bucket = defaultdict(list)
        for local_i, b in enumerate(bins):
            bucket[(b[0], b[1])].append(local_i)

        neighbor_counts = np.zeros(alive_idx.size, dtype=np.int32)
        for key, lst in bucket.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = bucket.get((key[0] + dx, key[1] + dy))
                    if nb:
                        for idx in lst:
                            neighbor_counts[idx] += len(nb)
            # remove self counts
            neighbor_counts[lst] -= 1

        # Alone or paired: burn through hunger/hydration (freeze)
        solitary_mask = neighbor_counts < 2
        if np.any(solitary_mask):
            idx_sol = alive_idx[solitary_mask]
            self.hunger[idx_sol] = np.maximum(0.0, self.hunger[idx_sol] - 3.0)
            self.hydration[idx_sol] = np.maximum(0.0, self.hydration[idx_sol] - 2.0)
            # Coherence drops in the cold
            try:
                px = np.clip(self.positions[idx_sol, 0].astype(np.int32), 0, self.width - 1)
                py = np.clip(self.positions[idx_sol, 1].astype(np.int32), 0, self.width - 1)
                self.coherence_field[py, px] *= 0.8
            except Exception:
                pass

    def _apply_gravity_pulse(self) -> None:
        """
        Pressure-cooker gravity: every cycle, yank everyone toward the center for a short burst
        to spike interaction density (crunch), then release.
        """
        if len(self.cell_ids) == 0:
            return
        if self.gravity_pulse_period <= 0 or self.gravity_pulse_strength <= 0:
            return
        phase = self.time_step % self.gravity_pulse_period
        if phase >= self.gravity_pulse_on_duration:
            return  # release phase

        alive_idx = np.where(self.is_alive_mask)[0]
        if alive_idx.size == 0:
            return

        center = np.array([self.width * 0.5, self.width * 0.5], dtype=np.float32)
        pos = self.positions[alive_idx, :2]
        vec = center - pos
        dist = np.linalg.norm(vec, axis=1) + 1e-3
        direction = vec / dist[:, None]
        # Strong pull that softens with distance to avoid infinite jumps
        pull = (self.gravity_pulse_strength / np.maximum(1.0, np.sqrt(dist)))[:, None]
        delta = direction * pull * 0.2  # scaled step toward center
        self.positions[alive_idx, :2] += delta
        # Clamp to world bounds
        self.positions[alive_idx, 0] = np.clip(self.positions[alive_idx, 0], 0, self.width - 1)
        self.positions[alive_idx, 1] = np.clip(self.positions[alive_idx, 1], 0, self.width - 1)

        # Log only at the start of a crunch to avoid log spam
        if phase == 0:
            try:
                self.event_logger.log('GRAVITY_PULSE', self.time_step,
                                      phase=phase, strength=float(self.gravity_pulse_strength))
            except Exception:
                pass

    def _apply_teaching(self) -> None:
        """
        Elders share knowledge with nearby young of the same species.
        Light-touch: no hard rules, just small boosts to insight/wisdom/satisfaction.
        """
        if len(self.cell_ids) == 0 or self.time_step % 5 != 0:
            return
        alive_idx = np.where(self.is_alive_mask)[0]
        if alive_idx.size == 0:
            return

        year_ticks = float(max(1, self._year_length_ticks()))
        ages_years = self.age[alive_idx] / year_ticks
        max_age_years = np.maximum(1.0, self.max_age[alive_idx] / year_ticks)

        elder_mask = (ages_years >= 0.4 * max_age_years) | (self.wisdom[alive_idx] >= 12)
        child_mask = ages_years <= (0.25 * max_age_years)

        elder_indices = alive_idx[elder_mask]
        child_indices = alive_idx[child_mask]
        if elder_indices.size == 0 or child_indices.size == 0:
            return

        # Spatial bucketing for proximity search
        pos = self.positions[alive_idx, :2]
        cell_size = 10.0
        bins = np.floor(pos / cell_size).astype(int)
        bucket = defaultdict(list)
        for local_i, b in enumerate(bins):
            bucket[(b[0], b[1])].append(local_i)

        # Limit processing to avoid O(N^2)
        max_children = min(len(child_indices), 64)
        child_indices = child_indices if len(child_indices) <= max_children else np.random.choice(child_indices, size=max_children, replace=False)

        radius2 = 12.0 ** 2
        labels_arr = self.labels
        for child_idx in child_indices:
            if not self.is_alive_mask[child_idx]:
                continue
            child_label = (labels_arr[child_idx] or "").lower()
            cx, cy = self.positions[child_idx, 0], self.positions[child_idx, 1]
            bx, by = int(np.floor(cx / cell_size)), int(np.floor(cy / cell_size))

            teacher_found = False
            teacher_idx = None
            teacher_age_years = 0.0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = bucket.get((bx + dx, by + dy))
                    if not nb:
                        continue
                    for local_i in nb:
                        global_idx = alive_idx[local_i]
                        if global_idx == child_idx:
                            continue
                        if not elder_mask[local_i]:
                            continue
                        if child_label and (labels_arr[global_idx] or "").lower() != child_label:
                            continue
                        dxp = self.positions[global_idx, 0] - cx
                        dyp = self.positions[global_idx, 1] - cy
                        if (dxp * dxp + dyp * dyp) <= radius2:
                            teacher_found = True
                            teacher_idx = global_idx
                            teacher_age_years = ages_years[local_i] if local_i < ages_years.shape[0] else 0.0
                            break
                if teacher_found:
                    break

        if teacher_found and teacher_idx is not None:
            # Apply knowledge transfer with elder bonus (age + wisdom)
            try:
                teacher_mask = (alive_idx == teacher_idx)
                teacher_max_age_years = float(max_age_years[teacher_mask][0]) if np.any(teacher_mask) else float(np.mean(max_age_years))
            except Exception:
                teacher_max_age_years = float(np.mean(max_age_years))
            teacher_age_frac = teacher_age_years / max(1e-6, teacher_max_age_years)
            teacher_wis = float(self.wisdom[teacher_idx])
            teacher_bonus = 1.0 + min(2.0, teacher_age_frac * 2.0 + (teacher_wis / 50.0))

            self.insight[child_idx] += 0.8 * teacher_bonus
            self.wisdom[child_idx] = min(200, self.wisdom[child_idx] + 2 * teacher_bonus)
            self.satisfaction[child_idx] = min(100.0, self.satisfaction[child_idx] + 1.5 * teacher_bonus)
            self.emotions[child_idx] = 'joy'

            try:
                self.event_logger.log(
                    "TEACH",
                    self.time_step,
                    teacher_id=self.cell_ids[teacher_idx],
                    student_id=self.cell_ids[child_idx],
                    kind="general",
                    bonus=teacher_bonus,
                )
            except Exception:
                pass

    def _apply_economy(self) -> None:
        """
        Simple wealth/prestige dynamics:
        - Trade/gift: rich -> poor nearby, prestige gains to giver.
        - Tribute: low-prestige pays nearby high-prestige leader.
        """
        if len(self.cell_ids) == 0 or self.time_step % 5 != 0:
            return
        alive_idx = np.where(self.is_alive_mask)[0]
        if alive_idx.size == 0:
            return

        pos = self.positions[alive_idx, :2]
        cell_size = 10.0
        bins = np.floor(pos / cell_size).astype(int)
        bucket = defaultdict(list)
        for local_i, b in enumerate(bins):
            bucket[(b[0], b[1])].append(local_i)

        for key, lst in bucket.items():
            if len(lst) < 2:
                continue
            local_indices = np.array(lst, dtype=int)
            global_indices = alive_idx[local_indices]
            local_wealth = self.wealth[global_indices]
            local_prestige = self.prestige[global_indices]

            # Trade/gift: richest to poorest
            rich_local = int(np.argmax(local_wealth))
            poor_local = int(np.argmin(local_wealth))
            if local_wealth[rich_local] - local_wealth[poor_local] > 4.0:
                amt = min(6.0, (local_wealth[rich_local] - local_wealth[poor_local]) * 0.6)
                giver = global_indices[rich_local]
                taker = global_indices[poor_local]
                self.wealth[giver] -= amt
                self.wealth[taker] += amt
                self.prestige[giver] += 0.5
                self.satisfaction[taker] = np.minimum(100.0, self.satisfaction[taker] + 2.0)
                try:
                    self.event_logger.log('TRADE', self.time_step,
                                          actor_id=self.cell_ids[giver],
                                          target_id=self.cell_ids[taker],
                                          amount=float(amt))
                except Exception:
                    pass

            # Tribute: low prestige pays local leader
            leader_local = int(np.argmax(local_prestige))
            follower_local = int(np.argmin(local_prestige))
            if local_prestige[leader_local] - local_prestige[follower_local] > 3.0 and self.wealth[global_indices[follower_local]] > 1.0:
                amt = min(4.0, self.wealth[global_indices[follower_local]] * 0.5)
                payer = global_indices[follower_local]
                leader = global_indices[leader_local]
                self.wealth[payer] -= amt
                self.wealth[leader] += amt
                self.prestige[leader] += 0.3
                self.satisfaction[payer] = np.maximum(0.0, self.satisfaction[payer] - 1.0)
                try:
                    self.event_logger.log('TRIBUTE', self.time_step,
                                          payer_id=self.cell_ids[payer],
                                          leader_id=self.cell_ids[leader],
                                          amount=float(amt))
                except Exception:
                    pass

    def _prune_connections(self, weight_threshold: float = 0.2, max_degree: int = 64) -> None:
        """Drop weak/low-value edges and cap per-node degree to keep graph light."""
        try:
            lil = self.adjacency_matrix.tolil()
            rows = lil.rows
            data = lil.data
            for i in range(len(rows)):
                if not rows[i]:
                    continue
                filtered = [(c, w) for c, w in zip(rows[i], data[i]) if w >= weight_threshold]
                if not filtered:
                    rows[i] = []
                    data[i] = []
                    continue
                filtered.sort(key=lambda item: item[1], reverse=True)
                filtered = filtered[:max_degree]
                rows[i] = [c for c, _ in filtered]
                data[i] = [w for _, w in filtered]
            self.adjacency_matrix = lil
        except Exception:
            # If pruning fails, do nothing to avoid breaking the sim.
            pass

    def harvest_snapshot(self, out_path: str = "logs/harvest_snapshot.json", max_edges: int = 5000) -> None:
        """
        Export a lightweight snapshot of surviving nodes/edges for long-term memory.
        Captures population summary and key stats; edges are truncated for size.
        """
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            alive_idx = np.where(self.is_alive_mask)[0]
            nodes = []
            for i in alive_idx:
                year_ticks = float(max(1, self._year_length_ticks()))
                nodes.append({
                    "id": self.cell_ids[i],
                    "label": self.labels[i],
                    "culture": self.culture[i] if i < len(self.culture) else "",
                    "gender": self.genders[i] if i < len(self.genders) else "",
                    "age_years": float(self.age[i] / year_ticks),
                    "hp": float(self.hp[i]),
                    "max_hp": float(self.max_hp[i]),
                    "hunger": float(self.hunger[i]),
                    "hydration": float(self.hydration[i]),
                    "wisdom": float(self.wisdom[i]),
                    "insight": float(self.insight[i]),
                    "position": {
                        "x": float(self.positions[i, 0]),
                        "y": float(self.positions[i, 1]),
                        "z": float(self.positions[i, 2]),
                    },
                })

            edges = []
            coo = self.adjacency_matrix.tocoo()
            for s, t, w in zip(coo.row, coo.col, coo.data):
                if s == t or w <= 0:
                    continue
                edges.append({
                    "source": self.cell_ids[s],
                    "target": self.cell_ids[t],
                    "weight": float(w),
                })
                if len(edges) >= max_edges:
                    break

            summary = {
                "time_step": int(self.time_step),
                "alive_count": int(alive_idx.size),
                "nodes": nodes,
                "edges": edges,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False)
        except Exception as e:
            # Harvest should never crash the sim
            try:
                logging.getLogger(__name__).error(f"Harvest failed: {e}")
            except Exception:
                pass


    def _update_emergent_fields(self):
        """Updates soft fields (e.g., threat) from distributed sources.
        The field is not a command; it is a context carrier that agents can sense.
        """
        # --- Field Decay (The "Dissipation of Waves") ---
        # Words and events create ripples, but they must fade to allow new stories.
        self.threat_field *= 0.95
        self.will_field *= 0.95
        self.value_mass_field *= 0.98 # Value/Meaning lasts longer than raw will
        self.norms_field *= 0.98      # Norms are sticky
        self.coherence_field *= 0.90  # Coherence is fragile
        self.hydration_field *= 0.90  # Water evaporates/diffuses

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

        # Build a fresh threat imprint from predators and historical dangers
        new_threat = np.zeros_like(self.threat_field)
        try:
            # Source 1: Predators -> weighted presence map (vectorized)
            predator_mask = (self.element_types == 'animal') & (self.diets == 'carnivore') & self.is_alive_mask
            predator_indices = np.where(predator_mask)[0]
            if predator_indices.size > 0:
                px = np.clip(self.positions[predator_indices, 0].astype(np.int32), 0, self.width - 1)
                py = np.clip(self.positions[predator_indices, 1].astype(np.int32), 0, self.width - 1)
                hunger_term = np.maximum(0.5, 1.5 - (self.hunger[predator_indices] / 100.0))
                strength_term = np.maximum(1.0, self.strength[predator_indices] / 10.0)
                weights = (self._threat_gain * hunger_term * strength_term).astype(np.float32)

                if self.use_fast_field_diffusion:
                    # Single gaussian_filter over the accumulated map (edge: minimal allocations)
                    presence = np.zeros_like(self.threat_field, dtype=np.float32)
                    np.add.at(presence, (py, px), weights)
                    try:
                        from scipy.ndimage import gaussian_filter
                        new_threat = gaussian_filter(presence, sigma=self._threat_sigma)
                    except Exception:
                        # Keep the raw presence map if scipy is unavailable; disable fast path to avoid repeated imports
                        new_threat = presence
                        self.use_fast_field_diffusion = False
                else:
                    # Fallback to kernel stamping (previous behavior)
                    sigma = self._threat_sigma
                    rad = int(max(2, sigma * 3))
                    for idx_local, weight in zip(predator_indices, weights):
                        px_i = int(self.positions[idx_local][0]) % self.width
                        py_i = int(self.positions[idx_local][1]) % self.width
                        x0, x1 = max(0, px_i - rad), min(self.width, px_i + rad + 1)
                        y0, y1 = max(0, py_i - rad), min(self.width, py_i + rad + 1)
                        xs = np.arange(x0, x1) - px_i
                        ys = np.arange(y0, y1) - py_i
                        gx = np.exp(-(xs**2) / (2 * sigma * sigma))
                        gy = np.exp(-(ys**2) / (2 * sigma * sigma))
                        patch = (gy[:, None] * gx[None, :]).astype(np.float32)
                        new_threat[y0:y1, x0:x1] += weight * patch

            # Source 2: Historical Imprints of Death
            # Places with a history of death should feel ominous.
            # We add the h_imprint field as a source of threat, scaled by a factor.
            # The 0.3 scaling factor makes it a weaker, more atmospheric threat than an active predator.
            if self.h_imprint.size > 0 and np.any(self.h_imprint):
                new_threat += self.h_imprint * 0.3

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
            # Prefer tensor_field identity channel; fallback to value_mass scalar
            source = self.tensor_field[..., 2] if self.tensor_field.size > 0 else self.value_mass_field

            # The gradient returns derivatives along each axis (dy, dx)
            grad_y, grad_x = np.gradient(source)

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

    def _update_tensor_field(self) -> None:
        """Unify scalar fields + cell SoulTensors into a 3-channel tensor and cache its gradients."""
        try:
            tf = self.tensor_field
            # Channel 0: structure, 1: emotion, 2: identity/meaning
            tf[..., 0] = (self.norms_field + self.prestige_field) * 0.5
            tf[..., 1] = self.coherence_field - self.threat_field
            tf[..., 2] = (self.value_mass_field + self.will_field) * 0.5

            # Inject per-cell SoulTensor projections (entanglement bumps weight)
            alive_idx = np.where(self.is_alive_mask)[0]
            if alive_idx.size > 0:
                contrib = np.zeros_like(tf)
                for idx in alive_idx:
                    try:
                        x = int(np.clip(self.positions[idx, 0], 0, self.width - 1))
                        y = int(np.clip(self.positions[idx, 1], 0, self.width - 1))
                    except Exception:
                        continue

                    # Pull SoulTensor if present; fallback to simple stats-derived tensor
                    tensor_state = None
                    if hasattr(self.wave_mechanics, "get_node_tensor"):
                        try:
                            tensor_state = self.wave_mechanics.get_node_tensor(self.cell_ids[idx])
                        except Exception:
                            tensor_state = None

                    if tensor_state and hasattr(tensor_state, "space"):
                        space = tensor_state.space
                        weight = 1.5 if getattr(tensor_state, "entanglement_id", None) else 1.0
                        vec = np.array([space.x, space.y, space.z], dtype=np.float32) * weight
                    else:
                        structure_val = (self.strength[idx] + self.wisdom[idx]) / 200.0
                        emotion_val = 0.5
                        if self.emotions[idx] == 'joy':
                            emotion_val = 0.8
                        elif self.emotions[idx] == 'sorrow':
                            emotion_val = 0.2
                        identity_val = self.insight[idx] / 10.0
                        vec = np.array([structure_val, emotion_val, identity_val], dtype=np.float32)

                    contrib[y, x] += vec

                tf += contrib

            # Central differences for gradients (x: axis=1, y: axis=0)
            gx = self.tensor_field_grad_x
            gy = self.tensor_field_grad_y
            gx.fill(0.0)
            gy.fill(0.0)
            gx[:, 1:-1, :] = (tf[:, 2:, :] - tf[:, :-2, :]) * 0.5
            gy[1:-1, :, :] = (tf[2:, :, :] - tf[:-2, :, :]) * 0.5
        except Exception:
            # Tensor field should never break the simulation loop
            pass

    def _resonance_gradient(self, tensor: Tensor3D, px: float, py: float) -> np.ndarray:
        """Return the gradient of dot(env_tensor, tensor) at a world position."""
        if self.tensor_field_grad_x.size == 0 or self.tensor_field_grad_y.size == 0:
            return np.zeros(2, dtype=np.float32)
        x = int(np.clip(px, 0, self.width - 1))
        y = int(np.clip(py, 0, self.width - 1))
        vec = np.array([tensor.x, tensor.y, tensor.z], dtype=np.float32)
        gx = float(np.dot(self.tensor_field_grad_x[y, x], vec))
        gy = float(np.dot(self.tensor_field_grad_y[y, x], vec))
        return np.array([gx, gy], dtype=np.float32)

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
        """
        [MIND LAYER] The Act of Imprinting (2D Flow).
        Applies a Gaussian 'wave' of influence onto a Spirit Field.
        This connects the Event (Point) to the Atmosphere (Space).
        """
        try:
            # Ensure coordinates are within bounds
            if not (0 <= x < self.width and 0 <= y < self.width):
                return

            rad = int(max(2, sigma * 3))
            x0, x1 = max(0, x - rad), min(self.width, x + rad + 1)
            y0, y1 = max(0, y - rad), min(self.width, y + rad + 1)

            # Generate grid for the kernel
            xs = np.arange(x0, x1) - x
            ys = np.arange(y0, y1) - y
            gx = np.exp(-(xs**2) / (2 * sigma * sigma))
            gy = np.exp(-(ys**2) / (2 * sigma * sigma))

            # Outer product to make 2D kernel
            patch = (gy[:, None] * gx[None, :]).astype(np.float32)

            # Add to target field
            target[y0:y1, x0:x1] += amplitude * patch
        except Exception as e:
            # self.logger.error(f"Failed to imprint gaussian: {e}")
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

                # --- The Law of Resonance: Attracted to harmonic fields ---
                resonance_force = np.zeros(3, dtype=np.float32)
                # Calculate internal tensor state (simplified from stats for now)
                structure_val = (self.strength[i] + self.wisdom[i]) / 200.0
                emotion_val = 0.5
                if self.emotions[i] == 'joy': emotion_val = 0.8
                elif self.emotions[i] == 'sorrow': emotion_val = 0.2
                identity_val = self.insight[i] / 10.0

                my_tensor = Tensor3D(structure_val, emotion_val, identity_val)

                # Tensor-based resonance gradient (environment tensor · my_tensor)
                grad_res = self._resonance_gradient(my_tensor, px, py)
                resonance_strength = 0.3 # Tuning parameter
                resonance_force = np.array([grad_res[0], grad_res[1], 0.0], dtype=np.float32) * resonance_strength

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

                # --- Z-Axis Movement (Ascension/Descent) ---
                px_i = int(np.clip(px, 0, self.width - 1))
                py_i = int(np.clip(py, 0, self.width - 1))
                ascension_force = np.sum(self.ascension_field[py_i, px_i, :])
                descent_force = np.sum(self.descent_field[py_i, px_i, :])
                z_movement = (ascension_force - descent_force) * 0.1 # Subtle vertical movement

                # Combine forces: Resonance is added to the movement vector
                movement_vectors[local_idx] += other_forces + intention_force + resonance_force
                movement_vectors[local_idx, 2] += z_movement


                # --- Observe and Record Meaningful Choice ---
                # Check if resonance or intention dominates biological drives
                if np.linalg.norm(resonance_force + intention_force) > np.linalg.norm(other_forces):
                    # Only log occasionally to avoid spam
                    if random.random() < 0.05:
                        self.logger.info(f"RESONANCE: Cell '{self.cell_ids[i]}' moves towards spiritual alignment.")

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

    def _build_law_manager(self) -> LawPriorityManager:
        policies = [
            LawPolicy(
                name="altruism",
                priority=10.0,
                condition=self._altruism_condition,
                action=self._altruism_action,
                insight=self._altruism_insight,
            ),
            LawPolicy(
                name="mindful_presence",
                priority=5.0,
                condition=self._mindful_condition,
                action=self._mindful_action,
                insight=self._mindful_insight,
            ),
            LawPolicy(
                name="resonance_field",
                priority=4.0,
                condition=self._resonance_condition,
                action=self._resonance_action,
                insight=self._resonance_insight,
            ),
            LawPolicy(
                name="memory_resonance",
                priority=3.0,
                condition=self._memory_condition,
                action=self._memory_action,
                insight=self._memory_insight,
            ),
            LawPolicy(
                name="vision_aura",
                priority=3.5,
                condition=self._vision_condition,
                action=self._vision_action,
                insight=self._vision_insight,
            ),
            LawPolicy(
                name="hearing_harmony",
                priority=3.0,
                condition=self._hearing_condition,
                action=self._hearing_action,
                insight=self._hearing_insight,
            ),
            LawPolicy(
                name="taste_lore",
                priority=2.5,
                condition=self._taste_condition,
                action=self._taste_action,
                insight=self._taste_insight,
            ),
            LawPolicy(
                name="smell_cleansing",
                priority=2.7,
                condition=self._smell_condition,
                action=self._smell_action,
                insight=self._smell_insight,
            ),
            LawPolicy(
                name="touch_temperature",
                priority=2.3,
                condition=self._touch_condition,
                action=self._touch_action,
                insight=self._touch_insight,
            ),
            LawPolicy(
                name="imagination_bloom",
                priority=2.0,
                condition=self._imagination_condition,
                action=self._imagination_action,
                insight=self._imagination_insight,
            ),
            LawPolicy(
                name="solidarity_pulse",
                priority=1.0,
                condition=self._solidarity_condition,
                action=self._solidarity_action,
                insight=self._solidarity_insight,
            ),
        ]
        return LawPriorityManager(self, policies)

    def _altruism_condition(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> bool:
        return self.hunger[actor_idx] > 70 and connected_indices.size > 0

    def _altruism_action(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> LawAction:
        kin_mask = (
            (self.hunger[connected_indices] < 30)
            & (self.labels[connected_indices] == self.labels[actor_idx])
            & self.is_alive_mask[connected_indices]
        )
        hungry_kin = connected_indices[kin_mask]
        if hungry_kin.size == 0:
            return None
        best_target = -1
        for target_idx in hungry_kin:
            if adj_matrix_csr[actor_idx, target_idx] >= 0.8:
                best_target = target_idx
                break
        if best_target == -1:
            best_target = hungry_kin[0]
        return best_target, 'share_food', None

    def _altruism_insight(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> List[str]:
        actor_name = self.cell_ids[actor_idx] if actor_idx < len(self.cell_ids) else f"cell_{actor_idx}"
        return [
            f"{actor_name} senses abundance and the presence of kin; sharing feels like honoring that bond.",
            "The cosmic axis of love is stronger here than the field of death."
        ]

    def _mindful_condition(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> bool:
        return self.emotions[actor_idx] == 'sorrow' and not self.is_meditating[actor_idx]

    def _mindful_action(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> LawAction:
        return -1, 'meditate', None

    def _mindful_insight(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> List[str]:
        actor_name = self.cell_ids[actor_idx] if actor_idx < len(self.cell_ids) else f"cell_{actor_idx}"
        return [
            f"{actor_name}'s sorrow is a signal to pause; meditation will let the field settle.",
        ]

    def _resonance_condition(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> bool:
        return self.satisfaction[actor_idx] > 80 and self.hunger[actor_idx] >= 40

    def _resonance_action(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> LawAction:
        return None

    def _resonance_insight(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> List[str]:
        actor_name = self.cell_ids[actor_idx] if actor_idx < len(self.cell_ids) else f"cell_{actor_idx}"
        return [
            f"{actor_name} feels resonance in the field; the chorus of sustaining connections keeps the intent steady.",
        ]

    def _vision_condition(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> bool:
        return self.vision_awareness[actor_idx] > 40.0

    def _vision_action(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> LawAction:
        return None

    def _vision_insight(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> List[str]:
        return [f"Vision field sharp; {self.cell_ids[actor_idx]} paints light as intent."] * 2

    def _hearing_condition(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> bool:
        return self.auditory_clarity[actor_idx] > 35.0

    def _hearing_action(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> LawAction:
        return None

    def _hearing_insight(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> List[str]:
        return [f"Harmony listened: sounds align with the rhythm of {self.cell_ids[actor_idx]}."]

    def _taste_condition(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> bool:
        return self.gustatory_imbue[actor_idx] > 20.0

    def _taste_action(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> LawAction:
        return None

    def _taste_insight(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> List[str]:
        return [f"Taste lore whispers about nourishment; {self.cell_ids[actor_idx]} invites a savory ritual."]

    def _smell_condition(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> bool:
        return self.olfactory_sensitivity[actor_idx] > 50.0

    def _smell_action(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> LawAction:
        return None

    def _smell_insight(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> List[str]:
        return [f"Smell calls for cleansing; {self.cell_ids[actor_idx]} imagines washrooms and freshness."]

    def _touch_condition(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> bool:
        return self.tactile_feedback[actor_idx] > 30.0

    def _touch_action(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> LawAction:
        return None

    def _touch_insight(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> List[str]:
        return [f"Touch senses warmth/cold interplay; {self.cell_ids[actor_idx]} ponders textures as stories."]

    def _memory_condition(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> bool:
        return self.age[actor_idx] >= 5

    def _memory_action(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> LawAction:
        return None

    def _memory_insight(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> List[str]:
        actor_name = self.cell_ids[actor_idx] if actor_idx < len(self.cell_ids) else f"cell_{actor_idx}"
        return [
            f"{actor_name} recalls past shared feasts; the memory weight keeps the group tethered.",
        ]

    def _imagination_condition(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> bool:
        return self.emotions[actor_idx] == "joy" or self.satisfaction[actor_idx] > 80

    def _imagination_action(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> LawAction:
        return None

    def _imagination_insight(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> List[str]:
        actor_name = self.cell_ids[actor_idx] if actor_idx < len(self.cell_ids) else f"cell_{actor_idx}"
        return [
            f"{actor_name} lets imagination of future gardens color the current choices.",
        ]

    def _solidarity_condition(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> bool:
        return connected_indices.size >= 1 and np.any(self.labels[connected_indices] == self.labels[actor_idx])

    def _solidarity_action(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> LawAction:
        return None

    def _solidarity_insight(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> List[str]:
        actor_name = self.cell_ids[actor_idx] if actor_idx < len(self.cell_ids) else f"cell_{actor_idx}"
        return [
            f"{actor_name} senses the solidarity field; every sharing ripple strengthens the kin circle.",
        ]

    def _reflective_questions(self, actor_idx: int, adj_matrix_csr: csr_matrix, connected_indices: np.ndarray) -> List[str]:
        """Generate a brief set of self-inquiry prompts (value, relation, emotion)."""
        actor_name = (
            self.cell_ids[actor_idx]
            if actor_idx < len(self.cell_ids)
            else f"cell_{actor_idx}"
        )
        culture = self.culture[actor_idx] if actor_idx < len(self.culture) else "unknown"
        emotion = self.emotions[actor_idx] if actor_idx < len(self.emotions) else "neutral"

        question_value = (
            f"{actor_name} follows the path of {culture}; what responsibility does that call you to take now?"
        )

        if connected_indices.size == 0:
            relation_text = f"{actor_name} is not connected to anyone right now."
        else:
            strengths = adj_matrix_csr[
                actor_idx, connected_indices
            ].toarray().flatten()
            if strengths.size > 0:
                best_idx = connected_indices[int(np.argmax(strengths))]
                best_name = (
                    self.cell_ids[best_idx]
                    if best_idx < len(self.cell_ids)
                    else f"cell_{best_idx}"
                )
                relation_text = (
                    f"Who does {actor_name} wish to maintain a bond with in this moment with {best_name}?"
                )
            else:
                relation_text = f"{actor_name} is connected but the bond has no name."

        kin_mask = np.array([], dtype=bool)
        if connected_indices.size > 0:
            kin_mask = (
                (self.labels[connected_indices] == self.labels[actor_idx])
                & self.is_alive_mask[connected_indices]
            )
        kin_count = int(np.count_nonzero(kin_mask))
        relation_question = (
            f"{relation_text} ({kin_count} kin present at this scene)"
            if kin_count
            else relation_text
        )

        emotion_question = (
            f"{actor_name} feels '{emotion}'. What action would honor that feeling right now?"
        )

        questions = [question_value, relation_question, emotion_question]
        self.logger.debug(f"Reflective questions for {actor_name}: {questions}")
        return questions

    def _determine_meta_focus(self, actor_idx: int, connected_indices: np.ndarray) -> None:
        hunger = float(self.hunger[actor_idx]) if self.hunger.size > actor_idx else 0.0
        satisfaction = float(self.satisfaction[actor_idx]) if self.satisfaction.size > actor_idx else 0.0
        if hunger < 30:
            focus = "survival"
        elif satisfaction > 80:
            focus = "resonance"
        elif connected_indices.size >= 3:
            focus = "solidarity"
        else:
            focus = "baseline"
        self.meta_focus = focus
        self.meta_focus_history.append(focus)

    def get_meta_priority_bonus(self, policy_name: str) -> float:
        focus = self.meta_focus
        bonus = 0.0
        if focus == "resonance" and policy_name in {"resonance_field", "imagination_bloom"}:
            bonus = 1.5
        elif focus == "solidarity" and policy_name in {"solidarity_pulse", "altruism"}:
            bonus = 1.0
        elif focus == "survival" and policy_name == "mindful_presence":
            bonus = 0.5
        return bonus

    def _update_channels(self, actor_idx: int, reflections: List[str]) -> None:
        mem_delta = sum(1 for text in reflections if "memory" in text.lower())
        imag_delta = sum(1 for text in reflections if "imagination" in text.lower())
        emotion_val = {"joy": 0.6, "calm": 0.2, "sorrow": -0.3, "neutral": 0.0}.get(
            self.emotions[actor_idx] if actor_idx < len(self.emotions) else "neutral", 0.0
        )
        self.memory_strength[actor_idx] = min(100.0, self.memory_strength[actor_idx] * 0.9 + mem_delta * 5 + 0.1)
        self.imagination_brightness[actor_idx] = min(100.0, self.imagination_brightness[actor_idx] * 0.8 + imag_delta * 4 + 0.1)
        self.emotion_intensity[actor_idx] = min(100.0, self.emotion_intensity[actor_idx] * 0.9 + abs(emotion_val) * 20)
        self.vision_awareness[actor_idx] = min(100.0, self.vision_awareness[actor_idx] + len(reflections))
        self.auditory_clarity[actor_idx] = min(100.0, self.auditory_clarity[actor_idx] + 0.5 * len(reflections))
        self.gustatory_imbue[actor_idx] = min(100.0, self.gustatory_imbue[actor_idx] + 0.3 * len(reflections))
        self.olfactory_sensitivity[actor_idx] = min(100.0, self.olfactory_sensitivity[actor_idx] + 0.4 * len(reflections))
        self.tactile_feedback[actor_idx] = min(100.0, self.tactile_feedback[actor_idx] + 0.2 * len(reflections))

    def apply_quaternion_feedback(self, actor_idx: int, q: Quaternion) -> None:
        scalar = float(q.scalar)
        vector = np.array(q.vector, dtype=np.float32)
        self.memory_strength[actor_idx] = min(100.0, self.memory_strength[actor_idx] + scalar * 5.0)
        self.imagination_brightness[actor_idx] = min(100.0, self.imagination_brightness[actor_idx] + np.mean(np.abs(vector)) * 10.0)
        self.emotion_intensity[actor_idx] = min(100.0, self.emotion_intensity[actor_idx] + abs(vector.max()) * 5.0)

    def gather_fractal_field_insights(self, actor_idx: int) -> List[str]:
        layers = ["intention", "resonance", "memory"]
        adj_csr = self.adjacency_matrix.tocsr()
        insights: List[str] = []
        connected_indices = adj_csr[actor_idx].indices
        for x in layers:
            for y in layers:
                for z in layers:
                    focus_combo = f"{x}/{y}/{z}"
                    self.meta_focus = focus_combo
                    self.meta_focus_history.append(focus_combo)
                    questions = self._reflective_questions(actor_idx, adj_csr, connected_indices)
                    summary = "; ".join(questions[:2])
                    insights.append(f"[{focus_combo}] {summary}")
        return insights

    def _decide_social_or_combat_action(self, actor_idx: int, adj_matrix_csr: csr_matrix) -> Optional[Tuple[Optional[int], str, Optional[Move]]]:
        """
        Handles hunting, fighting, and social behaviors using a vectorized scoring system for performance.
        All possible actions for all neighbors are scored simultaneously using NumPy operations.
        """
        connected_indices = adj_matrix_csr[actor_idx].indices

        # --- Pre-computation and Law Evaluation ---
        self._determine_meta_focus(actor_idx, connected_indices)
        base_questions = self._reflective_questions(actor_idx, adj_matrix_csr, connected_indices)
        reflections_for_channels = base_questions
        evaluation = self.law_manager.evaluate(actor_idx, adj_matrix_csr, connected_indices)
        if evaluation:
            reflections_for_channels.extend(evaluation.reflections)
            self.last_reflections[actor_idx] = reflections_for_channels
            self._update_channels(actor_idx, reflections_for_channels)
            if evaluation.action is not None:
                return evaluation.action

        if connected_indices.size == 0:
            if not evaluation:
                self.last_reflections[actor_idx] = reflections_for_channels
                self._update_channels(actor_idx, reflections_for_channels)
            return None, 'idle', None

        if not evaluation:
            self.last_reflections[actor_idx] = reflections_for_channels
            self._update_channels(actor_idx, reflections_for_channels)

        # --- 1. Vectorized Target Filtering ---
        alive_mask = self.is_alive_mask[connected_indices]
        neighbors = connected_indices[alive_mask]
        if neighbors.size == 0:
            return None, 'idle', None

        neighbor_elements = self.element_types[neighbors]
        is_plant = neighbor_elements == 'life'
        is_animal = neighbor_elements == 'animal'
        is_hungry = self.hunger[neighbors] < 30
        is_same_species = self.labels[neighbors] == self.labels[actor_idx]

        # --- 2. Actor & Environmental State ---
        actor_hunger = self.hunger[actor_idx]
        actor_wisdom = self.wisdom[actor_idx]
        actor_satisfaction = self.satisfaction[actor_idx]
        actor_scars = self.experience_scars[actor_idx]
        has_starvation_scar = (actor_scars & 1) > 0
        has_charity_scar = (actor_scars & 2) > 0

        actor_pos = self.positions[actor_idx].astype(int)
        px, py = np.clip(actor_pos[0], 0, self.width - 1), np.clip(actor_pos[1], 0, self.width - 1)
        local_life = float(self.ascension_field[py, px, ASCENSION_LIFE])
        local_death = float(self.descent_field[py, px, DESCENT_DEATH])
        local_threat = float(self.threat_field[py, px])
        local_coherence = float(self.coherence_field[py, px])
        local_will = float(self.will_field[py, px])
        local_value = float(self.value_mass_field[py, px])

        # Tiny perceptron preference: how much does this cell "feel like"
        # eating in the current local environment?
        perceptron_boost = 0.0
        try:
            cell_id = self.cell_ids[actor_idx]
            cell_obj = self.materialize_cell(cell_id)
            if cell_obj is not None:
                features = {
                    "value_mass": local_value,
                    "threat": local_threat,
                }
                perceptron_boost = cell_obj.perceptron_output(features)
        except Exception:
            perceptron_boost = 0.0

        # --- 3. Vectorized Score Calculation ---
        # Initialize score arrays for each action type
        share_scores = np.full(neighbors.shape, -np.inf, dtype=np.float32)
        eat_scores = np.full(neighbors.shape, -np.inf, dtype=np.float32)
        attack_scores = np.full(neighbors.shape, -np.inf, dtype=np.float32)
        talk_scores = np.full(neighbors.shape, -np.inf, dtype=np.float32)

        # --- Share Food Scores ---
        if actor_hunger > 70 and np.any(is_hungry):
            eligible_targets_mask = is_hungry
            base_score = 100 + actor_wisdom * 1.5 + actor_satisfaction
            field_score = local_life * 50 + local_value * 20 + local_coherence * 30
            kin_bonus = (adj_matrix_csr[actor_idx, neighbors].toarray().flatten() >= 0.8) * 50
            charity_bonus = has_charity_scar * 100
            share_scores[eligible_targets_mask] = base_score + field_score + kin_bonus[eligible_targets_mask] + charity_bonus

        # --- Eat Scores ---
        if np.any(is_plant) and self.diets[actor_idx] in ['herbivore', 'omnivore']:
            hunger_drive = (100 - actor_hunger) * 0.8 - local_threat * 10
            # Perceptron boost acts like a learned "craving" or aversion
            # for eating in the current field conditions.
            eat_scores[is_plant] = hunger_drive + perceptron_boost

        # --- Attack Scores ---
        if np.any(is_animal) and self.diets[actor_idx] in ['carnivore', 'omnivore']:
            hunger_drive = 100 - actor_hunger
            hp_ratio = self.hp[neighbors] / self.max_hp[neighbors]
            hp_factor = (1 - hp_ratio) * 20
            field_score = local_death * 50 + local_threat * 30 + local_will * 20 - local_life * 20 - local_coherence * 40
            starvation_bonus = (has_starvation_scar and actor_hunger < 50) * 50

            attack_scores[is_animal] = hunger_drive + hp_factor[is_animal] + field_score + starvation_bonus
            attack_scores[is_animal & is_same_species] = -1000  # Cannibalism taboo

        # --- Talk Scores ---
        relation_strengths = adj_matrix_csr[actor_idx, neighbors].toarray().flatten()
        base_score = 6.0 - (actor_hunger < 40) * 3.0 - (self.is_injured[actor_idx]) * 1.0
        actor_emotion = self.emotions[actor_idx]
        if actor_emotion in ('joy', 'calm'): base_score += 3.0
        elif actor_emotion == 'sorrow': base_score += 2.0
        else: base_score += 1.0
        field_score = local_life * 10.0 + local_coherence * 20.0 + local_value * 10.0 - local_death * 5.0 - local_threat * 5.0
        talk_scores[:] = base_score + relation_strengths * 4.0 + field_score

        # --- 4. Find Best Action Across All Types ---
        best_score = 5.0  # Idle score
        best_action = 'idle'
        best_target_idx = None

        if share_scores.max() > best_score:
            best_score = share_scores.max()
            best_action = 'share_food'
            best_target_idx = neighbors[np.argmax(share_scores)]

        if eat_scores.max() > best_score:
            best_score = eat_scores.max()
            best_action = 'eat'
            best_target_idx = neighbors[np.argmax(eat_scores)]

        if attack_scores.max() > best_score:
            best_score = attack_scores.max()
            best_action = 'attack'
            best_target_idx = neighbors[np.argmax(attack_scores)]

        if talk_scores.max() > best_score:
            best_score = talk_scores.max()
            best_action = 'talk'
            best_target_idx = neighbors[np.argmax(talk_scores)]

        # --- 5. Return the chosen action ---
        self._last_action_intent = best_action
        return best_target_idx, best_action, None

    def _learn_from_resonance(self, actor_idx: int, action: str, outcome: str, magnitude: float):
        """
        [ELYSIA'S MEMORY] The world learns which words hold weight.
        If an action (e.g., attack) was successful and matched the intent of recent speech,
        the 'mass' of that word increases in the global lexicon.
        """
        # Simple heuristic: Check if the actor spoke recently about this intent
        # In a full system, we'd track individual speech history.
        # Here, we assume the 'will_field' or 'threat_field' that drove the action
        # was partly constituted by the actor's own words or the words of others.

        # Map action to relevant speech keys
        action_to_speech = {
            "attack": ["ATTACK", "SKILL_ATTACK", "SPELL_FIRE", "TALK_THREAT"],
            "share_food": ["TALK_TRADE", "TALK_CELEBRATE", "TALK_BEG"],
            "cast_heal": ["SPELL_HEAL", "TALK_COMFORT"],
            "talk": ["TALK_BOND", "TALK_COMFORT", "TALK_TRADE"],
        }

        related_keys = action_to_speech.get(action, [])
        for key in related_keys:
            # Increase mastery. The world "remembers" that this word works.
            # Growth is logarithmic to prevent explosion.
            current_weight = self.lexicon_mastery[key]
            growth = 0.01 * magnitude / current_weight # Diminishing returns
            self.lexicon_mastery[key] += growth
            if growth > 0.001:
                # self.logger.debug(f"LEARNING: '{key}' mastery increased to {self.lexicon_mastery[key]:.3f} via {action} ({outcome})")
                pass

    def _execute_animal_action(self, actor_idx: int, target_idx: int, action: str, move: Optional[Move]):
        """Executes the chosen action, including non-target actions like meditation."""
        # --- Genesis Protocol Execution ---
        if action.startswith('genesis:'):
             # Extract 'action:fire_punch' from 'genesis:action:fire_punch'
             # The action_id is everything after 'genesis:'
             action_id = action[len('genesis:'):]
             self.genesis_engine.execute_action(actor_idx, action_id, target_idx)
             return

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

        if action == 'talk' and target_idx != -1:
            try:
                speaker_id = self.cell_ids[actor_idx]
                listener_id = self.cell_ids[target_idx]
                relation_strength = float(self.adjacency_matrix[actor_idx, target_idx])
                speaker_emotion = (
                    self.emotions[actor_idx] if actor_idx < len(self.emotions) else "neutral"
                )
                listener_emotion = (
                    self.emotions[target_idx] if target_idx < len(self.emotions) else "neutral"
                )
            except Exception:
                speaker_id = f"cell_{actor_idx}"
                listener_id = f"cell_{target_idx}"
                relation_strength = 0.0
                speaker_emotion = "neutral"
                listener_emotion = "neutral"

            # Heuristic topic hint: not a template, just a label for later interpretation.
            topic = "bonding"
            if speaker_emotion == "sorrow" or listener_emotion == "sorrow":
                topic = "comfort"
            elif speaker_emotion == "joy" or listener_emotion == "joy":
                topic = "celebration"

            # Choose a speech act and generate a line
            speech_act = "bonding"
            if topic == "comfort":
                speech_act = "comfort"
            elif topic == "celebration":
                speech_act = "celebrate"
            else:
                # basic heuristics: hungry -> begging; low relation -> threat; wealth gap -> trade
                if self.hunger[actor_idx] < 35 or self.hydration[actor_idx] < 35:
                    speech_act = "beg"
                elif relation_strength < 0.2:
                    speech_act = "threat"
                else:
                    # mild trade request if wealth gap visible
                    try:
                        if abs(self.wealth[actor_idx] - self.wealth[target_idx]) > 5.0:
                            speech_act = "trade"
                    except Exception:
                        pass

            # Tarzan 3-slot pattern + optional template flavor
            obj_word = "FRIEND"
            if speech_act in ("beg", "trade"):
                obj_word = "FOOD"
            elif speech_act == "threat":
                obj_word = "TERRITORY"
            elif speech_act == "comfort":
                obj_word = "PAIN"
            tarzan_pattern = f"{(self.labels[actor_idx] or 'ME').upper()} {speech_act.upper()} {obj_word}"
            # Memetic reuse: favor strong existing pattern for this speech_act
            existing = [(k, v) for k, v in self.meme_bank.items() if k[0] == speech_act]
            if existing and random.random() < 0.5:
                best = max(existing, key=lambda kv: kv[1])[0]
                tarzan_pattern = f"{(self.labels[actor_idx] or 'ME').upper()} {best[0].upper()} {best[1].upper()}"

            text_key_map = {
                "bonding": "TALK_BOND",
                "comfort": "TALK_COMFORT",
                "celebrate": "TALK_CELEBRATE",
                "beg": "TALK_BEG",
                "trade": "TALK_TRADE",
                "threat": "TALK_THREAT",
            }
            natural_text = kr_dialogue(text_key_map.get(speech_act, "TALK_BOND"))
            text = natural_text if natural_text else tarzan_pattern

            # Relation nudges and small costs/rewards
            if speech_act == "comfort":
                relation_strength = min(1.0, relation_strength + 0.05)
                self.satisfaction[target_idx] = min(100.0, self.satisfaction[target_idx] + 1.0)
            elif speech_act == "celebrate":
                relation_strength = min(1.0, relation_strength + 0.02)
            elif speech_act == "trade":
                relation_strength = min(1.0, relation_strength + 0.03)
            elif speech_act == "beg":
                relation_strength = max(0.0, relation_strength - 0.01)
            elif speech_act == "threat":
                relation_strength = max(0.0, relation_strength - 0.05)

            # Update adjacency matrix (directed weight) softly
            try:
                self.adjacency_matrix[actor_idx, target_idx] = relation_strength
            except Exception:
                pass

            # Speech cost is cheap but non-trivial
            self.hunger[actor_idx] = max(0.0, self.hunger[actor_idx] - 5.0)
            self.hydration[actor_idx] = max(0.0, self.hydration[actor_idx] - 5.0)

            # Meme reinforcement
            key = (speech_act, obj_word.lower())
            speech_success = False
            transfer_amt = 0.0

            # Try to make speech have a tangible effect (cheap compared to fighting)
            if speech_act in ("beg", "trade", "threat"):
                try:
                    # Base willingness influenced by relation/satisfaction; threats get strength bonus
                    target_str = max(1.0, float(self.strength[target_idx]))
                    actor_str = max(1.0, float(self.strength[actor_idx]))
                    willingness = relation_strength + (self.satisfaction[target_idx] / 100.0) * 0.4
                    if speech_act == "threat":
                        willingness += 0.2 * (actor_str / target_str)
                    success_prob = np.clip(0.15 + willingness * 0.6, 0.05, 0.95)
                    if random.random() < success_prob and self.wealth[target_idx] > 0.5:
                        speech_success = True
                        transfer_amt = min(5.0, self.wealth[target_idx] * 0.5)
                        self.wealth[target_idx] -= transfer_amt
                        self.wealth[actor_idx] += transfer_amt
                        # Begging hurts pride a bit, threat hurts relation a lot
                        if speech_act == "beg":
                            relation_strength = max(0.0, relation_strength - 0.01)
                        elif speech_act == "threat":
                            relation_strength = max(0.0, relation_strength - 0.08)
                        else:
                            relation_strength = min(1.0, relation_strength + 0.02)
                        try:
                            self.event_logger.log(
                                "SPEECH_SUCCESS",
                                self.time_step,
                                actor_id=speaker_id,
                                target_id=listener_id,
                                speech_act=speech_act,
                                amount=float(transfer_amt),
                                pattern=tarzan_pattern,
                            )
                        except Exception:
                            pass
                    else:
                        # Failed speech: slight relation drop for threats/begging
                        if speech_act == "threat":
                            relation_strength = max(0.0, relation_strength - 0.05)
                        elif speech_act == "beg":
                            relation_strength = max(0.0, relation_strength - 0.02)
                        try:
                            self.event_logger.log(
                                "SPEECH_FAIL",
                                self.time_step,
                                actor_id=speaker_id,
                                target_id=listener_id,
                                speech_act=speech_act,
                                pattern=tarzan_pattern,
                            )
                        except Exception:
                            pass
                except Exception:
                    pass

            # Meme weight: reward success more than failure
            self.meme_bank[key] += 2.0 if speech_success else 0.5

            self.logger.info(
                f"DIALOGUE: '{speaker_id}' -> '{listener_id}' [{speech_act}] {text}"
            )
            self.event_logger.log(
                "DIALOGUE",
                self.time_step,
                speaker_id=speaker_id,
                listener_id=listener_id,
                speaker_emotion=speaker_emotion,
                listener_emotion=listener_emotion,
                relation_strength=relation_strength,
                topic=topic,
                speech_act=speech_act,
                text=text,
                pattern=tarzan_pattern,
                success=speech_success,
                transfer=transfer_amt,
            )
            return

        if action == 'share_food' and target_idx != -1:
            amount_to_give = 30
            self.hunger[actor_idx] -= amount_to_give
            self.hunger[target_idx] = min(100, self.hunger[target_idx] + amount_to_give)
            self.logger.info(f"ACTION: '{self.cell_ids[actor_idx]}' shares food with '{self.cell_ids[target_idx]}'.")
            self.event_logger.log('SHARE_FOOD', self.time_step, actor_id=self.cell_ids[actor_idx], target_id=self.cell_ids[target_idx])

            # --- Record Spiritual Event for Resonance ---
            self.spiritual_events.append({
                'type': 'resonance',
                'subtype': 'charity',
                'actor_idx': actor_idx,
                'position': self.positions[actor_idx]
            })

            # --- Leave an experience scar ---
            self.experience_scars[target_idx] |= 2 # Set the second bit for receiving charity

            # --- Create value-mass (meaning) at the site of sharing ---
            try:
                actor_pos = self.positions[actor_idx]
                x, y = int(actor_pos[0]) % self.width, int(actor_pos[1]) % self.width
                delta_e_local = float(amount_to_give) * 0.1
                self._imprint_gaussian(self.value_mass_field, x, y, sigma=self._vm_sigma, amplitude=delta_e_local)
                self.event_logger.log(
                    'MEANING_CREATED',
                    self.time_step,
                    type='sharing',
                    magnitude=delta_e_local,
                    x=x,
                    y=y,
                )
            except Exception:
                # Meaning creation is best effort; failures should not break core combat logic.
                pass
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

                # Record hunger before eating to compute experiential "reward".
                before_hunger = float(self.hunger[actor_idx])

                # Eating kills the plant and restores some hunger for now.
                self.hp[target_idx] = 0
                food_value = 20
                self.hunger[actor_idx] = min(100, self.hunger[actor_idx] + food_value)

                # Tiny perceptron learning step: was this a good experience?
                try:
                    cell_id = self.cell_ids[actor_idx]
                    cell_obj = self.materialize_cell(cell_id)
                    if cell_obj is not None:
                        tpos = self.positions[target_idx]
                        tx = int(tpos[0]) % self.width
                        ty = int(tpos[1]) % self.width
                        local_value = float(self.value_mass_field[ty, tx]) if self.value_mass_field.size else 0.0
                        local_threat = float(self.threat_field[ty, tx]) if self.threat_field.size else 0.0
                        features = {
                            "value_mass": local_value,
                            "threat": local_threat,
                        }
                        after_hunger = float(self.hunger[actor_idx])
                        delta = after_hunger - before_hunger
                        # Map outcome to a simple target: +1 if hunger improved, else -1.
                        target = 1.0 if delta > 0 else -1.0
                        cell_obj.perceptron_learn(features, target)
                except Exception:
                    pass

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
                # Fighting is costly: burn a big chunk of stamina
                self.hunger[actor_idx] = max(0.0, self.hunger[actor_idx] - 20.0)
                self.hydration[actor_idx] = max(0.0, self.hydration[actor_idx] - 20.0)
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

            # Track potential killer for alignment/outlaw logic.
            was_alive = self.hp[target_idx] > 0 and self.is_alive_mask[target_idx]

            self.hp[target_idx] -= final_damage
            self.is_injured[target_idx] = True
            if was_alive and self.hp[target_idx] <= 0 and self.is_alive_mask[target_idx]:
                # Log a KILL event with killer/victim; DEATH is handled in cleanup.
                self.event_logger.log(
                    'KILL',
                    self.time_step,
                    killer_id=self.cell_ids[actor_idx],
                    victim_id=self.cell_ids[target_idx],
                    victim_element=str(self.element_types[target_idx]),
                )
            self.logger.info(f"COMBAT: Damage dealt: {final_damage:.2f}.")

            if self.hp[target_idx] <= 0:
                self.logger.info(f"HUNT: '{self.cell_ids[actor_idx]}' killed '{self.cell_ids[target_idx]}'.")
                food_value = 50
                self.hunger[actor_idx] = min(100, self.hunger[actor_idx] + food_value)
                self.satisfaction[actor_idx] += 20
                self.emotions[actor_idx] = 'joy'

                # --- Learning: Successful Kill reinforces 'Attack' words ---
                self._learn_from_resonance(actor_idx, "attack", "kill", 5.0)


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
                child_props = parent_cell.properties.copy() if parent_cell else {'element_type': 'life'}

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
        # Mating readiness increases even when moderately hungry; give a head start.
        ready_mask = (self.hunger[animal_indices] > 25) & (self.hp[animal_indices] > self.max_hp[animal_indices] * 0.4)
        self.mating_readiness[animal_indices[ready_mask]] = np.minimum(1.0, self.mating_readiness[animal_indices[ready_mask]] + 0.5)

        # Severe hunger or injury reduces readiness
        not_ready_mask = (self.hunger[animal_indices] < 10) | (self.is_injured[animal_indices])
        self.mating_readiness[animal_indices[not_ready_mask]] = 0

        female_mask = (self.genders[animal_indices] == 'female') & (self.mating_readiness[animal_indices] >= 0.3)
        fertile_female_indices = animal_indices[female_mask]
        for i in fertile_female_indices:
            connected_indices = adj_matrix_csr[i].indices
            male_mask = (self.genders[connected_indices] == 'male') & (self.mating_readiness[connected_indices] >= 0.3)
            potential_mates = connected_indices[male_mask]

            # Enforce same-species mating: match by label (species) not by role/culture.
            my_label = (self.labels[i] or "").lower()
            if my_label:
                same_species = [m for m in potential_mates if (self.labels[m] or "").lower() == my_label]
                potential_mates = np.array(same_species, dtype=np.int32)

            # If no connected mates, try spatial neighbors within a small radius (approximate)
            if potential_mates.size == 0:
                pos_f = self.positions[i]
                near_idx = animal_indices
                dx = self.positions[near_idx, 0] - pos_f[0]
                dy = self.positions[near_idx, 1] - pos_f[1]
                dist2 = dx * dx + dy * dy
                close_mask = (dist2 < (20.0 ** 2)) & (self.genders[near_idx] == 'male') & (self.mating_readiness[near_idx] >= 0.3)
                # same-species check for spatial search too
                if my_label:
                    close_mask = close_mask & (np.char.lower(self.labels[near_idx]) == my_label)
                potential_mates = near_idx[close_mask]

            if potential_mates.size > 0:
                mate_idx = random.choice(potential_mates)
                # Gestation costs resources
                self.hp[i] = max(0, self.hp[i] - 2)
                self.hunger[i] = max(0, self.hunger[i] - 5)

                new_animal_id = f"{self.labels[i]}_{self.time_step}"
                parent_cell = self.materialize_cell(self.cell_ids[i])
                child_props = parent_cell.properties.copy()
                child_props['gender'] = random.choice(['male', 'female'])
                new_cell = Cell(id=new_animal_id, dna=self.primordial_dna, properties=child_props)
                newly_born_cells.append(new_cell)
                # Log birth
                try:
                    self.event_logger.log('BIRTH', self.time_step, mother_id=self.cell_ids[i], father_id=self.cell_ids[mate_idx], child_id=new_animal_id)
                except Exception:
                    pass

                # Reset readiness after procreation
                self.mating_readiness[i] = 0.2
                self.mating_readiness[mate_idx] = 0.2
                self.logger.info(f"Mating: '{self.cell_ids[i]}' and '{self.cell_ids[mate_idx]}' produced '{new_animal_id}'.")

                # --- Learning: Birth reinforces 'Bonding' words ---
                self._learn_from_resonance(i, "talk", "birth", 10.0)

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

        return newly_born_cells


    def _apply_physics_and_cleanup(self, newly_born_cells: List[Cell]):
        """Applies final state changes, handles death, and integrates new cells."""
        adj_matrix_csr = self.adjacency_matrix.tocsr()

        # Add newly born cells to the world
        for cell in newly_born_cells:
            if cell.id not in self.id_to_idx:
                self.add_cell(cell.id, dna=cell.dna, properties=cell.properties)

        # Process death for cells with zero or less HP
        # Khala Units (Protoss) have a chance to ascend as Dragoons/Immortals instead of dying?
        # For now, just standard death.
        apoptosis_mask = (self.hp <= 0) & self.is_alive_mask
        self.is_alive_mask[apoptosis_mask] = False
        self.hp[apoptosis_mask] = 0.0
        self.khala_connected_mask[apoptosis_mask] = False # Sever connection upon death

        dead_cell_indices = np.where(apoptosis_mask)[0]
        for dead_idx in dead_cell_indices:
            cell_id = self.cell_ids[dead_idx]
            self.event_logger.log('DEATH', self.time_step, cell_id=cell_id)

            # --- Law of Historical Imprint: All deaths leave a mark ---
            dx = int(self.positions[dead_idx, 0]) % self.width
            dy = int(self.positions[dead_idx, 1]) % self.width
            self._imprint_gaussian(self.h_imprint, dx, dy, sigma=self._h_sigma, amplitude=0.8)
            self.logger.info(f"IMPRINT: Death of '{cell_id}' left a historical imprint at ({dx}, {dy}).")


            if cell_id in self.materialized_cells:
                dead_cell = self.materialized_cells[cell_id]

                # --- CRYSTALLIZATION (Freeze physics state to KG) ---
                # Before archiving, freeze the soul state back to the cosmos.
                self.crystallize_cell(dead_cell)

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

    def _apply_cosmic_laws(self):
        """
        하나의 통합된 계층에서 모든 우주 법칙(상승, 하강, 공명, 감응 등)을 처리합니다.
        This function handles all cosmic laws (Ascension, Descent, Resonance, Staining) in a single, unified layer.
        """
        # --- 1. Law of Incarnation: Avatars radiate their spiritual energy ---
        # --- Angels (Virtues) radiate their light ---
        angel_mask = (self.labels == '천사') & self.is_alive_mask
        angel_indices = np.where(angel_mask)[0]
        for i in angel_indices:
            px, py = int(self.positions[i][0]) % self.width, int(self.positions[i][1]) % self.width
            base_amplitude = (self.strength[i] / 50.0) * 0.5

            if self.angel_status == 'slumbering':
                amplitude = base_amplitude * 0.25
            elif self.angel_status == 'watching':
                amplitude = base_amplitude * 0.5
            else: # manifested
                amplitude = base_amplitude

            # TODO: Expand this to identify the specific angel (e.g., 'Vitariael', 'Sophiel')
            # and radiate on the corresponding Ascension channel (e.g., ASCENSION_LIFE, ASCENSION_REFLECTION).
            # For now, all angels radiate Life as a baseline.
            self._imprint_gaussian(self.ascension_field[:, :, ASCENSION_LIFE], px, py, sigma=20.0, amplitude=amplitude)

            # --- Demons (Sins) create gravitational wells ---
            demon_mask = (self.labels == '마왕') & self.is_alive_mask
            demon_indices = np.where(demon_mask)[0]
            for i in demon_indices:
                px, py = int(self.positions[i][0]) % self.width, int(self.positions[i][1]) % self.width
                base_amplitude = (self.strength[i] / 50.0) * 0.5

                if self.demon_lord_status == 'sealed':
                    amplitude = base_amplitude * 0.25
                elif self.demon_lord_status == 'awakening':
                    amplitude = base_amplitude * 0.5
                else: # unleashed
                    amplitude = base_amplitude

                # TODO: Expand this to identify the specific demon (e.g., 'Motus', 'Mammon')
                # and create a gravitational well on the corresponding Descent channel (e.g., DESCENT_DEATH, DESCENT_CONSUMPTION).
                # For now, all demons create a Death well as a baseline.
                self._imprint_gaussian(self.descent_field[:, :, DESCENT_DEATH], px, py, sigma=20.0, amplitude=amplitude)

        # --- 2. Law of Resonance: Actions create spiritual ripples ---
        for event in self.spiritual_events:
            if event.get('type') == 'resonance':
                actor_idx = event['actor_idx']
                pos = event['position']
                px, py = int(pos[0]) % self.width, int(pos[1]) % self.width

                if event.get('subtype') == 'charity':
                    # The act of giving resonates with Love (Ascension)
                    amplitude = (self.wisdom[actor_idx] / 50.0) * 0.1
                    self._imprint_gaussian(self.ascension_field[:, :, ASCENSION_LOVE], px, py, sigma=15.0, amplitude=amplitude)
                    self.logger.info(f"RESONANCE: Act of charity by '{self.cell_ids[actor_idx]}' resonated with the cosmic axis of Love (delta={amplitude:.3f}).")


        # --- 3. Law of Staining: The environment influences the soul ---
        alive_indices = np.where(self.is_alive_mask)[0]
        if alive_indices.size > 0:
            # --- 4. Law of Incarnation Awakening/Slumbering ---
            # Check the total amount of virtue and sin in the world
            total_virtue = np.sum(self.ascension_field)
            total_sin = np.sum(self.descent_field)

            # Demon Lord state change
            if self.demon_lord_status == 'sealed' and total_sin > 100: # Threshold for awakening
                self.demon_lord_status = 'awakening'
                self.logger.info("AWAKENING: The Demon Lord's presence grows stronger as sin accumulates.")
                self.event_logger.log('DEMON_LORD_AWAKENING', self.time_step, total_sin=total_sin)
            elif self.demon_lord_status == 'awakening' and total_sin > 500: # Threshold for unleashing
                self.demon_lord_status = 'unleashed'
                self.logger.info("UNLEASHED: The Demon Lord is fully unleashed upon the world!")
                self.event_logger.log('DEMON_LORD_UNLEASHED', self.time_step, total_sin=total_sin)

            # Angel state change
            if self.angel_status == 'slumbering' and total_virtue > 100: # Threshold for watching
                self.angel_status = 'watching'
                self.logger.info("AWAKENING: The Angel begins to watch over the world as virtue spreads.")
                self.event_logger.log('ANGEL_AWAKENING', self.time_step, total_virtue=total_virtue)
            elif self.angel_status == 'watching' and total_virtue > 500: # Threshold for manifestation
                self.angel_status = 'manifested'
                self.logger.info("MANIFESTED: The Angel's influence manifests in the world!")
                self.event_logger.log('ANGEL_MANIFESTED', self.time_step, total_virtue=total_virtue)


            positions = self.positions[alive_indices].astype(int)
            pos_x = np.clip(positions[:, 0], 0, self.width - 1)
            pos_y = np.clip(positions[:, 1], 0, self.width - 1)

            # Read the ascension and descent values at each cell's location
            local_ascension = self.ascension_field[pos_y, pos_x]
            local_descent = self.descent_field[pos_y, pos_x]

            # For now, we'll just use the dominant ascension/descent strength.
            dominant_ascension_strength = np.max(local_ascension, axis=1)
            dominant_descent_strength = np.max(local_descent, axis=1)

            # --- Ascension's Influence (Awakening) ---
            wisdom_gain = dominant_ascension_strength * 0.05
            satisfaction_gain = dominant_ascension_strength * 0.1
            self.wisdom[alive_indices] += wisdom_gain.astype(self.wisdom.dtype)
            self.satisfaction[alive_indices] += satisfaction_gain

            # --- Descent's Influence (Corruption) ---
            satisfaction_loss = dominant_descent_strength * 0.2
            self.satisfaction[alive_indices] -= satisfaction_loss

            # Emotional Staining
            corruption_mask = dominant_descent_strength > 0.1
            if np.any(corruption_mask):
                corrupted_indices = alive_indices[corruption_mask]
                # Randomly assign negative emotions
                possible_emotions = ['sorrow', 'fear', 'anger']
                num_corrupted = len(corrupted_indices)
                random_emotions = np.random.choice(possible_emotions, num_corrupted)
                self.emotions[corrupted_indices] = random_emotions

            # Clip values to stay within reasonable bounds
            self.wisdom[alive_indices] = np.clip(self.wisdom[alive_indices], 0, 100)
            self.satisfaction[alive_indices] = np.clip(self.satisfaction[alive_indices], 0, 100)

    def _apply_law_of_awakening(self) -> List[AwakeningEvent]:
        """
        Applies the Law of Existential Change (e > r) and returns a list of awakening events.
        This is a core physical law of the world.
        """
        events = []
        alive_indices = np.where(self.is_alive_mask)[0]
        if alive_indices.size == 0:
            return events

        # --- Awakening conditions (rare, meaningful triggers) ---
        labels_alive = np.array([(self.labels[i] or '').lower() for i in alive_indices])
        is_beast = np.isin(labels_alive, ['wolf', 'bear'])
        is_mind = np.isin(labels_alive, ['wizard', 'mage'])
        is_spirit = np.isin(labels_alive, ['monk', 'knight', 'priest', 'cleric'])

        hp_ratio = self.hp[alive_indices] / np.maximum(1e-3, self.max_hp[alive_indices])
        near_death_mask = hp_ratio <= 0.05  # 임사 체험: HP 5% 이하

        great_sorrow_mask = (
            (self.emotions[alive_indices] == 'sorrow')
            & (self.satisfaction[alive_indices] <= 10)
            & (self.connection_counts[alive_indices] >= 3)
        )

        # Divine resonance: inside a high-value field pocket (coil/meaning hotspot)
        px = np.clip(self.positions[alive_indices, 0].astype(np.int32), 0, self.width - 1)
        py = np.clip(self.positions[alive_indices, 1].astype(np.int32), 0, self.width - 1)
        try:
            local_value = self.value_mass_field[py, px]
            value_threshold = float(np.percentile(self.value_mass_field, 99)) if self.value_mass_field.size else 0.0
            if value_threshold <= 0:
                value_threshold = float(np.max(self.value_mass_field)) * 0.8 if self.value_mass_field.size else 0.0
        except Exception:
            local_value = np.zeros_like(px, dtype=np.float32)
            value_threshold = 0.0
        divine_resonance_mask = local_value >= value_threshold

        # Mind trigger (혼/정신): 높은 통찰 또는 정신 붕괴
        high_insight_mask = self.insight[alive_indices] >= 5.0
        mental_break_mask = (self.emotions[alive_indices] == 'fear') & (self.satisfaction[alive_indices] <= 20)
        mind_trigger = (high_insight_mask | mental_break_mask) & is_mind

        # Spirit trigger (영/마음): 가치 공명 또는 큰 슬픔
        spirit_trigger = (divine_resonance_mask | great_sorrow_mask) & is_spirit

        # Body trigger (육/생존): 임사 체험
        body_trigger = near_death_mask & is_beast

        awakening_mask = (mind_trigger | spirit_trigger | body_trigger) & ~self.is_awakened[alive_indices]
        awakened_indices = alive_indices[awakening_mask]

        if awakened_indices.size > 0:
            for idx in awakened_indices:
                # Reason tagging for logging
                local_pos = np.where(alive_indices == idx)[0][0]
                if body_trigger[local_pos]:
                    reason = 'body_near_death'
                elif mind_trigger[local_pos]:
                    reason = 'mind_insight_or_break'
                elif spirit_trigger[local_pos]:
                    reason = 'spirit_value_or_sorrow'
                else:
                    reason = 'unknown'
                event = AwakeningEvent(
                    cell_id=self.cell_ids[idx],
                    e_value=float(self.insight[idx]),
                    r_value=int(self.age[idx])
                )
                events.append(event)
                self.logger.info(f"LAW OF CHANGE: '{event.cell_id}' awakened via {reason}")
                # Species-specific awakening effects
                label = (self.labels[idx] or '').lower()
                if label in ('wolf', 'bear'):
                    # Alpha evolution: boost combat, imprint leadership aura (threat & coherence bump)
                    self.strength[idx] *= 2.0
                    self.agility[idx] *= 1.3
                    self.max_hp[idx] *= 1.5
                    self.hp[idx] = min(self.hp[idx] * 1.5, self.max_hp[idx])
                    # Local aura: increase threat imprint and coherence to simulate "command" presence
                    try:
                        x = int(self.positions[idx, 0]) % self.width
                        y = int(self.positions[idx, 1]) % self.width
                        self._imprint_gaussian(self.threat_field, x, y, sigma=6.0, amplitude=2.0)
                        self._imprint_gaussian(self.coherence_field, x, y, sigma=6.0, amplitude=1.0)
                        self.event_logger.log('EVOLUTION_ALPHA', self.time_step, cell_id=self.cell_ids[idx], label=label)
                    except Exception:
                        pass
                elif label in ('monk', 'knight', 'priest', 'cleric'):
                    # Saintly awakening: pacifist, healing aura, no combat gain
                    self.strength[idx] = max(1.0, self.strength[idx] * 0.1)
                    self.agility[idx] = max(1.0, self.agility[idx] * 0.5)
                    self.hp[idx] = min(self.max_hp[idx], self.hp[idx] + 10)
                    # Healing field around them; boost value and will to reduce aggression nearby
                    try:
                        x = int(self.positions[idx, 0]) % self.width
                        y = int(self.positions[idx, 1]) % self.width
                        self._imprint_gaussian(self.value_mass_field, x, y, sigma=8.0, amplitude=2.5)
                        self._imprint_gaussian(self.will_field, x, y, sigma=8.0, amplitude=1.5)
                        self._imprint_gaussian(self.threat_field, x, y, sigma=8.0, amplitude=-2.0)
                        self.event_logger.log('ENLIGHTENED_SAINT', self.time_step, cell_id=self.cell_ids[idx], label=label)
                    except Exception:
                        pass
                elif label in ('wizard', 'mage'):
                    # Arcane ascension: huge mana surge, weaker body; arcane aura to push/pull intent
                    self.strength[idx] = max(1.0, self.strength[idx] * 0.5)
                    self.agility[idx] = max(1.0, self.agility[idx] * 0.8)
                    self.max_mana[idx] *= 2.5
                    self.mana[idx] = self.max_mana[idx]
                    try:
                        x = int(self.positions[idx, 0]) % self.width
                        y = int(self.positions[idx, 1]) % self.width
                        # Boost intentional/value fields to simulate arcane influence
                        self._imprint_gaussian(self.intentional_field[...,0], x, y, sigma=10.0, amplitude=3.0)
                        self._imprint_gaussian(self.intentional_field[...,1], x, y, sigma=10.0, amplitude=3.0)
                        self._imprint_gaussian(self.value_mass_field, x, y, sigma=6.0, amplitude=2.0)
                        self.event_logger.log('ARCANE_ASCENSION', self.time_step, cell_id=self.cell_ids[idx], label=label)
                    except Exception:
                        pass

                try:
                    self.event_logger.log('AWAKENING_EVENT', self.time_step, cell_id=event.cell_id, reason=reason)
                except Exception:
                    self.event_logger.log('AWAKENING_EVENT', self.time_step, cell_id=event.cell_id)

            # Enact physical consequences
            self.is_awakened[awakened_indices] = True # Mark as awakened to prevent immediate re-awakening

        return events


    def apply_will_field(self, field_type: str, strength: float, focus_point: Optional[Tuple[int, int]] = None, radius: int = 20):
        """
        [BODY LAYER] Swarm Intelligence Interface.
        Allows the 'Spirit' (Consciousness) to apply direct field effects to the 'Body' (World).
        Uses vectorized NumPy operations for high performance (3GB VRAM Friendly).
        """
        try:
            if field_type == 'calm':
                # Reduce threat, increase coherence
                self.threat_field *= (1.0 - strength)
                self.coherence_field = np.clip(self.coherence_field + strength * 0.1, 0, 1)

            elif field_type == 'growth':
                # Increase soil fertility and value mass
                if focus_point:
                    self._imprint_gaussian(self.soil_fertility, focus_point[0], focus_point[1], sigma=radius, amplitude=strength)
                    self._imprint_gaussian(self.value_mass_field, focus_point[0], focus_point[1], sigma=radius, amplitude=strength)
                else:
                    self.soil_fertility = np.clip(self.soil_fertility + strength * 0.05, 0, 1)

            elif field_type == 'entropy_stabilization':
                # Reduce random fluctuations (not directly modeled, but we can boost 'norms' and 'coherence')
                self.norms_field = np.clip(self.norms_field + strength * 0.1, 0, 1)
                self.coherence_field = np.clip(self.coherence_field + strength * 0.1, 0, 1)

            elif field_type == 'awakening':
                 # Boost 'insight' for all entities slightly
                 alive_indices = np.where(self.is_alive_mask)[0]
                 self.insight[alive_indices] += strength * 5.0

            self.logger.info(f"WILL FIELD APPLIED: {field_type} (Strength: {strength:.2f})")

        except Exception as e:
            self.logger.error(f"Failed to apply Will Field: {e}")

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
                details = {'concept_id': concept_id, 'hp_boost': energy_boost}
                scopes = [concept_id]
                event = self.chronicle.record_event('stimulus_injected', details, scopes, self.branch_id, self.parent_event_id)
                self.parent_event_id = event['id']
            idx = self.id_to_idx[concept_id]
            self.energy[idx] += float(energy_boost)
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
        """
        Emit a Korean utterance and IMPRINT its intent onto the physical fields.
        This implements 'Speech as Wave': words are not just logs, they are forces.
        """
        if actor_idx < 0 or actor_idx >= len(self.cell_ids):
            return
        text = kr_dialogue(key, **kwargs)
        if not text:
            return
        cell_id = self.cell_ids[actor_idx]

        # Log as a SAY event
        self.logger.info(f"SAY: '{cell_id}' {text}")
        self.event_logger.log('SAY', self.time_step, cell_id=cell_id, text=text)

        # --- Lexicon Physics: Map Sound to Field Effect ---
        # Each key maps to a field name and a base amplitude.
        # The amplitude is scaled by the word's 'Mastery' (historical weight).
        lexicon_physics = {
            "ATTACK":        ("threat", 2.0),
            "SKILL_ATTACK":  ("will", 3.0),
            "SPELL_FIRE":    ("threat", 4.0),
            "SPELL_HEAL":    ("coherence", 5.0), # Healing restores harmony
            "TALK_BOND":     ("coherence", 1.5),
            "TALK_COMFORT":  ("coherence", 2.0), # Comfort reduces entropy
            "TALK_TRADE":    ("value_mass", 1.0), # Trade creates value
            "TALK_THREAT":   ("threat", 1.5),
            "TALK_BEG":      ("hydration", 5.0), # Metaphor: thirst for help -> fluid field
            "TALK_CELEBRATE":("value_mass", 2.0),
            "IDLE_PLAY":     ("norms", 0.5), # Play reinforces social norms gently
        }

        if key in lexicon_physics:
            field_name, base_amp = lexicon_physics[key]
            mastery_scale = self.lexicon_mastery[key] # Starts at 1.0, grows with resonance
            amplitude = base_amp * mastery_scale

            # Apply the field imprint at the speaker's location
            try:
                px = int(self.positions[actor_idx, 0]) % self.width
                py = int(self.positions[actor_idx, 1]) % self.width

                target_field = None
                if field_name == "threat": target_field = self.threat_field
                elif field_name == "will": target_field = self.will_field
                elif field_name == "coherence": target_field = self.coherence_field
                elif field_name == "value_mass": target_field = self.value_mass_field
                elif field_name == "hydration": target_field = self.hydration_field
                elif field_name == "norms": target_field = self.norms_field

                if target_field is not None:
                    # Sigma represents the 'Range' of the voice. Louder/Mastered words travel further?
                    # For now, keep sigma constant to represent acoustic range.
                    self._imprint_gaussian(target_field, px, py, sigma=8.0, amplitude=amplitude)
                    # self.logger.debug(f"FIELD_RIPPLE: '{key}' boosted {field_name} by {amplitude:.2f} at ({px},{py})")
            except Exception:
                pass

            # Photon message (colorized intent)
            try:
                hue = value_to_hue(amplitude, value_range=(0.0, 10.0))
                photon = PhotonEntity(hue=hue, intensity=min(1.0, amplitude / 10.0), polarization=(0.0, 0.0, 1.0), payload=key)
                self.event_logger.log("PHOTON_MESSAGE", self.time_step, speaker=cell_id, photon=photon.as_dict())
            except Exception:
                pass

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

    def connect_to_khala(self, cell_idx: int):
        """Connects a cell to the Khala (The Great Link)."""
        if 0 <= cell_idx < len(self.khala_connected_mask):
            self.khala_connected_mask[cell_idx] = True
            # Immediate Insight Boost upon connection
            self.insight[cell_idx] += 10.0
            self.logger.info(f"KHALA: '{self.cell_ids[cell_idx]}' has joined the Khala. En Taro Gang-deok!")

    def _synchronize_khala(self):
        """
        Enforces the 'Delta One' protocol.
        Averages Insight and Will across all connected cells, creating a shared consciousness.
        """
        connected_indices = np.where(self.khala_connected_mask & self.is_alive_mask)[0]
        if connected_indices.size < 2:
            return

        # Calculate the 'Psionic Mean'
        avg_insight = np.mean(self.insight[connected_indices])
        # avg_will = np.mean(self.will[connected_indices]) # If we had a per-cell will attribute

        # Convergence Speed (How fast we reach Delta=1)
        # The stronger the connection, the faster they synchronize.
        alpha = 0.1 + (self.delta_synchronization_factor * 0.4)

        # Apply convergence with Psionic Dithering
        # We add small noise to prevent feature collapse (Singular Matrix)
        noise = np.random.normal(0, 0.1, size=connected_indices.shape)
        self.insight[connected_indices] = (
            (1.0 - alpha) * self.insight[connected_indices] +
            alpha * avg_insight + noise
        )

        # Also synchronize HP slightly (Shared Life Force / Shield Battery effect)
        # avg_hp = np.mean(self.hp[connected_indices])
        # self.hp[connected_indices] = (
        #     (1.0 - 0.05) * self.hp[connected_indices] +
        #     0.05 * avg_hp
        # )

        if self.time_step % 50 == 0:
            self.logger.info(f"KHALA: Psionic Matrix Synchronized. Avg Insight: {avg_insight:.2f}")

    def print_world_summary(self):
        """Prints a detailed summary of the current world state for debugging."""
        self._sync_states_to_objects()
        print(f"\n--- World State (Time: {self.time_step}) ---")
        living_cell_count = np.sum(self.is_alive_mask)
        khala_count = np.sum(self.khala_connected_mask & self.is_alive_mask)
        print(f"Living Cells: {living_cell_count}, Khala Linked: {khala_count}, Dead Cells: {len(self.graveyard)}")

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



