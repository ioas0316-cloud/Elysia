"""
Conceptual Phase Knob Engine (개념적 위상 노브 엔진)

This module implements the transition from physical 3D Euclidean distances
to Conceptual Phase Distance where the distance between entities is governed by:
1. Topological Isomorphism & Phase Resistance (Friction Error E -> 0 for isomorphic concepts).
2. Inverse Latent World Mechanism Extraction (Generating equation Theta, Boundary Conditions Delta).
3. Invariant vs. Variant Topological Classification (Invariance = structural law bedrock, Variant = dynamic wave).
4. Cognitive Knob-ification (Extracting adjustable parameters: friction, rotor phase angle, entropy, causal tension).
5. Conceptual State Rearrangement Loop (Dynamic re-simulation and reshaping of world state upon knob tuning).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


@dataclass
class CognitiveKnob:
    """Represents a tunable parameter extracted from the dynamic wave variants."""
    name: str
    value: float
    min_value: float = 0.0
    max_value: float = 2.0 * np.pi
    default_value: float = 1.0
    description: str = ""

    def tune(self, new_value: float) -> float:
        """Clamps and sets new value for the knob."""
        self.value = float(np.clip(new_value, self.min_value, self.max_value))
        return self.value


@dataclass
class ConceptualEntity:
    """Represents an entity or phenomenon mapped into the conceptual phase space."""
    name: str
    domain: str
    causal_vector: np.ndarray  # Direct state or feature vector
    intent_lens: np.ndarray    # Higher-order intent / functional purpose vector
    boundary_condition: np.ndarray  # Boundary constraints Delta
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptualDistanceResult:
    """Result of computing conceptual phase distance between two entities."""
    entity_a: str
    entity_b: str
    euclidean_physical_distance: float
    conceptual_phase_distance: float  # Friction error E
    clifford_rotor_angle: float        # Phase rotation Delta theta needed
    shared_invariance_score: float     # Isomorphism degree [0, 1]
    is_topologically_isomorphic: bool


@dataclass
class ExtractedWorldMechanism:
    """Structural decomposition into generating mechanisms, invariants, variants, and knobs."""
    generating_mechanism_theta: np.ndarray
    boundary_condition_delta: np.ndarray
    invariant_bedrock: np.ndarray
    variant_dynamic_wave: np.ndarray
    knobs: Dict[str, CognitiveKnob]


class ConceptualPhaseKnobEngine:
    """
    Engine for transforming discrete surface data and physical coordinates into
    a continuous conceptual phase manifold where distance is friction/resonance,
    and world models can be dynamically tuned via Cognitive Knobs.
    """

    def __init__(self, dimension: int = 64, friction_threshold: float = 0.15):
        self.dimension = dimension
        self.friction_threshold = friction_threshold
        self.entities: Dict[str, ConceptualEntity] = {}
        self.active_knobs: Dict[str, CognitiveKnob] = {}

    def register_entity(self, entity: ConceptualEntity) -> None:
        """Registers a conceptual entity into the phase manifold."""
        # Ensure vectors match engine dimension
        if len(entity.causal_vector) < self.dimension:
            padded = np.zeros(self.dimension)
            padded[:len(entity.causal_vector)] = entity.causal_vector
            entity.causal_vector = padded
        elif len(entity.causal_vector) > self.dimension:
            entity.causal_vector = entity.causal_vector[:self.dimension]

        if len(entity.intent_lens) < self.dimension:
            padded = np.zeros(self.dimension)
            padded[:len(entity.intent_lens)] = entity.intent_lens
            entity.intent_lens = padded
        elif len(entity.intent_lens) > self.dimension:
            entity.intent_lens = entity.intent_lens[:self.dimension]

        self.entities[entity.name] = entity

    def compute_conceptual_distance(
        self,
        name_a: str,
        name_b: str,
        physical_coords_a: Optional[np.ndarray] = None,
        physical_coords_b: Optional[np.ndarray] = None
    ) -> ConceptualDistanceResult:
        """
        Computes the conceptual phase distance (friction error E and Clifford rotor angle Delta theta)
        between two entities when viewed through higher-order intent lenses.
        """
        entity_a = self.entities[name_a]
        entity_b = self.entities[name_b]

        # 1. Physical Euclidean Distance
        if physical_coords_a is not None and physical_coords_b is not None:
            euclidean_dist = float(np.linalg.norm(physical_coords_a - physical_coords_b))
        else:
            euclidean_dist = float(np.linalg.norm(entity_a.causal_vector - entity_b.causal_vector))

        # 2. Higher-order Intent Lens Alignment (Functional Isomorphism)
        lens_a = entity_a.intent_lens / (np.linalg.norm(entity_a.intent_lens) + 1e-9)
        lens_b = entity_b.intent_lens / (np.linalg.norm(entity_b.intent_lens) + 1e-9)

        # Cosine similarity in intent space
        alignment = float(np.dot(lens_a, lens_b))
        alignment = float(np.clip(alignment, -1.0, 1.0))

        # Clifford Rotor angle Delta theta = arccos(dot(lens_a, lens_b))
        clifford_rotor_angle = float(np.arccos(alignment))

        # 3. Conceptual Phase Distance (Friction Error E)
        # E = (1 - alignment) * (1 + friction_penalty_from_boundary)
        boundary_diff = np.linalg.norm(entity_a.boundary_condition - entity_b.boundary_condition)
        friction_error = float((1.0 - alignment) * (1.0 + 0.1 * boundary_diff))

        # 4. Invariance score and isomorphism
        shared_invariance_score = float(1.0 - (friction_error / (1.0 + friction_error)))
        is_isomorphic = friction_error <= self.friction_threshold

        return ConceptualDistanceResult(
            entity_a=name_a,
            entity_b=name_b,
            euclidean_physical_distance=euclidean_dist,
            conceptual_phase_distance=friction_error,
            clifford_rotor_angle=clifford_rotor_angle,
            shared_invariance_score=shared_invariance_score,
            is_topologically_isomorphic=is_isomorphic
        )

    def extract_world_mechanism(
        self,
        time_series_states: np.ndarray,
        boundary_conditions: np.ndarray
    ) -> ExtractedWorldMechanism:
        """
        Performs Inverse Latent Mechanism Extraction on state trajectories:
        - Extracts generating parameters Theta
        - Separates Invariant structural bedrock from Variant dynamic waves
        - Extracts Cognitive Knobs (Friction, Rotor Phase Angle, Entropy, Causal Tension)
        """
        # Time series shape: (T, D)
        T, D = time_series_states.shape

        # 1. Mean over time as Invariant structural bedrock
        invariant_bedrock = np.mean(time_series_states, axis=0)

        # 2. Deviations over time as Variant dynamic waves
        variant_dynamic_wave = time_series_states - invariant_bedrock

        # 3. Inverse mechanism Theta extraction (covariances & transition dynamics)
        cov_matrix = np.cov(variant_dynamic_wave, rowvar=False)
        generating_mechanism_theta = cov_matrix

        # 4. Extract Cognitive Knobs
        # Friction Knob (Variance scale)
        friction_val = float(np.mean(np.std(time_series_states, axis=0)))
        friction_knob = CognitiveKnob(
            name="friction_coefficient",
            value=friction_val,
            min_value=0.001,
            max_value=5.0,
            default_value=1.0,
            description="Controls resistance and damping in the state manifold"
        )

        # Rotor Phase Angle Knob (Phase rotation momentum)
        diffs = np.diff(time_series_states, axis=0)
        rotor_val = float(np.arctan2(np.mean(diffs[:, 1] if D > 1 else diffs[:, 0]),
                                     np.mean(diffs[:, 0]) + 1e-9) % (2 * np.pi))
        rotor_knob = CognitiveKnob(
            name="phase_rotor_angle",
            value=rotor_val,
            min_value=0.0,
            max_value=2.0 * np.pi,
            default_value=0.0,
            description="Controls the topological rotation angle of space-time flow"
        )

        # Causal Tension Knob (Constraint tightness)
        tension_val = float(np.linalg.norm(boundary_conditions))
        tension_knob = CognitiveKnob(
            name="causal_tension",
            value=tension_val,
            min_value=0.1,
            max_value=10.0,
            default_value=1.0,
            description="Controls boundary rigidity and structural tension"
        )

        # Entropy Knob (Distribution dispersion)
        entropy_val = float(np.trace(cov_matrix) if D > 1 else float(cov_matrix))
        entropy_knob = CognitiveKnob(
            name="entropy_dispersion",
            value=entropy_val,
            min_value=0.0,
            max_value=10.0,
            default_value=1.0,
            description="Controls thermal fluctuation and entropy release"
        )

        knobs = {
            "friction": friction_knob,
            "rotor_angle": rotor_knob,
            "causal_tension": tension_knob,
            "entropy": entropy_knob
        }

        self.active_knobs = knobs

        return ExtractedWorldMechanism(
            generating_mechanism_theta=generating_mechanism_theta,
            boundary_condition_delta=boundary_conditions,
            invariant_bedrock=invariant_bedrock,
            variant_dynamic_wave=variant_dynamic_wave,
            knobs=knobs
        )

    def resimulate_world_state(
        self,
        mechanism: ExtractedWorldMechanism,
        knob_adjustments: Dict[str, float]
    ) -> np.ndarray:
        """
        Dynamically resimulates and reshapes the world state based on adjusted Cognitive Knobs.
        """
        # Apply knob adjustments
        for knob_key, new_val in knob_adjustments.items():
            if knob_key in mechanism.knobs:
                mechanism.knobs[knob_key].tune(new_val)

        friction = mechanism.knobs["friction"].value
        rotor_angle = mechanism.knobs["rotor_angle"].value
        tension = mechanism.knobs["causal_tension"].value
        entropy = mechanism.knobs["entropy"].value

        # Base invariant bedrock
        state = mechanism.invariant_bedrock.copy()

        # Apply phase rotor rotation
        D = len(state)
        cos_r, sin_r = np.cos(rotor_angle), np.sin(rotor_angle)
        if D >= 2:
            rot_matrix = np.eye(D)
            rot_matrix[0, 0] = cos_r
            rot_matrix[0, 1] = -sin_r
            rot_matrix[1, 0] = sin_r
            rot_matrix[1, 1] = cos_r
            state = rot_matrix @ state

        # Apply friction damping & entropy scaling on variant wave amplitude (RMS/abs mean)
        variant_amplitude = np.mean(np.abs(mechanism.variant_dynamic_wave), axis=0)
        scaled_variant = (variant_amplitude / (friction + 1e-5)) * (1.0 + 0.1 * entropy)

        # Apply causal tension constraint
        constrained_state = (state + scaled_variant) / (1.0 + 0.05 * tension)

        return constrained_state
