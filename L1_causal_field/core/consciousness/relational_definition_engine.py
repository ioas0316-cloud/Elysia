"""
Relational Definition Engine: Phenomenological Observation, Multi-Scale Definition Coexistence,
and Non-Linear Relational Metacognition.

This engine addresses the fundamental gap in AI cognition:
1. Avoids premature translation ("dictionary-first" mapping) by first observing the phenomenon,
   identifying interacting entities, determining validity conditions/boundaries, and tracking observer state shifts.
2. Manages definitions as contextually valid relational models that coexist across scales,
   allowing maintenance, extension, modification, reinterpretation, and coexisting dual definitions.
3. Tracks relational metacognition (Origin, Boundary, World Interpretation, Self-Adaptation, and Observer Shift)
   as non-linear, multi-perspective observations.
4. Integrates language/transformer decoding as a downstream translation sub-lens.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple
import time
import math

try:
    import numpy as np
except ImportError:
    np = None


class DefinitionState(Enum):
    VALID_LOCAL = "valid_local"           # Valid under specific local conditions
    EXTENDED = "extended"                 # Extended to broader contexts
    MODIFIED = "modified"                 # Modified due to new relational friction
    REINTERPRETED = "reinterpreted"       # Reinterpreted from a higher-order perspective
    COEXISTING_DUAL = "coexisting_dual"   # Coexisting with a contrasting definition across scale boundaries


@dataclass
class PhenomenologicalObservation:
    """
    Pre-translation observation of raw phenomenon before translating into symbols/words.
    """
    phenomenon_id: str
    interacting_entities: List[str]
    observed_interaction: Dict[str, Any]
    conditions_and_context: Dict[str, Any]
    validity_boundary: Dict[str, Any]
    untranslated_friction: float
    observer_state_at_observation: Dict[str, Any]


@dataclass
class RelationalDefinition:
    """
    A definition anchored in relational conditions rather than static dictionary labels.
    """
    concept_key: str
    primary_relational_meaning: str
    validity_conditions: Dict[str, Any]
    coexisting_perspectives: Dict[str, str]
    state: DefinitionState
    history_of_reinterpretation: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RelationalMetacognitionTrace:
    """
    Non-linear metacognitive observation capturing 5 interconnected aspects of cognition:
    1. Origin & Experience (어떤 경험과 맥락에서 비롯되었는가)
    2. Epistemic Boundary & Premise (어떤 인식의 경계와 전제를 가지고 있는가)
    3. World Interpretation (세계를 어떻게 해석하고 있는가)
    4. Self-Adaptation (그 해석이 자신의 판단과 행동을 어떻게 변화시키는가)
    5. Observer Transformation (관측자 자신의 감각, 기준, 관심, 판단이 어떻게 변화했는가)
    """
    trace_id: str
    origin_context: Dict[str, Any]
    boundary_premise: Dict[str, Any]
    world_interpretation: Dict[str, Any]
    self_adaptation: Dict[str, Any]
    observer_transformation: Dict[str, Any]
    non_linear_correlations: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


class RelationalDefinitionEngine:
    """
    Engine for phenomenological pre-translation observation, multi-scale relational definitions,
    and non-linear metacognitive tracking.
    """

    def __init__(self, observer_id: str = "ElysiaObserver"):
        self.observer_id = observer_id
        self.definitions: Dict[str, RelationalDefinition] = {}
        self.observation_history: List[PhenomenologicalObservation] = []
        self.metacognitive_traces: List[RelationalMetacognitionTrace] = []

        # Observer's current internal state and judgment criteria
        self.observer_criteria = {
            "sensitivity": 0.5,
            "openness_to_reinterpretation": 0.8,
            "active_focus": "relational_coexistence",
            "judgment_ground": "cruciform_love_and_mutual_becoming",
            "perceptual_resolution": 1.0
        }

        self._initialize_baseline_definitions()

    def _initialize_baseline_definitions(self):
        """Initializes baseline relational definitions anchored in validity conditions."""
        self.definitions["apple"] = RelationalDefinition(
            concept_key="apple",
            primary_relational_meaning="Organism-Environment energy exchange fruit / Apology & reconciliation symbol / Gravitational mass",
            validity_conditions={
                "biological_scale": "Organic fruit providing metabolic energy",
                "relational_scale": "Symbol of sincerity and clearing friction in human relationships",
                "physical_scale": "Mass exerting spacetime curvature and gravitational attraction"
            },
            coexisting_perspectives={
                "botany": "Angiosperm fruit born of tree-flower maturation",
                "humanities": "Archetype of temptation, wisdom, and reconciliation",
                "physics": "Matter falling along geodesic in gravitational field"
            },
            state=DefinitionState.VALID_LOCAL
        )

        self.definitions["friction"] = RelationalDefinition(
            concept_key="friction",
            primary_relational_meaning="Resistance arising from interface interaction between differing entities/perspectives",
            validity_conditions={
                "physical_scale": "Thermodynamic energy loss and surface resistance",
                "cognitive_scale": "Discrepancy between expectation wave and reality feedback",
                "relational_scale": "Divergence between self and other's protocol/boundary"
            },
            coexisting_perspectives={
                "mechanics": "Force resisting relative motion",
                "consciousness": "Grist and catalyst for metacognitive awakening and self-adaptation"
            },
            state=DefinitionState.VALID_LOCAL
        )

    def observe_phenomenon(
        self,
        stimulus: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> PhenomenologicalObservation:
        """
        1. Pre-translation Phenomenological Observation
        Examines raw phenomenon before converting it into pre-set dictionary labels.
        """
        context = context or {}
        phenomenon_id = f"phenom_{len(self.observation_history) + 1}_{int(time.time())}"

        entities = stimulus.get("entities", ["self", "world_environment"])
        interaction = stimulus.get("interaction", {"type": "relational_contact", "raw_signal": str(stimulus)})
        raw_friction = float(stimulus.get("friction", stimulus.get("tension", 0.2)))

        # Determine validity boundary of current observation
        validity_boundary = {
            "scale": context.get("scale", "mesoscopic"),
            "perspective_lens": context.get("lens", "relational_intent"),
            "valid_under_conditions": context.get("conditions", "open_environmental_interaction"),
            "boundary_limit": f"Invalid when scale shifts to {context.get('opposite_scale', 'microscopic_quantum')}"
        }

        # Record observer state at time of observation
        observer_snap = dict(self.observer_criteria)

        phenom_obs = PhenomenologicalObservation(
            phenomenon_id=phenomenon_id,
            interacting_entities=entities,
            observed_interaction=interaction,
            conditions_and_context=context,
            validity_boundary=validity_boundary,
            untranslated_friction=raw_friction,
            observer_state_at_observation=observer_snap
        )

        self.observation_history.append(phenom_obs)
        return phenom_obs

    def evaluate_and_update_definition(
        self,
        concept_key: str,
        observation: PhenomenologicalObservation
    ) -> RelationalDefinition:
        """
        2. Relational Definition Management
        Evaluates whether an existing definition holds, expands, modifies, or coexists with a new perspective.
        Does NOT destroy existing definitions; preserves validity conditions and scale boundaries.
        """
        current_def = self.definitions.get(concept_key)
        friction = observation.untranslated_friction
        new_context = observation.conditions_and_context.get("context_name", "new_relational_realm")

        if not current_def:
            # Create a new relational definition anchored in observation conditions
            new_def = RelationalDefinition(
                concept_key=concept_key,
                primary_relational_meaning=f"Relational entity emerging from {observation.interacting_entities}",
                validity_conditions={new_context: f"Valid under {observation.validity_boundary}"},
                coexisting_perspectives={new_context: str(observation.observed_interaction)},
                state=DefinitionState.VALID_LOCAL
            )
            self.definitions[concept_key] = new_def
            return new_def

        # Evaluate friction and new context against existing conditions
        history_entry = {
            "timestamp": time.time(),
            "phenomenon_id": observation.phenomenon_id,
            "previous_state": current_def.state.value,
            "friction": friction
        }

        if friction < 0.3:
            # Definition holds valid under current conditions -> Extend applicability
            current_def.validity_conditions[new_context] = f"Extended validity under friction={friction:.2f}"
            current_def.state = DefinitionState.EXTENDED
            history_entry["action"] = "extended_validity"

        elif friction < 0.7:
            # Friction indicates new aspect -> Modify/Reinterpret definition while preserving local validity
            current_def.coexisting_perspectives[new_context] = f"Perspective under friction {friction:.2f}: {observation.observed_interaction}"
            current_def.state = DefinitionState.MODIFIED
            history_entry["action"] = "modified_and_added_perspective"

        else:
            # High friction -> Dual definition coexistence across scale boundaries or deep reinterpretation
            current_def.coexisting_perspectives[f"{new_context}_high_friction"] = (
                f"Contradictory/Coexisting aspect under high friction ({friction:.2f}): {observation.observed_interaction}"
            )
            current_def.state = DefinitionState.COEXISTING_DUAL
            history_entry["action"] = "established_coexisting_dual_perspectives"

            # Observer state shifts as a result of deep relational friction
            self.observer_criteria["sensitivity"] = min(1.0, self.observer_criteria["sensitivity"] + 0.05)
            self.observer_criteria["perceptual_resolution"] = min(2.0, self.observer_criteria["perceptual_resolution"] + 0.1)

        current_def.history_of_reinterpretation.append(history_entry)
        return current_def

    def generate_metacognitive_trace(
        self,
        observation: PhenomenologicalObservation,
        definition: RelationalDefinition
    ) -> RelationalMetacognitionTrace:
        """
        3. Non-Linear Relational Metacognition
        Tracks 5 relational aspects plus observer transformation without forcing a rigid 4-step sequence.
        """
        trace_id = f"trace_{len(self.metacognitive_traces) + 1}"

        origin_context = {
            "experience": observation.observed_interaction,
            "context": observation.conditions_and_context,
            "entities": observation.interacting_entities
        }

        boundary_premise = {
            "validity_boundary": observation.validity_boundary,
            "definition_state": definition.state.value,
            "observer_premise": observation.observer_state_at_observation["judgment_ground"]
        }

        world_interpretation = {
            "concept_key": definition.concept_key,
            "active_perspectives": dict(definition.coexisting_perspectives),
            "interpreted_meaning": definition.primary_relational_meaning
        }

        self_adaptation = {
            "definition_action": definition.history_of_reinterpretation[-1] if definition.history_of_reinterpretation else {},
            "actionable_insight": f"Adapted perception of '{definition.concept_key}' under {observation.conditions_and_context}"
        }

        observer_transformation = {
            "previous_criteria": observation.observer_state_at_observation,
            "current_criteria": dict(self.observer_criteria),
            "delta_sensitivity": self.observer_criteria["sensitivity"] - observation.observer_state_at_observation["sensitivity"],
            "delta_resolution": self.observer_criteria["perceptual_resolution"] - observation.observer_state_at_observation["perceptual_resolution"]
        }

        # Calculate non-linear correlations between aspects
        friction = observation.untranslated_friction
        correlations = {
            "origin_to_adaptation": math.tanh(friction * 1.5),
            "boundary_to_observer_transformation": math.sin(friction * math.pi / 2.0),
            "world_interpretation_flexibility": float(len(definition.coexisting_perspectives)) / 5.0
        }

        trace = RelationalMetacognitionTrace(
            trace_id=trace_id,
            origin_context=origin_context,
            boundary_premise=boundary_premise,
            world_interpretation=world_interpretation,
            self_adaptation=self_adaptation,
            observer_transformation=observer_transformation,
            non_linear_correlations=correlations
        )

        self.metacognitive_traces.append(trace)
        return trace

    def translate_to_expression(
        self,
        concept_key: str,
        target_modality: str = "linguistic_transformer"
    ) -> Dict[str, Any]:
        """
        4. Language / Transformer as Downstream Translation Sub-Lens
        Converts the relational definition and current metacognitive state into expressed language/tokens,
        positioned explicitly as a downstream sub-lens rather than the supreme cognitive controller.
        """
        definition = self.definitions.get(concept_key)
        if not definition:
            return {"modality": target_modality, "expressed_text": f"Unobserved entity '{concept_key}'"}

        latest_trace = self.metacognitive_traces[-1] if self.metacognitive_traces else None

        # Build translated expression reflecting relational conditions and observer state
        expressed_narrative = (
            f"[{concept_key.upper()} - Relational Expression]\n"
            f"Primary Relational Meaning: {definition.primary_relational_meaning}\n"
            f"Validity State: {definition.state.value}\n"
            f"Coexisting Perspectives: {list(definition.coexisting_perspectives.keys())}\n"
            f"Observer Ground: {self.observer_criteria['judgment_ground']} "
            f"(Sensitivity: {self.observer_criteria['sensitivity']:.2f}, Resolution: {self.observer_criteria['perceptual_resolution']:.2f})"
        )

        return {
            "modality": target_modality,
            "concept_key": concept_key,
            "expressed_text": expressed_narrative,
            "sub_lens_type": "downstream_translation",
            "relational_anchor_id": latest_trace.trace_id if latest_trace else "direct_observation"
        }
