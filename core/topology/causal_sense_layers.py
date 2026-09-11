"""
Causal Sense Layers
===================

Domain knowledge is treated as an internal sense layer, not as a flat fact store.
Language, mathematics, physics, chemistry, and microbial-scale cognition each
require their own causal receptors before the system can think in that domain.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from core.topology.informational_phase_observation import (
    ChromaticVector,
    InformationalPhaseObservationEngine,
    PhaseNodalProjection,
)


class CouplingResponse(Enum):
    ASSIMILATE = "assimilate"
    QUARANTINE = "quarantine"
    MUTATE = "mutate"
    REJECT = "reject"


@dataclass
class CausalPrinciple:
    """
    A domain-inherent generating principle.

    This is closer to a biological receptor motif than a stored proposition:
    it defines the kind of causal motion a layer can recognize.
    """

    name: str
    invariants: List[str]
    receptor_vector: np.ndarray
    chromatic: ChromaticVector
    scale_band: str
    expression_strength: float = 1.0

    def __post_init__(self):
        self.receptor_vector = _normalize(self.receptor_vector)


@dataclass
class ExternalCausalLigand:
    """External information prepared as a bindable causal signal."""

    ligand_id: str
    content: Any
    modality: str
    phase_projection: PhaseNodalProjection
    domain_hint: Optional[str] = None


@dataclass
class LayerCoupling:
    """Result of an external ligand binding against an internal causal layer."""

    layer_name: str
    ligand_id: str
    response: CouplingResponse
    affinity: float
    friction: float
    chromatic_shift: np.ndarray
    expressed_invariants: List[str] = field(default_factory=list)
    generated_principle: Optional[CausalPrinciple] = None


@dataclass
class CausalSenseLayer:
    """
    A scale/domain-specific sense organ.

    The layer does not merely classify incoming information. It exposes a
    receptor manifold that external information can reshape through resonance,
    friction, and chromatic displacement.
    """

    name: str
    scale_level: float
    domain: str
    principles: List[CausalPrinciple]
    permeability: float = 0.55
    plasticity: float = 0.15
    tension_memory: float = 0.0
    chromatic_baseline: np.ndarray = field(
        default_factory=lambda: np.array([0.33, 0.33, 0.34], dtype=np.float32)
    )

    def bind(self, ligand: ExternalCausalLigand) -> LayerCoupling:
        if not self.principles:
            return LayerCoupling(
                layer_name=self.name,
                ligand_id=ligand.ligand_id,
                response=CouplingResponse.QUARANTINE,
                affinity=0.0,
                friction=1.0,
                chromatic_shift=np.zeros(3, dtype=np.float32),
            )

        ligand_vector = _normalize(ligand.phase_projection.phase_vector)
        principle_scores = [
            float(np.dot(_resize(principle.receptor_vector, ligand_vector.size), ligand_vector))
            * principle.expression_strength
            for principle in self.principles
        ]
        best_index = int(np.argmax(principle_scores))
        best_principle = self.principles[best_index]

        affinity = float(np.clip(principle_scores[best_index], -1.0, 1.0))
        chromatic_shift = ligand.phase_projection.chromatic.to_array() - self.chromatic_baseline
        chromatic_pressure = float(np.linalg.norm(chromatic_shift))
        friction = float(np.clip((1.0 - max(0.0, affinity)) + chromatic_pressure, 0.0, 2.0))

        self.tension_memory = float(
            (1.0 - self.plasticity) * self.tension_memory + self.plasticity * friction
        )

        response = self._select_response(affinity, friction)
        generated = None

        if response is CouplingResponse.ASSIMILATE:
            best_principle.expression_strength = float(
                np.clip(best_principle.expression_strength + self.plasticity * affinity, 0.1, 3.0)
            )
            self.chromatic_baseline = _normalize_chromatic(
                self.chromatic_baseline + self.plasticity * chromatic_shift
            )
        elif response is CouplingResponse.MUTATE:
            generated = self._mutate_principle(best_principle, ligand_vector, ligand)
            self.principles.append(generated)

        return LayerCoupling(
            layer_name=self.name,
            ligand_id=ligand.ligand_id,
            response=response,
            affinity=affinity,
            friction=friction,
            chromatic_shift=chromatic_shift,
            expressed_invariants=list(best_principle.invariants),
            generated_principle=generated,
        )

    def _select_response(self, affinity: float, friction: float) -> CouplingResponse:
        assimilation_margin = affinity * self.permeability
        mutation_pressure = friction * self.plasticity

        if assimilation_margin >= 0.42 and friction <= 0.8:
            return CouplingResponse.ASSIMILATE
        if affinity >= 0.15 and mutation_pressure >= 0.12:
            return CouplingResponse.MUTATE
        if affinity < -0.15 or friction >= 1.45:
            return CouplingResponse.REJECT
        return CouplingResponse.QUARANTINE

    def _mutate_principle(
        self,
        parent: CausalPrinciple,
        ligand_vector: np.ndarray,
        ligand: ExternalCausalLigand,
    ) -> CausalPrinciple:
        parent_vector = _resize(parent.receptor_vector, ligand_vector.size)
        interference = _normalize((parent_vector + ligand_vector) * 0.5)
        if ligand_vector.size >= 3:
            orthogonal = np.cross(parent_vector[:3], ligand_vector[:3])
            norm = np.linalg.norm(orthogonal)
            if norm > 1e-8:
                interference[:3] = _normalize(interference[:3] + 0.1 * orthogonal / norm)

        invariants = list(dict.fromkeys(parent.invariants + [f"{ligand.modality}_coupling"]))
        return CausalPrinciple(
            name=f"{self.domain}_mutated_{len(self.principles) + 1}",
            invariants=invariants,
            receptor_vector=interference,
            chromatic=ligand.phase_projection.chromatic,
            scale_band=f"{self.scale_level:g}:{ligand.modality}",
            expression_strength=0.45,
        )


class CausalSenseLayerEngine:
    """Coordinates domain-specific causal sense layers."""

    def __init__(self, target_dimension: int = 8):
        self.phase_engine = InformationalPhaseObservationEngine(target_dimension=target_dimension)
        self.layers: Dict[str, CausalSenseLayer] = {}

    def register_layer(self, layer: CausalSenseLayer):
        self.layers[layer.name] = layer

    def ingest_ligand(
        self,
        ligand_id: str,
        content: Any,
        modality: str,
        chromatic: Optional[ChromaticVector] = None,
        domain_hint: Optional[str] = None,
    ) -> ExternalCausalLigand:
        projection = self.phase_engine.project_to_nodal_phase(
            node_id=ligand_id,
            raw_data=content,
            chromatic=chromatic,
            modality=modality,
        )
        return ExternalCausalLigand(
            ligand_id=ligand_id,
            content=content,
            modality=modality,
            phase_projection=projection,
            domain_hint=domain_hint,
        )

    def couple(self, ligand: ExternalCausalLigand) -> List[LayerCoupling]:
        candidate_layers: Iterable[CausalSenseLayer]
        if ligand.domain_hint and ligand.domain_hint in self.layers:
            candidate_layers = [self.layers[ligand.domain_hint]]
        else:
            candidate_layers = self.layers.values()

        return [layer.bind(ligand) for layer in candidate_layers]

    @classmethod
    def with_foundational_layers(cls, target_dimension: int = 8) -> "CausalSenseLayerEngine":
        engine = cls(target_dimension=target_dimension)
        for layer in _foundational_layers(target_dimension):
            engine.register_layer(layer)
        return engine


def _foundational_layers(target_dimension: int) -> List[CausalSenseLayer]:
    return [
        CausalSenseLayer(
            name="microbial_layer",
            scale_level=0.1,
            domain="microbial",
            permeability=0.45,
            principles=[
                _principle(
                    "gradient_survival",
                    ["chemotactic_gradient", "metabolic_boundary", "local_homeostasis"],
                    [0.8, 0.2, 0.6, 0.1],
                    target_dimension,
                    ChromaticVector(flux=1.3, order=0.7, entropy=0.25),
                    "micro",
                )
            ],
        ),
        CausalSenseLayer(
            name="language_layer",
            scale_level=1.0,
            domain="language",
            principles=[
                _principle(
                    "grammar_reference_flow",
                    ["syntax_dependency", "semantic_reference", "dialogic_context"],
                    [0.1, 0.9, 0.3, 0.7],
                    target_dimension,
                    ChromaticVector(flux=0.9, order=1.2, entropy=0.35),
                    "human-symbolic",
                )
            ],
        ),
        CausalSenseLayer(
            name="mathematics_layer",
            scale_level=1.2,
            domain="mathematics",
            principles=[
                _principle(
                    "invariant_transformation",
                    ["equivalence_preservation", "operator_closure", "proof_constraint"],
                    [0.2, 0.1, 1.0, 0.4],
                    target_dimension,
                    ChromaticVector(flux=0.5, order=1.7, entropy=0.1),
                    "formal",
                )
            ],
        ),
        CausalSenseLayer(
            name="physics_layer",
            scale_level=1.4,
            domain="physics",
            principles=[
                _principle(
                    "conserved_dynamical_flow",
                    ["energy_momentum_conservation", "force_field_gradient", "boundary_condition"],
                    [0.9, 0.4, 0.2, 1.0],
                    target_dimension,
                    ChromaticVector(flux=1.1, order=1.1, entropy=0.2),
                    "physical",
                )
            ],
        ),
        CausalSenseLayer(
            name="chemistry_layer",
            scale_level=1.3,
            domain="chemistry",
            principles=[
                _principle(
                    "bond_energy_reconfiguration",
                    ["valence_constraint", "reaction_pathway", "activation_energy"],
                    [0.5, 0.8, 0.7, 0.2],
                    target_dimension,
                    ChromaticVector(flux=1.0, order=1.0, entropy=0.3),
                    "molecular",
                )
            ],
        ),
    ]


def _principle(
    name: str,
    invariants: List[str],
    seed: List[float],
    target_dimension: int,
    chromatic: ChromaticVector,
    scale_band: str,
) -> CausalPrinciple:
    return CausalPrinciple(
        name=name,
        invariants=invariants,
        receptor_vector=_resize(np.array(seed, dtype=np.float32), target_dimension),
        chromatic=chromatic,
        scale_band=scale_band,
    )


def _resize(vector: np.ndarray, target_size: int) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).flatten()
    if arr.size == target_size:
        return arr
    if arr.size == 0:
        return np.zeros(target_size, dtype=np.float32)
    indices = np.linspace(0, arr.size - 1, target_size)
    return np.interp(indices, np.arange(arr.size), arr).astype(np.float32)


def _normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).flatten()
    norm = np.linalg.norm(arr)
    if norm <= 1e-8:
        return arr
    return arr / norm


def _normalize_chromatic(vector: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(vector, dtype=np.float32), 0.0, None)
    total = float(np.sum(arr))
    if total <= 1e-8:
        return np.array([0.33, 0.33, 0.34], dtype=np.float32)
    return arr / total
