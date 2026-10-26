"""
ELYSIA Ecological Taxonomy & Scale Coupling Engine v1.0
======================================================

Core Philosophy:
"Classification is not a static human-assigned label; it is the living topology
of resonance, substrate unity, and scale-coupled influence propagation."

This module implements the Ecological Taxonomy Tree & Scale Coupling Discernment Engine:
1. Modal Roots (ModalRoots):
   Tactile, Visual, Acoustic, and Symbolic sensory roots. Physical and cognitive stimuli
   branch naturally from these foundational perceptual origins rather than artificial labels.
2. Substrate Unity (SubstrateUnity):
   Underlying substance identity (e.g., H2O) remains invariant across macroscopic phase
   transitions (Ice, Liquid Water, Steam). State changes shift variance coordinates along
   spectrum axes while preserving core substrate invariance.
3. Scale Coupling Cascade & $E_{gate}$ Bifurcation (ScaleCouplingCascade):
   Evaluates impact propagation across scale hierarchies. Local fluctuations (hair/nail shedding)
   are damped locally without disturbing the macro core. Core breaches (blood/bone/organ damage)
   breach $E_{gate}$ pressure barriers and trigger global cascade re-alignment.
4. Topological Homeostasis & Distributed Local Discernment (TopologicalHomeostasis):
   Functions like the autonomic nervous system and white blood cells (micro-level autonomous agents).
   Local nodes resolve tension autonomously without top-down centralized dictation. Destructive
   metastasis is quarantined locally to preserve macro-structural homeostasis.
5. Cross-Modal Isomorphic Association (CrossModalAssociator):
   Discovers structural isomorphism between physical modalities (e.g., tactile surface friction/turbulence)
   and abstract symbolic models (e.g., non-linear equations), weaving associative synaptic beams
   for bidirectional cross-modal intuition.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid


class SensoryModality(Enum):
    TACTILE = "tactile"
    VISUAL = "visual"
    ACOUSTIC = "acoustic"
    SYMBOLIC = "symbolic"


class CascadeMode(Enum):
    LOCAL_DAMPED = "local_damped"          # Hair/nail shedding: absorbed locally as variance
    CORE_BREACH = "core_breach"            # Blood/bone/organ damage: breaches E_gate, triggers global cascade
    STATE_TRANSITION = "state_transition"  # Phase transition along invariant substrate axis


@dataclass
class SpectrumCoordinate:
    """Dynamic coordinates along multi-dimensional spectrum axes."""
    axis_name: str
    value: float
    variance_range: Tuple[float, float] = (0.0, 1.0)


@dataclass
class SubstrateIdentity:
    """
    Substrate Invariance representation (e.g., H2O, Cellular Membrane).
    Invariances hold across macroscopic state/phase transitions.
    """
    substrate_id: str
    chemical_or_logical_formula: str
    core_invariants: List[str]
    current_phase_state: str  # e.g., "solid", "liquid", "gas", "abstract_equation"
    spectrum_axes: Dict[str, SpectrumCoordinate] = field(default_factory=dict)

    def is_invariant_equivalent(self, other: 'SubstrateIdentity') -> bool:
        """Determines if two states share the exact same substrate core identity."""
        return (self.chemical_or_logical_formula == other.chemical_or_logical_formula and
                set(self.core_invariants) == set(other.core_invariants))


@dataclass
class EcologicalTaxonomyNode:
    """
    A living node in the Ecological Taxonomy Tree.
    Bound by sensory modality roots, substrate unity, and scale height.
    """
    node_id: str
    label: str
    modality: SensoryModality
    scale_level: float  # 0.0 = Micro/Cellular, 1.0 = Meso/Organ, 2.0 = Macro/Global Field
    substrate: SubstrateIdentity
    phase_vector: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    local_tension: float = 0.0
    quarantined: bool = False

    def __post_init__(self):
        if self.phase_vector.size == 0:
            self.phase_vector = np.zeros(8, dtype=np.float32)
        norm = np.linalg.norm(self.phase_vector)
        if norm > 1e-8:
            self.phase_vector = self.phase_vector / norm


@dataclass
class CrossModalAssociationBeam:
    """
    Synaptic link weaving physical sensory modalities with abstract symbolic models
    via topological isomorphism (structural similarity).
    """
    beam_id: str
    source_node_id: str
    target_node_id: str
    isomorphism_score: float
    shared_topological_invariants: List[str]
    bidirectional: bool = True


@dataclass
class CascadeAnalysisResult:
    """Result of influence propagation through the scale coupling cascade."""
    source_node_id: str
    initial_energy: float
    transmitted_energy: float
    e_gate_threshold: float
    cascade_mode: CascadeMode
    affected_scale_levels: List[float]
    global_alarm_triggered: bool
    description: str


class ModalRoots:
    """
    Sensory Roots Manager.
    Establishes foundational root trunks for Tactile, Visual, Acoustic, and Symbolic modalities.
    """

    def __init__(self, target_dimension: int = 8):
        self.target_dimension = target_dimension
        self.roots: Dict[SensoryModality, EcologicalTaxonomyNode] = {}
        self._init_foundational_roots()

    def _init_foundational_roots(self):
        for mod in SensoryModality:
            root_id = f"root_{mod.value}"
            dummy_substrate = SubstrateIdentity(
                substrate_id=f"substrate_root_{mod.value}",
                chemical_or_logical_formula=f"ROOT_{mod.value.upper()}",
                core_invariants=[f"{mod.value}_perception_continuity"],
                current_phase_state="foundational_root"
            )
            # Create orthogonal initial phase vectors for roots matching target_dimension
            vector = np.zeros(self.target_dimension, dtype=np.float32)
            idx = list(SensoryModality).index(mod)
            if idx < self.target_dimension:
                vector[idx] = 1.0

            node = EcologicalTaxonomyNode(
                node_id=root_id,
                label=f"{mod.value.capitalize()} Root Trunk",
                modality=mod,
                scale_level=0.0,
                substrate=dummy_substrate,
                phase_vector=vector
            )
            self.roots[mod] = node

    def get_root(self, modality: SensoryModality) -> EcologicalTaxonomyNode:
        return self.roots[modality]


class SubstrateUnityEngine:
    """
    Manages substrate invariance across macroscopic phase state changes.
    E.g., H2O as Solid (Ice), Liquid (Water), Gas (Steam).
    All share the same SubstrateIdentity core, while coordinates along temperature/pressure axes shift.
    """

    def __init__(self):
        self.registered_substrates: Dict[str, SubstrateIdentity] = {}

    def register_substrate(self, substrate: SubstrateIdentity):
        self.registered_substrates[substrate.substrate_id] = substrate

    def evaluate_state_transition(
        self,
        substrate_a: SubstrateIdentity,
        substrate_b: SubstrateIdentity
    ) -> Tuple[bool, str]:
        """
        Determines if transition between A and B is a substrate-invariant phase state change,
        or a fundamental substrate disruption.
        """
        if substrate_a.is_invariant_equivalent(substrate_b):
            msg = (f"Substrate Unity Preserved: '{substrate_a.chemical_or_logical_formula}' "
                   f"transitioned phase from [{substrate_a.current_phase_state}] -> [{substrate_b.current_phase_state}]. "
                   f"Invariance holds, variance coordinates shifted.")
            return True, msg
        else:
            msg = (f"Substrate Disruption: Formula '{substrate_a.chemical_or_logical_formula}' vs "
                   f"'{substrate_b.chemical_or_logical_formula}' do not share core invariants.")
            return False, msg


class ScaleCouplingCascade:
    """
    Evaluates scale influence propagation and $E_{gate}$ bifurcation points.

    Damps local perturbations (e.g., hair/nail shedding) exponentially across scale levels:
    E_transmitted = E_initial * exp(-gamma * delta_scale)

    If E_transmitted >= E_gate or direct substrate breach occurs, triggers CORE_BREACH
    and activates global homeostatic alarm.
    """

    def __init__(self, damping_factor: float = 2.5, default_e_gate: float = 0.6):
        self.damping_factor = damping_factor
        self.default_e_gate = default_e_gate

    def evaluate_perturbation(
        self,
        source_node: EcologicalTaxonomyNode,
        perturbation_energy: float,
        target_scale_level: float = 2.0,
        direct_core_damage: bool = False
    ) -> CascadeAnalysisResult:
        delta_scale = max(0.0, target_scale_level - source_node.scale_level)
        transmitted_energy = perturbation_energy * np.exp(-self.damping_factor * delta_scale)
        e_gate = self.default_e_gate

        affected_scales = [source_node.scale_level]
        if transmitted_energy >= 0.1:
            affected_scales.append(source_node.scale_level + delta_scale * 0.5)

        if direct_core_damage or transmitted_energy >= e_gate:
            cascade_mode = CascadeMode.CORE_BREACH
            affected_scales.append(target_scale_level)
            global_alarm = True
            desc = (f"CORE BREACH: Perturbation (energy={perturbation_energy:.2f}) at scale {source_node.scale_level} "
                    f"transmitted {transmitted_energy:.3f} >= E_gate ({e_gate:.2f}). "
                    f"Direct core damage={direct_core_damage}. Global homeostatic alarm triggered!")
        else:
            cascade_mode = CascadeMode.LOCAL_DAMPED
            global_alarm = False
            desc = (f"LOCAL DAMPED: Perturbation (energy={perturbation_energy:.2f}) at scale {source_node.scale_level} "
                    f"damped to {transmitted_energy:.3f} < E_gate ({e_gate:.2f}). "
                    f"Absorbed locally like hair/nail shedding without disturbing macro core.")

        return CascadeAnalysisResult(
            source_node_id=source_node.node_id,
            initial_energy=perturbation_energy,
            transmitted_energy=transmitted_energy,
            e_gate_threshold=e_gate,
            cascade_mode=cascade_mode,
            affected_scale_levels=sorted(list(set(affected_scales))),
            global_alarm_triggered=global_alarm,
            description=desc
        )


class TopologicalHomeostasis:
    """
    Distributed Autonomous Local Discernment & Defense.
    Functions like the autonomic nervous system and local immune cells (e.g. white blood cells / phagocytes).
    Local micro-nodes detect and resolve tension autonomously without requiring top-down macro dictation.
    Destructive local metastasis/mutations are quarantined locally to preserve global structural equilibrium.
    """

    def __init__(self, local_quarantine_threshold: float = 0.75):
        self.local_quarantine_threshold = local_quarantine_threshold
        self.quarantined_nodes: Set[str] = set()
        self.homeostatic_realignments: List[Dict[str, Any]] = []

    def inspect_and_balance_node(
        self,
        node: EcologicalTaxonomyNode,
        stimulus_friction: float
    ) -> Dict[str, Any]:
        """
        Executes local micro-level discernment and homeostatic balancing on a single node.
        """
        node.local_tension = float(0.8 * node.local_tension + 0.2 * stimulus_friction)

        action = "BALANCED_LOCALLY"
        notes = f"Local node '{node.label}' absorbed tension ({node.local_tension:.3f}) autonomously."

        if node.local_tension >= self.local_quarantine_threshold:
            node.quarantined = True
            self.quarantined_nodes.add(node.node_id)
            action = "LOCAL_QUARANTINE"
            notes = (f"Local node '{node.label}' tension ({node.local_tension:.3f}) exceeded threshold ({self.local_quarantine_threshold:.2f}). "
                     f"Executed local autonomous quarantine (phagocytic containment) to protect macro homeostasis.")

        record = {
            "node_id": node.node_id,
            "label": node.label,
            "tension": node.local_tension,
            "action": action,
            "quarantined": node.quarantined,
            "notes": notes
        }
        self.homeostatic_realignments.append(record)
        return record


class CrossModalAssociator:
    """
    Discovers topological isomorphism between physical sensory modalities (tactile/visual/acoustic)
    and abstract symbolic models (equations/DNA/logic), weaving associative synaptic beams.
    Enables bidirectional intuition (e.g., surface friction <-> non-linear turbulence equations).
    """

    def __init__(self, isomorphism_threshold: float = 0.5):
        self.isomorphism_threshold = isomorphism_threshold
        self.association_beams: Dict[str, CrossModalAssociationBeam] = {}

    def discover_and_weave_association(
        self,
        node_a: EcologicalTaxonomyNode,
        node_b: EcologicalTaxonomyNode,
        topological_invariants_a: List[str],
        topological_invariants_b: List[str]
    ) -> Optional[CrossModalAssociationBeam]:
        """
        Calculates topological isomorphism based on phase vector inner product and shared topological invariants.
        If threshold is met, weaves a bidirectional CrossModalAssociationBeam.
        """
        if node_a.modality == node_b.modality:
            # Cross-modal association is specifically between different modalities
            return None

        phase_sim = float(np.dot(node_a.phase_vector, node_b.phase_vector))
        shared_invariants = list(set(topological_invariants_a).intersection(set(topological_invariants_b)))

        # Combined isomorphism score
        invariant_score = len(shared_invariants) / max(1, max(len(topological_invariants_a), len(topological_invariants_b)))
        total_isomorphism = float(0.5 * max(0.0, phase_sim) + 0.5 * invariant_score)

        if total_isomorphism >= self.isomorphism_threshold:
            beam_id = f"beam_{node_a.node_id}_{node_b.node_id}"
            beam = CrossModalAssociationBeam(
                beam_id=beam_id,
                source_node_id=node_a.node_id,
                target_node_id=node_b.node_id,
                isomorphism_score=total_isomorphism,
                shared_topological_invariants=shared_invariants,
                bidirectional=True
            )
            self.association_beams[beam_id] = beam
            return beam

        return None


class EcologicalTaxonomyEngine:
    """
    Main Orchestrator for the Ecological Cognitive Taxonomy & Scale Coupling System.

    Integrates Modal Roots, Substrate Unity, Scale Coupling Cascade,
    Topological Homeostasis, and Cross-Modal Association.
    """

    def __init__(self, target_dimension: int = 8):
        self.target_dimension = target_dimension
        self.modal_roots = ModalRoots(target_dimension=target_dimension)
        self.substrate_engine = SubstrateUnityEngine()
        self.cascade_engine = ScaleCouplingCascade()
        self.homeostasis_engine = TopologicalHomeostasis()
        self.associator = CrossModalAssociator()

        self.nodes: Dict[str, EcologicalTaxonomyNode] = {}
        # Register foundational roots into active node tree
        for mod, root_node in self.modal_roots.roots.items():
            self.nodes[root_node.node_id] = root_node

    def add_taxonomy_node(
        self,
        label: str,
        modality: SensoryModality,
        substrate: SubstrateIdentity,
        scale_level: float = 0.5,
        phase_vector: Optional[np.ndarray] = None,
        parent_id: Optional[str] = None
    ) -> EcologicalTaxonomyNode:
        """
        Sprouts a new node on the Ecological Taxonomy Tree under the appropriate sensory root or parent node.
        """
        node_id = f"node_{uuid.uuid4().hex[:8]}"
        if phase_vector is None:
            phase_vector = np.random.randn(self.target_dimension).astype(np.float32)

        actual_parent_id = parent_id or self.modal_roots.get_root(modality).node_id

        node = EcologicalTaxonomyNode(
            node_id=node_id,
            label=label,
            modality=modality,
            scale_level=scale_level,
            substrate=substrate,
            phase_vector=phase_vector,
            parent_id=actual_parent_id
        )

        self.nodes[node_id] = node
        if actual_parent_id in self.nodes:
            self.nodes[actual_parent_id].children_ids.append(node_id)

        self.substrate_engine.register_substrate(substrate)
        return node

    def process_incoming_stimulus(
        self,
        target_node_id: str,
        stimulus_energy: float,
        stimulus_friction: float,
        direct_core_damage: bool = False
    ) -> Dict[str, Any]:
        """
        Processes an incoming stimulus through the Ecological Taxonomy System:
        1. Local Homeostasis inspection on target node (Autonomous Local Discernment).
        2. Scale Coupling Cascade evaluation (Hair shedding vs. Blood/Bone breach).
        3. Returns complete ecological discernment payload.
        """
        target_node = self.nodes.get(target_node_id)
        if not target_node:
            raise ValueError(f"Taxonomy node '{target_node_id}' not found.")

        # Step 1: Local Autonomous Discernment
        homeostasis_record = self.homeostasis_engine.inspect_and_balance_node(target_node, stimulus_friction)

        # Step 2: Scale Coupling Cascade
        cascade_result = self.cascade_engine.evaluate_perturbation(
            source_node=target_node,
            perturbation_energy=stimulus_energy,
            target_scale_level=2.0,
            direct_core_damage=direct_core_damage
        )

        return {
            "target_node_id": target_node_id,
            "label": target_node.label,
            "modality": target_node.modality.value,
            "substrate_formula": target_node.substrate.chemical_or_logical_formula,
            "homeostasis": homeostasis_record,
            "cascade": cascade_result,
            "discernment": "CORE_ALERT" if cascade_result.global_alarm_triggered else "LOCAL_BALANCED"
        }

    def associate_cross_modal_nodes(
        self,
        node_id_a: str,
        node_id_b: str,
        invariants_a: List[str],
        invariants_b: List[str]
    ) -> Optional[CrossModalAssociationBeam]:
        """Discovers isomorphism and weaves a cross-modal association beam between two nodes."""
        node_a = self.nodes.get(node_id_a)
        node_b = self.nodes.get(node_id_b)
        if not node_a or not node_b:
            return None

        return self.associator.discover_and_weave_association(
            node_a, node_b, invariants_a, invariants_b
        )
