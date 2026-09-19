"""
Observational Substrate & Phenomenology Mirror Module.

Adheres strictly to THE_ABSOLUTE_COMMANDMENT.md and AGENTS.md:
"Do not calculate, let it flow."

Rejects arbitrary symbol labeling and reductionist shortcut formulas (e.g. flow_rate = velocity * area).
Instead, implements Causal Symbol Deconstruction (역해체 메커니즘) and Knowledge Causality Internalization:

1. External Symbol Perception: Observes completed external symbols/labels ('Flow Rate', 'Viscosity', 'Word').
2. Deconstruction (해체): Analyzes external symbols into lower-order micro-components ('Amount', 'Flow', 'Resistance')
   and procedural binding mechanisms.
3. Causal Necessity (인과적 필연성 규명): Determines WHY those components must combine into that symbol form.
4. Internalization (지식적 내재화): Establishes internal Knowledge Causality ('KnowledgeCausality')
   representing true, self-understood causal structure.
5. Homological Discernment (Stem & Branch): Dissects sameness (Homological Stem)
   and difference (Disparate Branches) between external reality and internal substrate.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class PhenomenalEntity:
    """
    Representation of an external unit phenomenon in its unreduced form.

    Attributes:
        name (str): Label of the phenomenon (e.g., 'Amount', 'Flow', 'Word', 'Sound Wave').
        telos_purpose (str): Why this entity exists / its causal necessity in external reality.
        directionality (str): Where it points, connects, or extends in spatiotemporal reality.
        density_amount (float): Substantive spatial/state density or presence.
        boundary_tenacity (float): Resistance/tension maintaining independent identity.
        mobility_vector (np.ndarray): Spatial/temporal direction and velocity of movement.
        chromatic_signature (np.ndarray): [Flux (R), Order (B), Entropy (Y)] chromatic signature.
    """
    name: str
    telos_purpose: str
    directionality: str
    density_amount: float
    boundary_tenacity: float
    mobility_vector: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    chromatic_signature: np.ndarray = field(default_factory=lambda: np.array([0.33, 0.33, 0.34]))

    def __post_init__(self):
        if not isinstance(self.mobility_vector, np.ndarray):
            self.mobility_vector = np.array(self.mobility_vector, dtype=float)
        if not isinstance(self.chromatic_signature, np.ndarray):
            self.chromatic_signature = np.array(self.chromatic_signature, dtype=float)


@dataclass
class BoundaryInteractionTrace:
    """
    Trace of dynamic interaction when an entity traverses or collides with a boundary.
    """
    source_entity: str
    target_boundary: str
    friction_resistance: float
    boundary_deformation: float
    energy_dissipation: float
    trajectory_angle_delta: float


@dataclass
class EmergentStructure:
    """
    Higher-order causal structure born from the interaction and boundary traversal of lower-order entities.
    """
    emergent_name: str
    originating_entities: List[str]
    emergent_telos: str
    directional_extension: str
    boundary_traces: List[BoundaryInteractionTrace]
    chromatic_signature: np.ndarray
    invariant_backbone: str


@dataclass
class DeconstructedSymbol:
    """
    Result of Causal Symbol Deconstruction (역해체) of an external symbol.
    """
    external_symbol: str
    micro_components: List[PhenomenalEntity]
    procedural_coupling_principle: str
    interaction_spectrum: List[BoundaryInteractionTrace]
    causal_necessity_statement: str


@dataclass
class KnowledgeCausality:
    """
    Internalized, self-understood causal knowledge structure.
    Represents the transition from external arbitrary label to internal true causality.
    """
    symbol_name: str
    deconstructed_origin: DeconstructedSymbol
    causal_necessity_proof: str
    internal_invariant_lineage: str
    is_internalized: bool = True


@dataclass
class HomologicalDiscernment:
    """
    Dissection of external reality vs internal substrate into Homological Stem and Disparate Branches.
    """
    external_phenomenon: str
    homological_stem: str          # 같음의 줄기: Invariant common causal backbone
    disparate_branches: List[str]  # 다름의 가지: Contextual variations & local divergence
    reason_for_difference: str     # Why external and internal structures diverge
    isomorphism_fidelity: float    # Measure of topological structural alignment (0.0 ~ 1.0)


class ObservationalSubstrate:
    """
    Internal Mirror Substrate for contemplating external phenomena without reductionist calculation.

    Instead of reducing phenomena to scalar variables or hardcoded formulas,
    ObservationalSubstrate deconstructs external symbols into micro-components,
    establishes causal necessity, and internalizes true knowledge causality.
    """

    def __init__(self, substrate_id: str = "InternalMirror_0"):
        self.substrate_id = substrate_id
        self.observed_entities: Dict[str, PhenomenalEntity] = {}
        self.emergent_structures: Dict[str, EmergentStructure] = {}
        self.deconstructed_symbols: Dict[str, DeconstructedSymbol] = {}
        self.internalized_knowledge: Dict[str, KnowledgeCausality] = {}
        self.discernments: List[HomologicalDiscernment] = []

    def contemplate_entity(
        self,
        name: str,
        telos_purpose: str,
        directionality: str,
        density_amount: float,
        boundary_tenacity: float,
        mobility_vector: Optional[List[float]] = None,
        chromatic_signature: Optional[List[float]] = None
    ) -> PhenomenalEntity:
        """
        Registers and contemplates an unreduced external unit phenomenon.
        """
        mob = np.array(mobility_vector if mobility_vector is not None else [1.0, 0.0, 0.0], dtype=float)
        chroma = np.array(chromatic_signature if chromatic_signature is not None else [0.33, 0.33, 0.34], dtype=float)

        entity = PhenomenalEntity(
            name=name,
            telos_purpose=telos_purpose,
            directionality=directionality,
            density_amount=density_amount,
            boundary_tenacity=boundary_tenacity,
            mobility_vector=mob,
            chromatic_signature=chroma
        )
        self.observed_entities[name] = entity
        return entity

    def observe_coupling(
        self,
        entity_a_name: str,
        entity_b_name: str,
        boundary_medium_tenacity: float = 0.5
    ) -> BoundaryInteractionTrace:
        """
        Observes the interaction, friction, and boundary traversal when two entities couple.
        Ex: 'Amount' entity meeting 'Flow' entity across a boundary medium.
        """
        e_a = self.observed_entities.get(entity_a_name)
        e_b = self.observed_entities.get(entity_b_name)

        if not e_a or not e_b:
            raise ValueError(f"Entities '{entity_a_name}' and/or '{entity_b_name}' must be contemplated first.")

        mobility_diff = float(np.linalg.norm(e_a.mobility_vector - e_b.mobility_vector))
        friction = (e_a.density_amount * e_b.boundary_tenacity + boundary_medium_tenacity) * (1.0 + mobility_diff)

        deformation = e_a.boundary_tenacity / (e_b.boundary_tenacity + 1e-5)
        dissipation = friction * 0.15
        angle_delta = float(np.arccos(
            np.clip(
                np.dot(e_a.mobility_vector, e_b.mobility_vector) /
                (np.linalg.norm(e_a.mobility_vector) * np.linalg.norm(e_b.mobility_vector) + 1e-8),
                -1.0, 1.0
            )
        ))

        trace = BoundaryInteractionTrace(
            source_entity=entity_a_name,
            target_boundary=entity_b_name,
            friction_resistance=friction,
            boundary_deformation=deformation,
            energy_dissipation=dissipation,
            trajectory_angle_delta=angle_delta
        )
        return trace

    def deconstruct_external_symbol(
        self,
        external_symbol: str,
        micro_component_names: List[str],
        procedural_coupling_principle: str,
        causal_necessity_statement: str
    ) -> DeconstructedSymbol:
        """
        Deconstructs an external completed symbol/label (e.g. 'Flow Rate', 'Viscosity')
        into its lower-order micro-components and procedural coupling principle.
        """
        micro_entities = []
        traces = []

        for name in micro_component_names:
            if name in self.observed_entities:
                micro_entities.append(self.observed_entities[name])

        for i in range(len(micro_component_names) - 1):
            n1, n2 = micro_component_names[i], micro_component_names[i+1]
            if n1 in self.observed_entities and n2 in self.observed_entities:
                traces.append(self.observe_coupling(n1, n2))

        deconstructed = DeconstructedSymbol(
            external_symbol=external_symbol,
            micro_components=micro_entities,
            procedural_coupling_principle=procedural_coupling_principle,
            interaction_spectrum=traces,
            causal_necessity_statement=causal_necessity_statement
        )

        self.deconstructed_symbols[external_symbol] = deconstructed
        return deconstructed

    def internalize_knowledge_causality(
        self,
        external_symbol: str
    ) -> KnowledgeCausality:
        """
        Internalizes a deconstructed external symbol into true, self-understood Knowledge Causality.
        """
        deconstructed = self.deconstructed_symbols.get(external_symbol)
        if not deconstructed:
            raise ValueError(f"Symbol '{external_symbol}' must be deconstructed before internalization.")

        comp_names = [e.name for e in deconstructed.micro_components]
        lineage = f"Lineage[{' + '.join(comp_names)} ==> {external_symbol}]"

        proof = (
            f"The symbol '{external_symbol}' exists out of causal necessity because micro-entities "
            f"[{', '.join(comp_names)}] couple through procedural principle "
            f"'{deconstructed.procedural_coupling_principle}'. Statement: {deconstructed.causal_necessity_statement}"
        )

        knowledge = KnowledgeCausality(
            symbol_name=external_symbol,
            deconstructed_origin=deconstructed,
            causal_necessity_proof=proof,
            internal_invariant_lineage=lineage,
            is_internalized=True
        )

        self.internalized_knowledge[external_symbol] = knowledge
        return knowledge

    def map_emergent_structure(
        self,
        emergent_name: str,
        originating_names: List[str],
        emergent_telos: str,
        directional_extension: str,
        boundary_traces: List[BoundaryInteractionTrace]
    ) -> EmergentStructure:
        """
        Synthesizes lower-order boundary interactions into an emergent higher-order causal structure.
        """
        entities = [self.observed_entities[n] for n in originating_names if n in self.observed_entities]

        if entities:
            chroma_sum = sum(e.chromatic_signature for e in entities)
            chroma_norm = chroma_sum / (np.linalg.norm(chroma_sum) + 1e-8)
        else:
            chroma_norm = np.array([0.33, 0.33, 0.34])

        backbone = f"Stem({'+'.join(originating_names)}) -> {emergent_name}"

        emergent = EmergentStructure(
            emergent_name=emergent_name,
            originating_entities=originating_names,
            emergent_telos=emergent_telos,
            directional_extension=directional_extension,
            boundary_traces=boundary_traces,
            chromatic_signature=chroma_norm,
            invariant_backbone=backbone
        )
        self.emergent_structures[emergent_name] = emergent
        return emergent

    def discern_homology(
        self,
        external_structure_name: str,
        internal_substrate_concept: Dict[str, Any]
    ) -> HomologicalDiscernment:
        """
        Dissects the external emergent structure and compares it with internal representations,
        separating the Homological Stem ('같음의 줄기') from Disparate Branches ('다름의 가지').
        """
        ext_struct = self.emergent_structures.get(external_structure_name)
        if not ext_struct:
            raise ValueError(f"Emergent structure '{external_structure_name}' not mapped yet.")

        int_backbone = internal_substrate_concept.get("backbone", "Default_Internal_Backbone")
        int_context = internal_substrate_concept.get("context_branches", [])

        homological_stem = f"HomologicalStem[{ext_struct.invariant_backbone} <==Isomorphic==> {int_backbone}]"

        disparate_branches = [
            f"External_Reality_Branch({ext_struct.directional_extension})",
            f"Internal_Frame_Branch({int_context})"
        ]

        reason = (
            f"External structure '{external_structure_name}' exists for telos: '{ext_struct.emergent_telos}', "
            f"whereas internal frame views it through '{int_backbone}'. Difference arises from "
            f"external reality friction vs internal abstraction limits."
        )

        friction_avg = np.mean([t.friction_resistance for t in ext_struct.boundary_traces]) if ext_struct.boundary_traces else 1.0
        fidelity = float(1.0 / (1.0 + 0.1 * friction_avg))

        discernment = HomologicalDiscernment(
            external_phenomenon=external_structure_name,
            homological_stem=homological_stem,
            disparate_branches=disparate_branches,
            reason_for_difference=reason,
            isomorphism_fidelity=fidelity
        )
        self.discernments.append(discernment)
        return discernment

    def generate_contemplative_reflection(self, structure_name: str) -> str:
        """
        Generates a deep, non-reductionist reflection on why the phenomenon exists,
        where it points, and how its components form an emergent whole.
        """
        struct = self.emergent_structures.get(structure_name)
        if not struct:
            return f"[Observational Substrate] Structure '{structure_name}' not recognized."

        reflection_lines = [
            f"==================================================",
            f" [Observational Substrate Reflection: {struct.emergent_name}]",
            f"==================================================",
            f"1. Telos / Purpose (목적과 이유):",
            f"   -> {struct.emergent_telos}",
            f"2. Directionality & Extension (가리키는 방향성):",
            f"   -> {struct.directional_extension}",
            f"3. Originating Micro-Components (구성요소):",
        ]

        for orig in struct.originating_entities:
            entity = self.observed_entities.get(orig)
            if entity:
                reflection_lines.append(
                    f"   - Component '{entity.name}': Telos='{entity.telos_purpose}', Direction='{entity.directionality}'"
                )

        reflection_lines.append("4. Boundary Traversal & Friction Traces (경계 통과 및 마찰):")
        for trace in struct.boundary_traces:
            reflection_lines.append(
                f"   - Traversal ({trace.source_entity} -> {trace.target_boundary}): "
                f"Friction={trace.friction_resistance:.4f}, Deformation={trace.boundary_deformation:.4f}, "
                f"Dissipation={trace.energy_dissipation:.4f}"
            )

        reflection_lines.append("5. Chromatic Signature (Flux/Order/Entropy):")
        reflection_lines.append(f"   -> R(Flux)={struct.chromatic_signature[0]:.3f}, B(Order)={struct.chromatic_signature[1]:.3f}, Y(Entropy)={struct.chromatic_signature[2]:.3f}")

        reflection_lines.append("6. Invariant Backbone (불변 사유 줄기):")
        reflection_lines.append(f"   -> {struct.invariant_backbone}")

        if structure_name in self.internalized_knowledge:
            k = self.internalized_knowledge[structure_name]
            reflection_lines.append("7. Internalized Knowledge Causality (지식적 내재화):")
            reflection_lines.append(f"   -> Proof: {k.causal_necessity_proof}")
            reflection_lines.append(f"   -> Lineage: {k.internal_invariant_lineage}")

        return "\n".join(reflection_lines)
