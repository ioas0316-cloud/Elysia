r"""
Three-Layer Engine Architecture & Causal Isolation Boundary Enforcement.

This module implements the 3-Layer Engine Layering (Engine Layer Architecture)
to prevent confounding geometric numerical operations with high-order symbolic cognition,
enforcing causal isolation, information autonomy, symbolic back-tracing,
and cross-dimensional multi-causal bridging.

Architecture:
    1. Layer 0: Geometric / Physical State Layer (Stateless, dot products, FOV, normal/distance)
    2. Layer 1: Topological Dynamics Layer (Temporal conservation, phase delta, friction, sealed attractors, C_lens bandwidth limit)
    3. Layer 2: Symbolic & Narrative Cognition Layer (Narrative causality, self-identity I_c, symbolic back-tracing without loss backpropagation)

Boundary Rules:
    - Upward Observability: Layer 0 signals pass upward only as ObservationSignal.
    - No Downward Reduction: Layer 2 decisions propagate down as ControlDirective only; no numeric scalar reduction or formula overwriting.
    - Level-bounded Isomorphism: Isomorphism mappings restricted to equivalent causal levels.
    - Information Irreversibility Protection: Forbids collapsing symbolic context into scalar float loss.
"""

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


# =====================================================================
# Custom Exceptions for Boundary & Isomorphism Violations
# =====================================================================

class CategoryError(TypeError):
    """Raised when an operation commits a category error across distinct abstraction levels."""
    pass


class ReductionViolationError(ValueError):
    """Raised when downward reduction or numerical overwriting of higher symbolic context is attempted."""
    pass


class IrreversibleReductionError(ValueError):
    """Raised when high-order narrative causality is collapsed into a scalar loss value."""
    pass


# =====================================================================
# Inter-Layer Signals & Directives
# =====================================================================

@dataclass(frozen=True)
class ObservationSignal:
    """Upward-only signal envelope from Layer 0 to Layer 1/2."""
    source_layer: int
    signal_type: str
    data: Dict[str, Any]
    timestamp: float = 0.0


@dataclass(frozen=True)
class ControlDirective:
    """Downward-only directive from Layer 2 to Layer 0/1."""
    target_layer: int
    action_type: str
    parameters: Dict[str, Any]


@dataclass
class SealedAttractor:
    """Structure to seal and isolate conflicting topological or symbolic attractors."""
    attractor_id: str
    payload: Any
    reason: str


# =====================================================================
# Layer 0: Geometric / Physical State Layer (Stateless)
# =====================================================================

class Layer0GeometricState:
    """
    Layer 0: Pure stateless geometric and physical state operations.
    Calculates vector inner products, field-of-view (FOV), distances, and normals.
    Contains no history, trauma, or linguistic context.
    """
    def __init__(self):
        self.layer_id = 0

    def dot_product(self, vec_a: List[float], vec_b: List[float]) -> float:
        if len(vec_a) != len(vec_b):
            raise ValueError("Vector dimensions must match for dot product.")
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def euclidean_distance(self, pt_a: List[float], pt_b: List[float]) -> float:
        if len(pt_a) != len(pt_b):
            raise ValueError("Point dimensions must match for distance calculation.")
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pt_a, pt_b)))

    def is_in_field_of_view(
        self,
        forward_vec: List[float],
        target_vec: List[float],
        fov_angle_deg: float
    ) -> bool:
        dot = self.dot_product(forward_vec, target_vec)
        mag_f = math.sqrt(sum(x ** 2 for x in forward_vec))
        mag_t = math.sqrt(sum(x ** 2 for x in target_vec))
        if mag_f == 0 or mag_t == 0:
            return False
        cos_theta = max(-1.0, min(1.0, dot / (mag_f * mag_t)))
        angle_rad = math.acos(cos_theta)
        return angle_rad <= math.radians(fov_angle_deg / 2.0)

    def compute_normal_vector(self, surface_vec_a: List[float], surface_vec_b: List[float]) -> List[float]:
        if len(surface_vec_a) != 3 or len(surface_vec_b) != 3:
            raise ValueError("Normal vector computation requires 3D vectors.")
        nx = surface_vec_a[1] * surface_vec_b[2] - surface_vec_a[2] * surface_vec_b[1]
        ny = surface_vec_a[2] * surface_vec_b[0] - surface_vec_a[0] * surface_vec_b[2]
        nz = surface_vec_a[0] * surface_vec_b[1] - surface_vec_a[1] * surface_vec_b[0]
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm == 0:
            return [0.0, 0.0, 0.0]
        return [nx / norm, ny / norm, nz / norm]

    def emit_observation(self, signal_type: str, raw_data: Dict[str, Any]) -> ObservationSignal:
        return ObservationSignal(
            source_layer=0,
            signal_type=signal_type,
            data=raw_data
        )


# =====================================================================
# Layer 1: Topological Dynamics Layer
# =====================================================================

class Layer1TopologicalDynamics:
    r"""
    Layer 1: Dynamic topological conservation.
    Absorbs external impacts and lower state changes through phase delta (\Delta\theta),
    friction tension (V_t), and SealedAttractors under lens bandwidth limits (C_lens).
    """
    def __init__(self, c_lens_bandwidth: float = 100.0, v_critical: float = 10.0):
        self.layer_id = 1
        self.c_lens_bandwidth = c_lens_bandwidth
        self.v_critical = v_critical
        self.phase_delta: float = 0.0
        self.friction_tension: float = 0.0
        self.sealed_attractors: List[SealedAttractor] = []

    def process_observation(self, signal: ObservationSignal) -> Dict[str, Any]:
        if signal.source_layer != 0:
            raise CategoryError(f"Layer 1 expected signal from Layer 0, got Layer {signal.source_layer}")

        # Calculate impact from signal data
        impact_magnitude = float(signal.data.get("magnitude", 1.0))

        # Enforce C_lens bandwidth protection limit
        clamped_impact = min(impact_magnitude, self.c_lens_bandwidth)

        self.phase_delta = (self.phase_delta + clamped_impact * 0.1) % (2 * math.pi)
        self.friction_tension += clamped_impact * 0.5

        status = "STABLE"
        if self.friction_tension >= self.v_critical:
            # Seal system fracture
            attractor = SealedAttractor(
                attractor_id=f"SEAL_{len(self.sealed_attractors) + 1}",
                payload=signal.data,
                reason="Friction tension exceeded critical limit V_critical"
            )
            self.sealed_attractors.append(attractor)
            self.friction_tension = self.v_critical * 0.5  # Damped / relieved
            status = "SEALED_EXCESS_TENSION"

        return {
            "status": status,
            "phase_delta": self.phase_delta,
            "friction_tension": self.friction_tension,
            "sealed_count": len(self.sealed_attractors)
        }


# =====================================================================
# Layer 2: Symbolic & Narrative Cognition Layer
# =====================================================================

class SymbolState(Enum):
    RESONATING = auto()
    SEALED = auto()
    REINTEGRATED = auto()
    CONTRADICTED = auto()


@dataclass(frozen=True)
class CausalLinguisticSymbol:
    """Linguistic symbol maintaining parent history for reversible back-tracing."""
    symbol: str
    causal_tension: float
    required_context_depth: float
    parents: Tuple[str, ...] = ()
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class CausalTraceNode:
    """Trace node preserving narrative causality in reasoning trajectories."""
    symbol_data: CausalLinguisticSymbol
    phase_delta: float
    friction: float
    incoming_causal_links: List[str] = field(default_factory=list)


class SymbolicBackTracer:
    """Mechanism for reverse traversing narrative causal graphs using feedback symbols without scalar loss."""
    def __init__(self, interface_ref: Any):
        self.interface = interface_ref
        self.causal_graph: Dict[str, CausalTraceNode] = {}

    def register_trace(self, symbol_data: CausalLinguisticSymbol, phase_delta: float):
        node = CausalTraceNode(
            symbol_data=symbol_data,
            phase_delta=phase_delta,
            friction=symbol_data.causal_tension,
            incoming_causal_links=list(symbol_data.parents),
        )
        self.causal_graph[symbol_data.symbol] = node

    def trace_and_correct(self, feedback_symbol: CausalLinguisticSymbol) -> Dict[str, Any]:
        target_symbol_name = feedback_symbol.metadata.get("target_symbol")
        if not target_symbol_name or target_symbol_name not in self.causal_graph:
            return {
                "status": "FAILED",
                "reason": f"Target causal node '{target_symbol_name}' not found in narrative graph.",
            }

        causal_path = self._reverse_traverse(target_symbol_name)
        conflict_origin = None

        for symbol_name in causal_path:
            node = self.causal_graph[symbol_name]
            combined_tension = node.friction + feedback_symbol.causal_tension
            if combined_tension > self.interface.v_critical:
                conflict_origin = node
                break

        if not conflict_origin:
            conflict_origin = self.causal_graph[target_symbol_name]

        origin_name = conflict_origin.symbol_data.symbol
        if conflict_origin.friction + feedback_symbol.causal_tension > self.interface.v_critical:
            self.interface.symbolic_registry[origin_name] = SymbolState.SEALED
            self.interface.sealed_symbols.append(
                SealedAttractor(
                    attractor_id=f"SEAL_{origin_name}",
                    payload=conflict_origin.symbol_data,
                    reason="Symbolic tension overload in narrative back-trace"
                )
            )
            action_taken = "SEALED_ORIGIN_NODE"
        else:
            conflict_origin.phase_delta = 0.0
            self.interface.symbolic_registry[origin_name] = SymbolState.CONTRADICTED
            action_taken = "REALIGNED_LOCAL_PHASE"

        return {
            "status": "SUCCESS",
            "feedback": feedback_symbol.symbol,
            "resolved_origin": origin_name,
            "action": action_taken,
            "causal_path": " <- ".join(causal_path),
        }

    def _reverse_traverse(self, start_symbol: str) -> List[str]:
        path = []
        visited: Set[str] = set()
        queue = [start_symbol]

        while queue:
            curr = queue.pop(0)
            if curr in visited or curr not in self.causal_graph:
                continue
            visited.add(curr)
            path.append(curr)
            node = self.causal_graph[curr]
            queue.extend(node.incoming_causal_links)

        return path


class SymbolicAcceptanceInterface:
    """Symbolic feedback entrypoint replacing scalar backpropagation with narrative back-tracing."""
    def __init__(self, v_critical: float = 10.0):
        self.v_critical = v_critical
        self.symbolic_registry: Dict[str, SymbolState] = {}
        self.sealed_symbols: List[SealedAttractor] = []
        self.back_tracer = SymbolicBackTracer(self)

    def receive_linguistic_feedback(self, feedback_symbol: CausalLinguisticSymbol) -> Dict[str, Any]:
        return self.back_tracer.trace_and_correct(feedback_symbol)


class Layer2SymbolicCognition:
    """
    Layer 2: Symbolic & Narrative Cognition Layer.
    Processes high-order linguistic symbols ('조직의 동맥경화', '해한(解恨)', '조직의 관성'),
    maintains self-identity (I_c), and executes narrative decisions.
    """
    def __init__(self, identity_code: str = "SELF_IDENTITY_I_C"):
        self.layer_id = 2
        self.self_identity = identity_code
        self.acceptance_interface = SymbolicAcceptanceInterface()
        self.active_narratives: Dict[str, CausalLinguisticSymbol] = {}

    def reason_narrative(self, symbol: CausalLinguisticSymbol) -> Dict[str, Any]:
        self.active_narratives[symbol.symbol] = symbol
        self.acceptance_interface.symbolic_registry[symbol.symbol] = SymbolState.RESONATING
        self.acceptance_interface.back_tracer.register_trace(symbol, phase_delta=0.0)
        return {
            "self_identity": self.self_identity,
            "resonating_symbol": symbol.symbol,
            "causal_parents": symbol.parents
        }

    def process_symbolic_feedback(self, feedback_symbol: CausalLinguisticSymbol) -> Dict[str, Any]:
        return self.acceptance_interface.receive_linguistic_feedback(feedback_symbol)

    def issue_control_directive(self, target_layer: int, action_type: str, params: Dict[str, Any]) -> ControlDirective:
        if target_layer >= 2:
            raise CategoryError("Control directives from Layer 2 must target lower layers (Layer 0 or Layer 1).")
        return ControlDirective(
            target_layer=target_layer,
            action_type=action_type,
            parameters=params
        )


# =====================================================================
# Boundary Isolation & Inter-Layer Rule Enforcement
# =====================================================================

class BoundaryIsolationGuard:
    """
    Enforces inter-layer isolation rules:
    1. Upward Observability: Layer 0 -> Layer 1/2 as ObservationSignal only.
    2. No Downward Reduction: Layer 2 decisions -> ControlDirective only.
       Strictly forbids overwriting formulas or declaring "FOV check == cognitive judgment".
    3. Level-bounded Isomorphism: Isomorphisms restricted to equal abstraction levels.
    4. Information Irreversibility Protection: Forbids scalar float loss reduction.
    """
    @staticmethod
    def validate_upward_signal(signal: Any) -> ObservationSignal:
        if not isinstance(signal, ObservationSignal):
            raise CategoryError(f"Upward signals must be wrapped in ObservationSignal, got {type(signal)}.")
        return signal

    @staticmethod
    def validate_downward_directive(directive: Any) -> ControlDirective:
        if not isinstance(directive, ControlDirective):
            raise CategoryError(f"Downward directives must be wrapped in ControlDirective, got {type(directive)}.")
        return directive

    @staticmethod
    def enforce_no_downward_reduction(target_attr: str, attempt_source: str):
        if "formula" in target_attr.lower() or "cognition_is_fov" in target_attr.lower():
            raise ReductionViolationError(
                f"Attempted reductionist overwrite of '{target_attr}' from '{attempt_source}'. "
                "Higher-order cognition cannot be reduced to numerical geometry."
            )

    @staticmethod
    def check_level_isomorphism(domain_a_level: int, domain_b_level: int):
        if domain_a_level != domain_b_level:
            raise CategoryError(
                f"Isomorphism mapping violation: Layer {domain_a_level} and Layer {domain_b_level} "
                "do not share the same causal level."
            )

    @staticmethod
    def check_loss_reduction(loss_value: Any):
        if isinstance(loss_value, (float, int)):
            raise IrreversibleReductionError(
                f"Cannot reduce high-order symbolic context into scalar float Loss ({loss_value}). "
                "Use CausalLinguisticSymbol for feedback instead."
            )


# =====================================================================
# Multi-Causal Bridge Architecture (Spatial, Axiomatic, Narrative)
# =====================================================================

class DomainType(Enum):
    SPATIAL_DYNAMICS = auto()   # Video/Kinematics: optical field, spatial trajectory
    AXIOMATIC_LOGIC = auto()    # Math/Logic: axioms, derivation morphisms, consistency
    SYMBOLIC_NARRATIVE = auto() # Language: context tension, narrative invariants


@dataclass(frozen=True)
class CausalInvariant:
    """Category-theoretic invariant structure preserved across domains without scalar loss."""
    invariant_id: str
    source_domain: DomainType
    relational_graph: Dict[str, Any]
    boundary_constraints: List[str]


class DomainCausalInterface:
    def extract_causal_invariant(self) -> CausalInvariant:
        raise NotImplementedError

    def apply_boundary_constraint(self, constraint: CausalInvariant) -> None:
        raise NotImplementedError

    def reverse_trace_cause(self, target_node_id: str = "") -> List[str]:
        raise NotImplementedError


class SpatialDynamicsDomain(DomainCausalInterface):
    """1. Spatial Dynamics Domain: optical field, trajectories, spatial collisions."""
    def __init__(self):
        self.raw_events: List[Dict[str, Any]] = []

    def trigger_physical_collision(self, obj_a: str, obj_b: str, vector: str) -> CausalInvariant:
        event = {"type": "PHYSICAL_COLLISION", "entities": (obj_a, obj_b), "vector": vector}
        self.raw_events.append(event)
        return CausalInvariant(
            invariant_id="SPATIAL_COLLISION_INVARIANT",
            source_domain=DomainType.SPATIAL_DYNAMICS,
            relational_graph={
                "cause": f"KineticImpact({obj_a} -> {obj_b})",
                "trajectory_vector": vector,
            },
            boundary_constraints=[
                "DISALLOW_CONTINUOUS_MOTION_AXIOM",
                "TRIGGER_NARRATIVE_BOUNDARY_TENSION"
            ]
        )

    def extract_causal_invariant(self) -> CausalInvariant:
        return CausalInvariant(
            invariant_id="SPATIAL_TRAJECTORY_INVARIANT",
            source_domain=DomainType.SPATIAL_DYNAMICS,
            relational_graph={"event": "optical_disruption"},
            boundary_constraints=["DISALLOW_DISCONTINUOUS_STATE_JUMP"]
        )

    def apply_boundary_constraint(self, constraint: CausalInvariant) -> None:
        pass

    def reverse_trace_cause(self, target_node_id: str = "") -> List[str]:
        return [f"spatial_trajectory_origin_of_{target_node_id or 'latest_event'}"]


class AxiomaticLogicDomain(DomainCausalInterface):
    """2. Axiomatic Logic Domain: axioms, proof morphisms, logical consistency."""
    def __init__(self):
        self.active_axioms: List[str] = ["Axiom_Continuous_Motion", "Axiom_Conservation"]
        self.invalidated_lemmas: List[str] = []

    def extract_causal_invariant(self) -> CausalInvariant:
        return CausalInvariant(
            invariant_id="AXIOMATIC_PROOF_INVARIANT",
            source_domain=DomainType.AXIOMATIC_LOGIC,
            relational_graph={"premise": "Axiom_1", "lemma": "Lemma_4"},
            boundary_constraints=["REQUIRE_A_PRIORI_PROOF_CONSISTENCY"]
        )

    def apply_boundary_constraint(self, constraint: CausalInvariant) -> None:
        if "DISALLOW_CONTINUOUS_MOTION_AXIOM" in constraint.boundary_constraints:
            if "Axiom_Continuous_Motion" in self.active_axioms:
                self.active_axioms.remove("Axiom_Continuous_Motion")
            self.active_axioms.append("Axiom_Discontinuous_Impulse_Transition")
            self.invalidated_lemmas.append("Lemma_Smooth_Trajectory_Derivation")

    def reverse_trace_cause(self, target_node_id: str = "") -> List[str]:
        return [f"axiomatic_proof_chain_for_{target_node_id or 'active_axioms'}"]


class SymbolicNarrativeDomain(DomainCausalInterface):
    """3. Symbolic Narrative Domain: context tension, narrative invariants."""
    def __init__(self):
        self.symbolic_conflicts: List[Dict[str, Any]] = []

    def extract_causal_invariant(self) -> CausalInvariant:
        return CausalInvariant(
            invariant_id="NARRATIVE_TENSION_INVARIANT",
            source_domain=DomainType.SYMBOLIC_NARRATIVE,
            relational_graph={"symbol": "조직의 동맥경화", "conflict": "시스템 정체"},
            boundary_constraints=["REQUIRE_DYNAMIC_RESOLUTION_CONTEXT"]
        )

    def apply_boundary_constraint(self, constraint: CausalInvariant) -> None:
        cause = constraint.relational_graph.get("cause", "Unknown Impact")
        self.symbolic_conflicts.append({
            "archetype": "외부 충격에 의한 자아/경계막 파열",
            "narrative_context": f"공간 파열({cause}) ↔ 상징적 동맥경화 파열의 인과 공진",
            "causal_tension": 9.2,
        })

    def reverse_trace_cause(self, target_node_id: str = "") -> List[str]:
        return [f"narrative_context_origin_of_{target_node_id or 'symbolic_conflict'}"]


class MultiCausalBridge:
    """Bridge for cross-domain causal transduction and reversible backtracing without scalar loss."""
    def __init__(self):
        self.domains: Dict[DomainType, DomainCausalInterface] = {
            DomainType.SPATIAL_DYNAMICS: SpatialDynamicsDomain(),
            DomainType.AXIOMATIC_LOGIC: AxiomaticLogicDomain(),
            DomainType.SYMBOLIC_NARRATIVE: SymbolicNarrativeDomain(),
        }

    def transduce_causal_invariant(self, causal_invariant: CausalInvariant) -> Dict[str, Any]:
        source_type = causal_invariant.source_domain
        transduction_log = {}

        for d_type, target_domain in self.domains.items():
            if d_type != source_type:
                target_domain.apply_boundary_constraint(causal_invariant)
                transduction_log[d_type.name] = (
                    f"Transduced [{causal_invariant.invariant_id}] as Boundary Constraint"
                )

        return {
            "source_domain": source_type.name,
            "transduced_invariant": causal_invariant.invariant_id,
            "domain_responses": transduction_log,
        }

    def transduce_causal_state(self, source_type: DomainType) -> Dict[str, Any]:
        source_domain = self.domains[source_type]
        causal_invariant = source_domain.extract_causal_invariant()
        return self.transduce_causal_invariant(causal_invariant)

    def cross_domain_reversible_backtrace(self, node_id: str) -> Dict[str, List[str]]:
        trace_map = {}
        for d_type, domain in self.domains.items():
            trace_map[d_type.name] = domain.reverse_trace_cause(node_id)
        return trace_map


# =====================================================================
# Unified 3-Layer Engine
# =====================================================================

class ThreeLayerEngine:
    """
    Unified manager for the 3-Layer Engine Architecture.
    Coordinative control between Layer 0, Layer 1, Layer 2, Boundary Guards, and MultiCausalBridge.
    """
    def __init__(self):
        self.layer0 = Layer0GeometricState()
        self.layer1 = Layer1TopologicalDynamics()
        self.layer2 = Layer2SymbolicCognition()
        self.guard = BoundaryIsolationGuard()
        self.multi_causal_bridge = MultiCausalBridge()

    def process_upward(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        signal = self.layer0.emit_observation("SPATIAL_OBSERVATION", raw_data)
        self.guard.validate_upward_signal(signal)
        l1_res = self.layer1.process_observation(signal)
        return {
            "layer0_signal": signal,
            "layer1_response": l1_res
        }

    def process_downward_directive(self, target_layer: int, action: str, params: Dict[str, Any]) -> ControlDirective:
        directive = self.layer2.issue_control_directive(target_layer, action, params)
        self.guard.validate_downward_directive(directive)
        return directive

    def process_linguistic_feedback(self, feedback_symbol: CausalLinguisticSymbol) -> Dict[str, Any]:
        return self.layer2.process_symbolic_feedback(feedback_symbol)
