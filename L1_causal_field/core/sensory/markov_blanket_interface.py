import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Tuple, Union


class PhaseState(Enum):
    GAS = "GAS"         # Unbound / floating topological pointers
    LIQUID = "LIQUID"   # Active topological alignment & friction dissipation
    ICE = "ICE"         # Phase-locked frozen topology (FLOPs = 0 traversal)


class DimensionType(Enum):
    SPATIAL_2D = "SPATIAL_2D"             # 2D Spatial Grid Topology (Pixel/Grid Neighbor)
    TEMPORAL_1D = "TEMPORAL_1D"           # 1D Temporal Wave Topology (Continuous Oscillation)
    HIERARCHICAL_DAG = "HIERARCHICAL_DAG" # Hierarchical Tree Topology (Branch/Depth Pointers)


@dataclass
class TopologyPointer:
    """
    Pointer pointing to native adjacency and phase state within a modality.
    """
    node_id: str
    dimension: DimensionType
    adjacent_ids: Set[str] = field(default_factory=set)
    phase_offset: float = 0.0                  # Phase angle [0, tau)
    frequency: float = 1.0                     # Native frequency/oscillation rate
    flux_normal: Tuple[float, ...] = (1.0, 0.0) # In-flux/Out-flux directional vector


@dataclass
class NativeTopology:
    """
    Native topological structure accepted without external parser/floating-point conversion.
    """
    topology_id: str
    dimension: DimensionType
    pointers: Dict[str, TopologyPointer]
    boundary_nodes: Set[str]  # Nodes exposed at the Markov Blanket interface (Phi = 0)


@dataclass
class LightSchnittResult:
    """
    Observed metrics extracted at the Light (Multi-dimensional Protocol) Schnitt interface.
    """
    invariance_bridge: Dict[str, str]  # Common structural binding (Invariance)
    divergence_axis: Tuple[DimensionType, DimensionType] # Modality divergence axis
    thermal_friction: float            # Total boundary thermal friction
    shear_friction: float              # Dimensional shear friction
    flux_friction: float               # Directional flux misalignment friction
    temporal_friction: float           # Temporal phase/frequency shift friction
    phase_locked: bool                 # True if Phase-Lock (ICE state) is achieved


class MarkovBlanketInterface:
    """
    Markov Blanket interface preserving native topology.
    Observes 3 root causes of topological mismatch (Dimensional Shear, Flux Misalignment, Temporal Shift)
    and drives self-organized phase transition (GAS -> LIQUID -> ICE).
    """

    def __init__(
        self,
        friction_threshold: float = 0.08,
        weights: Tuple[float, float, float] = (0.35, 0.35, 0.30)
    ):
        self.friction_threshold = friction_threshold
        self.w_shear, self.w_flux, self.w_temp = weights

    def _calc_vector_dot(self, v1: Tuple[float, ...], v2: Tuple[float, ...]) -> float:
        """Calculates normalized vector dot product across dimension lengths."""
        min_dim = min(len(v1), len(v2))
        if min_dim == 0:
            return 0.0
        dot = sum(v1[i] * v2[i] for i in range(min_dim))
        norm1 = math.sqrt(sum(v1[i] ** 2 for i in range(len(v1)))) or 1.0
        norm2 = math.sqrt(sum(v2[i] ** 2 for i in range(len(v2)))) or 1.0
        return dot / (norm1 * norm2)

    def compute_point_friction(
        self,
        s_ptr: TopologyPointer,
        t_ptr: TopologyPointer
    ) -> Tuple[float, float, float, float]:
        """
        Computes the 3 topological friction components at a single boundary contact point:
        1. Dimensional Shear: Structural constraint asymmetry
        2. Flux Misalignment: Normal vector directional misalignment
        3. Temporal Phase Shift: Phase angle and frequency mismatch
        """
        # 1. Dimensional Shear (Structural constraint degree difference)
        deg_s = len(s_ptr.adjacent_ids)
        deg_t = len(t_ptr.adjacent_ids)
        shear_friction = abs(deg_s - deg_t) / max(1, deg_s + deg_t)

        # 2. Flux Misalignment (Facing opposing vectors = 0 friction, aligned = max friction)
        dot_prod = self._calc_vector_dot(s_ptr.flux_normal, t_ptr.flux_normal)
        flux_friction = (1.0 + dot_prod) / 2.0

        # 3. Temporal Phase Shift (Phase angle & frequency shift)
        phase_diff = abs(s_ptr.phase_offset - t_ptr.phase_offset) % math.tau
        phase_err = min(phase_diff, math.tau - phase_diff) / math.pi
        freq_err = abs(s_ptr.frequency - t_ptr.frequency) / max(0.1, s_ptr.frequency + t_ptr.frequency)
        temporal_friction = 0.5 * phase_err + 0.5 * min(1.0, freq_err)

        # Total weighted thermal friction
        total_friction = (
            self.w_shear * shear_friction +
            self.w_flux * flux_friction +
            self.w_temp * temporal_friction
        )

        return total_friction, shear_friction, flux_friction, temporal_friction

    def execute_light_schnitt(
        self,
        source: NativeTopology,
        target: NativeTopology
    ) -> LightSchnittResult:
        """
        Forms the Light Schnitt (Null Hypersurface) across boundary nodes and measures friction.
        """
        source_boundary = [source.pointers[nid] for nid in source.boundary_nodes if nid in source.pointers]
        target_boundary = [target.pointers[nid] for nid in target.boundary_nodes if nid in target.pointers]

        invariance_bridge: Dict[str, str] = {}
        total_f, total_s, total_x, total_t = 0.0, 0.0, 0.0, 0.0
        count = 0

        for s_ptr in source_boundary:
            best_target_id = None
            min_ptr_friction = float('inf')

            for t_ptr in target_boundary:
                tf, sf, ff, temp_f = self.compute_point_friction(s_ptr, t_ptr)
                total_f += tf
                total_s += sf
                total_x += ff
                total_t += temp_f
                count += 1

                if tf < min_ptr_friction:
                    min_ptr_friction = tf
                    best_target_id = t_ptr.node_id

            if best_target_id and min_ptr_friction < self.friction_threshold:
                invariance_bridge[s_ptr.node_id] = best_target_id

        denom = max(1, count)
        avg_f = total_f / denom
        is_phase_locked = (avg_f < self.friction_threshold) and (len(invariance_bridge) > 0)

        return LightSchnittResult(
            invariance_bridge=invariance_bridge,
            divergence_axis=(source.dimension, target.dimension),
            thermal_friction=avg_f,
            shear_friction=total_s / denom,
            flux_friction=total_x / denom,
            temporal_friction=total_t / denom,
            phase_locked=is_phase_locked
        )

    def phase_transition_step(
        self,
        source: NativeTopology,
        target: NativeTopology
    ) -> Tuple[PhaseState, Optional[Dict[str, Any]]]:
        """
        Executes phase transition (GAS -> LIQUID -> ICE) based on boundary Schnitt result.
        """
        schnitt_result = self.execute_light_schnitt(source, target)

        if not schnitt_result.invariance_bridge:
            return PhaseState.GAS, {
                "friction": schnitt_result.thermal_friction,
                "status": "Unbound boundary pointers (GAS)"
            }

        if not schnitt_result.phase_locked:
            return PhaseState.LIQUID, {
                "friction": schnitt_result.thermal_friction,
                "shear_friction": schnitt_result.shear_friction,
                "flux_friction": schnitt_result.flux_friction,
                "temporal_friction": schnitt_result.temporal_friction,
                "tentative_bridge": schnitt_result.invariance_bridge,
                "status": "Active topological alignment & friction dissipation"
            }

        unified_ice_structure = {
            "source_id": source.topology_id,
            "target_id": target.topology_id,
            "invariance_bridge": schnitt_result.invariance_bridge,
            "divergence_axis": schnitt_result.divergence_axis,
            "phase_state": PhaseState.ICE,
            "traversal_cost": "FLOPs = 0 (Direct Graph Pointer Traversal)",
            "status": "Phase-Locked Frozen Topology"
        }

        return PhaseState.ICE, unified_ice_structure
