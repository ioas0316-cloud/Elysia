import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Tuple, Union


class PhaseState(Enum):
    GAS = "GAS"         # Unaligned floating state (boundary pointers floating)
    LIQUID = "LIQUID"   # Active boundary friction (heat) & dynamic phase alignment search
    ICE = "ICE"         # Phase-Lock complete (friction=0, FLOPs=0 traversal state)


class DimensionType(Enum):
    SPATIAL_2D = "SPATIAL_2D"             # 2D spatial grid (e.g. PNG / image canvas)
    TEMPORAL_1D = "TEMPORAL_1D"           # 1D time-series wave (e.g. Audio / Signal stream)
    HIERARCHICAL_DAG = "HIERARCHICAL_DAG" # Hierarchical tree branches (e.g. JSON / AST)


@dataclass
class TopologyPointer:
    """
    A pointer indicating native adjacency and phase state within a modality.
    Maintains directional normal vector, intrinsic frequency, and phase offset.
    """
    node_id: str
    dimension: DimensionType
    adjacent_ids: Set[str] = field(default_factory=set)
    phase_offset: float = 0.0                     # Microscopic phase angle [0, tau)
    frequency: float = 1.0                        # Intrinsic oscillation frequency
    flux_normal: Tuple[float, ...] = (1.0, 0.0)   # Out-flux / In-flux directional normal vector


@dataclass
class NativeTopology:
    """
    A native topological structure ingested without artificial numerical transformation.
    """
    topology_id: str
    dimension: DimensionType
    pointers: Dict[str, TopologyPointer]
    boundary_nodes: Set[str]  # Nodes exposed at the Markov blanket boundary (\Phi = 0)


@dataclass
class LightSchnittResult:
    """
    Observation results extracted at the Schnitt (cut plane) of Light (cross-dimensional protocol).
    """
    invariance_bridge: Dict[str, str]                     # Invariance (Sameness): Linked pointer pairs across dimensions
    divergence_axis: Tuple[DimensionType, DimensionType] # Divergence (Difference): Unique dimensional expansion axes
    thermal_friction: float                               # Total boundary friction / heat (phase mismatch resistance)
    shear_friction: float                                 # Dimensional shear friction
    flux_friction: float                                  # Directional flux misalignment friction
    temporal_friction: float                              # Temporal phase shift and frequency mismatch friction
    phase_locked: bool                                    # Phase-Lock (ICE) transition status


class MarkovBlanketInterface:
    r"""
    Boundary Schnitt Interface preserving native topology.
    Observes structural shear, flux misalignment, and temporal phase shifts
    at the boundary (\Phi = 0) where heterogeneous dimensions meet.
    Drives spontaneous phase alignment and state transition (GAS -> LIQUID -> ICE).
    """

    def __init__(
        self,
        friction_threshold: float = 0.08,
        weights: Tuple[float, float, float] = (0.35, 0.35, 0.30)
    ):
        self.friction_threshold = friction_threshold
        self.w_shear, self.w_flux, self.w_temp = weights

    def _calc_vector_dot(self, v1: Tuple[float, ...], v2: Tuple[float, ...]) -> float:
        """
        Calculates normalized directional vector dot product.
        Aligns dimensions if vectors have differing tuple lengths.
        """
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
        Computes 3 fundamental sources of topological friction at a single pointer contact:
        1) Dimensional Shear (S_shear): Structural constraint / degree asymmetry
        2) Flux Misalignment (M_flux): Normal vector angular misalignment (180 deg / opposite facing is optimal contact)
        3) Temporal Phase Shift (P_temp): Phase angle error & frequency shift relative to topological closure tau (2*pi)
        """
        # 1. Dimensional Shear (Topological Strain)
        deg_s = len(s_ptr.adjacent_ids)
        deg_t = len(t_ptr.adjacent_ids)
        shear_friction = abs(deg_s - deg_t) / max(1, deg_s + deg_t)

        # 2. Flux Misalignment (Directional Normal Distortion)
        # Ideal contact occurs when flux vectors directly face each other (dot product = -1.0 -> friction = 0.0)
        dot_prod = self._calc_vector_dot(s_ptr.flux_normal, t_ptr.flux_normal)
        flux_friction = (1.0 + dot_prod) / 2.0  # Normalized to [0.0, 1.0]

        # 3. Temporal Phase Shift & Frequency Shift (Phase Closure Modulo tau)
        phase_diff = abs(s_ptr.phase_offset - t_ptr.phase_offset) % math.tau
        phase_err = min(phase_diff, math.tau - phase_diff) / math.pi  # Normalized to [0, 1]
        freq_err = abs(s_ptr.frequency - t_ptr.frequency) / max(0.1, s_ptr.frequency + t_ptr.frequency)
        temporal_friction = 0.5 * phase_err + 0.5 * min(1.0, freq_err)

        # Composite thermal friction
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
        r"""
        [Step 1 & 2: Schnitt Formation & Friction Measurement]
        Form Light Schnitt across Markov Blanket boundary nodes (\Phi = 0).
        Measure phase friction without matrix distortion or forced flattening.
        """
        source_boundary = [source.pointers[nid] for nid in source.boundary_nodes if nid in source.pointers]
        target_boundary = [target.pointers[nid] for nid in target.boundary_nodes if nid in target.pointers]

        if not source_boundary or not target_boundary:
            return LightSchnittResult(
                invariance_bridge={},
                divergence_axis=(source.dimension, target.dimension),
                thermal_friction=1.0,
                shear_friction=1.0,
                flux_friction=1.0,
                temporal_friction=1.0,
                phase_locked=False
            )

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
        [Step 3 & 4: Phase Transition (GAS -> LIQUID -> ICE) & FLOPs = 0 Freezing]
        Determines phase state based on Light Schnitt observation results.
        """
        schnitt_result = self.execute_light_schnitt(source, target)

        if not schnitt_result.invariance_bridge:
            # Failed pointer binding: Floating unaligned state (GAS)
            return PhaseState.GAS, {
                "friction": schnitt_result.thermal_friction,
                "reason": "No pointers satisfy friction threshold binding."
            }

        if not schnitt_result.phase_locked:
            # Active alignment in progress (LIQUID): Thermal friction (heat) still active
            return PhaseState.LIQUID, {
                "friction": schnitt_result.thermal_friction,
                "shear_friction": schnitt_result.shear_friction,
                "flux_friction": schnitt_result.flux_friction,
                "temporal_friction": schnitt_result.temporal_friction,
                "tentative_bridge": schnitt_result.invariance_bridge
            }

        # Complete Phase-Lock achieved (ICE): Thermal friction zeroed out
        # Direct pointer traversal structure with zero floating-point calculation cost (FLOPs = 0)
        unified_ice_structure = {
            "source_id": source.topology_id,
            "target_id": target.topology_id,
            "invariance_bridge": schnitt_result.invariance_bridge,  # Invariance (Sameness / Shared backbone)
            "divergence_axis": schnitt_result.divergence_axis,      # Divergence (Difference / Unique dimensions)
            "phase_state": PhaseState.ICE,
            "thermal_friction": schnitt_result.thermal_friction,
            "traversal_cost": "FLOPs = 0 (Direct Pointer Traversal)"
        }

        return PhaseState.ICE, unified_ice_structure
