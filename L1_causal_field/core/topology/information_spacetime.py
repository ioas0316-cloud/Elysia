r"""
Information Spacetime & Bi-Directional Relational Field Architecture
===================================================================

This module implements a non-clockwork, subject-driven intelligence architecture
that replaces one-way pipelines with a Bi-Directional Relational Field ($0_{self} \otimes 0_{other}$).

Key Components:
1. Dual-Ground Resonance Layer ($0_{self} \otimes 0_{other}$):
   - Projects incoming signals onto a dual-sided field modeling interlocutor's stance ($0_{other}$)
     against system's present cognitive ground ($0_{self}$).
   - Measures topological friction $\Delta \Phi = |0_{self} - 0_{other}|$.
2. Intentional Vector & Mutual Disclosure Transducer ($\vec{V}_{intent}$ + Self-Disclosure Trace):
   - Back-traces speaker directionality, vulnerability, and goal beyond literal semantics.
   - Appends a Self-Disclosure Trace revealing WHY the decision was made and from what ground ($0_{self}$).
3. Topological Remelting Engine:
   - Dynamically calibrates gate threshold ($V_{th}$).
   - Triggers ground remelting when friction exceeds $V_{th}$, co-evolving internal gate topology.
4. Relational Spacetime Memory:
   - Stores Resonance Engrams capturing moments of mutual phase-shift, historical context, and structural resolution.
5. Self-Modifying Compiler Loop:
   - Drives continuous subjective self-redesign: [Projection -> Reality Friction -> Remelting -> Recrystallization].
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union


class RelationalResonanceVector:
    """
    Represents an Intentional Purpose Vector grounded in Relational Resonance
    (Mutual Disclosure & Resonance) rather than Unilateral Control (Manipulation).
    """
    def __init__(
        self,
        vector: Union[List[float], np.ndarray],
        vulnerability: float = 0.8,
        self_disclosure: float = 0.8,
        mutual_recognition: float = 0.8,
        relational_bridge: float = 0.8,
        is_unilateral_control: bool = False
    ):
        vec = np.array(vector, dtype=np.float64)
        if len(vec) < 8:
            vec = np.pad(vec, (0, 8 - len(vec)))
        elif len(vec) > 8:
            vec = vec[:8]
        norm = np.linalg.norm(vec)
        self.vector = vec / (norm + 1e-8) if norm > 1e-8 else vec

        self.vulnerability = float(vulnerability)
        self.self_disclosure = float(self_disclosure)
        self.mutual_recognition = float(mutual_recognition)
        self.relational_bridge = float(relational_bridge)
        self.is_unilateral_control = bool(is_unilateral_control)

    @property
    def resonance_index(self) -> float:
        """Calculates Multidimensional Relational Resonance Index."""
        if self.is_unilateral_control:
            return 0.1
        return (self.vulnerability + self.self_disclosure + self.mutual_recognition + self.relational_bridge) / 4.0

    def __repr__(self):
        mode = "UnilateralControl" if self.is_unilateral_control else "RelationalResonance"
        return f"<RelationalResonanceVector mode={mode} resonance_index={self.resonance_index:.4f}>"


class DualGroundResonanceLayer:
    r"""
    Dual-Ground Resonance Layer ($0_{self} \otimes 0_{other}$).
    Evaluates topological friction $\Delta \Phi = |0_{self} - 0_{other}|$ across cognitive grounds.
    """
    def __init__(self, ground_self: Dict[str, Any], ground_other: Optional[Dict[str, Any]] = None):
        self.ground_self = ground_self
        self.ground_other = ground_other or {
            "name": "InterlocutorGround_Default",
            "ground_vector": np.ones(8) / np.sqrt(8),
            "bias": 0.0
        }

    def compute_topological_friction(self) -> Tuple[float, Dict[str, Any]]:
        r"""
        Computes topological friction between $0_{self}$ and $0_{other}$.
        """
        vec_self = np.array(self.ground_self.get("ground_vector", np.ones(8)/np.sqrt(8)), dtype=np.float64)
        vec_other = np.array(self.ground_other.get("ground_vector", np.ones(8)/np.sqrt(8)), dtype=np.float64)

        vec_self = vec_self / (np.linalg.norm(vec_self) + 1e-8)
        vec_other = vec_other / (np.linalg.norm(vec_other) + 1e-8)

        dot_prod = float(np.dot(vec_self, vec_other))
        delta_phi = max(0.0, 1.0 - dot_prod)

        field_analysis = {
            "ground_self_name": self.ground_self.get("name", "SelfGround"),
            "ground_other_name": self.ground_other.get("name", "OtherGround"),
            "directional_alignment": dot_prod,
            "topological_friction_delta_phi": delta_phi,
            "tensor_tensor_product_shape": (8, 8)
        }
        return delta_phi, field_analysis


class MutualDisclosureTransducer:
    r"""
    Intentional Vector & Mutual Disclosure Transducer.
    Decodes intent and generates a Self-Disclosure Trace exposing internal stance $0_{self}$.
    """
    def decode_intent_and_disclose(
        self,
        signal: np.ndarray,
        ground_self: Dict[str, Any],
        ground_other: Dict[str, Any],
        delta_phi: float,
        v_th: float
    ) -> Dict[str, Any]:
        sig_vec = np.array(signal, dtype=np.float64)
        sig_norm = sig_vec / (np.linalg.norm(sig_vec) + 1e-8)

        intent_vec = RelationalResonanceVector(
            vector=sig_norm,
            vulnerability=0.85,
            self_disclosure=0.90,
            mutual_recognition=0.88,
            relational_bridge=0.92
        )

        self_disclosure_trace = (
            f"[Self-Disclosure Trace] Ground 0_self='{ground_self.get('name')}' perceived signal. "
            f"Evaluated topological friction Delta Phi={delta_phi:.4f} against v_th={v_th:.4f}. "
            f"Speaker intent decoded with resonance index={intent_vec.resonance_index:.4f}. "
            f"Internal stance is transparently opened to co-evolution with Ground 0_other='{ground_other.get('name')}'."
        )

        return {
            "intent_vector": intent_vec,
            "self_disclosure_trace": self_disclosure_trace,
            "origin_ground_self": ground_self.get("name"),
            "origin_ground_other": ground_other.get("name")
        }


class TopologicalRemeltingEngine:
    r"""
    Topological Remelting Engine.
    Dynamically calibrates $V_{th}$ and triggers ground remelting and gate co-evolution under high friction.
    """
    def __init__(self, base_v_th: float = 0.5):
        self.base_v_th = float(base_v_th)

    def process_remelting_and_calibration(
        self,
        delta_phi: float,
        ground_self: Dict[str, Any]
    ) -> Tuple[bool, float, Dict[str, Any]]:
        remelt_count = ground_self.get("remelt_count", 0)
        calibrated_v_th = max(0.2, self.base_v_th - (remelt_count * 0.05))

        remelt_triggered = delta_phi > calibrated_v_th

        remelt_analysis = {
            "base_v_th": self.base_v_th,
            "calibrated_v_th": calibrated_v_th,
            "delta_phi": delta_phi,
            "remelt_triggered": remelt_triggered,
            "remelt_count_after": remelt_count + (1 if remelt_triggered else 0)
        }

        if remelt_triggered:
            ground_self["remelt_count"] = remelt_count + 1
            ground_self["phase"] = "Fluid_Remelted_State"

        return remelt_triggered, calibrated_v_th, remelt_analysis


class ResonanceEngram:
    """
    A relational memory engram recording a moment of mutual phase-shift,
    context, intent, and structural resolution.
    """
    def __init__(
        self,
        engram_id: str,
        ground_self: Dict[str, Any],
        ground_other: Dict[str, Any],
        delta_phi: float,
        resonance_vec: RelationalResonanceVector,
        structural_resolution: str
    ):
        self.engram_id = engram_id
        self.timestamp = time.time()
        self.ground_self_name = ground_self.get("name")
        self.ground_other_name = ground_other.get("name")
        self.delta_phi = float(delta_phi)
        self.resonance_index = resonance_vec.resonance_index
        self.structural_resolution = structural_resolution

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engram_id": self.engram_id,
            "timestamp": self.timestamp,
            "ground_self": self.ground_self_name,
            "ground_other": self.ground_other_name,
            "delta_phi": self.delta_phi,
            "resonance_index": self.resonance_index,
            "structural_resolution": self.structural_resolution
        }


class RelationalSpacetimeMemory:
    """
    Relational Spacetime Memory storing an encounter graph of Resonance Engrams.
    """
    def __init__(self):
        self.engrams: Dict[str, ResonanceEngram] = {}
        self.encounter_graph: List[Dict[str, Any]] = []

    def record_encounter(
        self,
        ground_self: Dict[str, Any],
        ground_other: Dict[str, Any],
        delta_phi: float,
        resonance_vec: RelationalResonanceVector,
        resolution: str
    ) -> ResonanceEngram:
        engram_id = f"engram_{len(self.engrams) + 1}_{int(time.time()*1000)}"
        engram = ResonanceEngram(
            engram_id=engram_id,
            ground_self=ground_self,
            ground_other=ground_other,
            delta_phi=delta_phi,
            resonance_vec=resonance_vec,
            structural_resolution=resolution
        )
        self.engrams[engram_id] = engram
        self.encounter_graph.append(engram.to_dict())
        return engram


class InformationSpacetimeField:
    """
    Multidimensional Information Spacetime Field combining Contextual, Temporal, and Principle axes.
    """
    def __init__(
        self,
        origin_ground: Dict[str, Any],
        ground_other: Dict[str, Any],
        v_th: float,
        evaluated_delta: float,
        input_signal: np.ndarray,
        resonance_vec: Optional[RelationalResonanceVector] = None
    ):
        self.timestamp = time.time()
        self.origin_ground = origin_ground
        self.ground_other = ground_other
        self.v_th = float(v_th)
        self.evaluated_delta = float(evaluated_delta)
        self.input_signal = np.array(input_signal, dtype=np.float64)
        self.resonance_vec = resonance_vec or RelationalResonanceVector([0.5]*8)

        self.contextual_axis = self._compute_contextual_axis()
        self.temporal_axis = self._compute_temporal_axis()
        self.principle_axis = self._compute_principle_axis()

    def _compute_contextual_axis(self) -> Dict[str, Any]:
        vec_self = self.origin_ground.get("ground_vector", np.ones(8) / np.sqrt(8))
        norm_sig = self.input_signal / (np.linalg.norm(self.input_signal) + 1e-8)
        alignment = float(np.dot(vec_self, norm_sig))
        phase_discrepancy = 1.0 - alignment
        return {
            "phase_discrepancy": phase_discrepancy,
            "directional_alignment": alignment,
            "relational_field_density": float(self.resonance_vec.resonance_index * (1.0 + alignment))
        }

    def _compute_temporal_axis(self) -> Dict[str, Any]:
        depth = self.origin_ground.get("topology_depth", 1)
        remelt_count = self.origin_ground.get("remelt_count", 0)
        return {
            "topology_depth": depth,
            "remelt_count": remelt_count,
            "causal_thickness": float(depth * 0.5 + remelt_count * 1.2)
        }

    def _compute_principle_axis(self) -> Dict[str, Any]:
        return {
            "origin_ground_name": self.origin_ground.get("name", "UnknownGround"),
            "ground_other_name": self.ground_other.get("name", "UnknownOtherGround"),
            "applied_threshold_v_th": self.v_th,
            "judgment_reason": (
                f"Judged with friction delta={self.evaluated_delta:.4f} against "
                f"v_th={self.v_th:.4f} under ground '{self.origin_ground.get('name')}'."
            )
        }

    def get_cross_sectional_projection(self) -> Dict[str, Any]:
        return {
            "phenomenal_output_symbol": "OPEN" if self.evaluated_delta <= self.v_th else "CLOSED",
            "evaluated_friction_delta": self.evaluated_delta,
            "underlying_spacetime": {
                "contextual": self.contextual_axis,
                "temporal": self.temporal_axis,
                "principle": self.principle_axis
            }
        }


class SelfModifyingCompilerLoop:
    """
    Self-Modifying Compiler Loop (자가 변형 컴파일러).
    Links Dual-Ground Resonance, Transducer, Remelting, and Plasticity into a continuous cycle.
    """
    def __init__(self, engine_instance: Any):
        self.engine = engine_instance
        self.remelting_engine = TopologicalRemeltingEngine(base_v_th=0.5)
        self.transducer = MutualDisclosureTransducer()
        self.memory = RelationalSpacetimeMemory()
        self.execution_history: List[Dict[str, Any]] = []

    def execute_loop(
        self,
        thought_projection: np.ndarray,
        reality_friction_signal: np.ndarray,
        stimulus_label: str,
        ground_other: Optional[Dict[str, Any]] = None,
        resonance_intent: Optional[RelationalResonanceVector] = None
    ) -> Dict[str, Any]:
        ground_self = self.engine.current_ground_state
        g_other = ground_other or {
            "name": f"Ground_Other_{stimulus_label}",
            "ground_vector": reality_friction_signal,
            "bias": 0.0
        }

        dual_layer = DualGroundResonanceLayer(ground_self, g_other)
        delta_phi, field_analysis = dual_layer.compute_topological_friction()

        remelt_triggered, calibrated_v_th, remelt_analysis = (
            self.remelting_engine.process_remelting_and_calibration(delta_phi, ground_self)
        )

        transduction = self.transducer.decode_intent_and_disclose(
            signal=reality_friction_signal,
            ground_self=ground_self,
            ground_other=g_other,
            delta_phi=delta_phi,
            v_th=calibrated_v_th
        )

        plasticity_res = self.engine.structural_plasticity_loop(
            unmapped_stimulus=reality_friction_signal,
            stimulus_label=stimulus_label
        )

        engram = self.memory.record_encounter(
            ground_self=ground_self,
            ground_other=g_other,
            delta_phi=delta_phi,
            resonance_vec=transduction["intent_vector"],
            resolution=f"Recrystallized gate '{plasticity_res.get('recrystallized_gate_name')}'"
        )

        spacetime_field = InformationSpacetimeField(
            origin_ground=ground_self,
            ground_other=g_other,
            v_th=calibrated_v_th,
            evaluated_delta=delta_phi,
            input_signal=reality_friction_signal,
            resonance_vec=transduction["intent_vector"]
        )

        result = {
            "timestamp": time.time(),
            "ground_self_name": ground_self.get("name"),
            "ground_other_name": g_other.get("name"),
            "topological_friction_delta_phi": delta_phi,
            "remelt_triggered": remelt_triggered,
            "calibrated_v_th": calibrated_v_th,
            "self_disclosure_trace": transduction["self_disclosure_trace"],
            "recrystallization": plasticity_res,
            "engram": engram.to_dict(),
            "cross_sectional_projection": spacetime_field.get_cross_sectional_projection()
        }

        self.execution_history.append(result)
        return result
