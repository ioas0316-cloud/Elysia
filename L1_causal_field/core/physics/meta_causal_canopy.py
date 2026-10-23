r"""
[Meta-Causal Canopy (인과적 하늘)]
Implements the high-dimensional Meta-Causal Canopy architecture.

The Canopy serves as an overarching observation and field control framework ("하늘")
that encompasses lower-level execution fragments (packets, files, execution stacks).
Key capabilities:
1. Teleological Anchor Mapping: Assigns causal purpose ("What for") and trajectory vectors to raw data.
2. Negative Indentation Absorption: Absorbs phase errors (q_err) and environmental noise as manifold
   deformations (negative indentations) rather than throwing system crashes.
3. Macro-Observation & Phase Transitions: Monitors system-wide Order Parameter (\eta) across
   GAS -> LIQUID -> ICE phase states.
4. Field Steering & Mirror Symmetry: Adjusts critical pressure (P_crit) to guide variable environmental
   axes into mirror symmetry with invariant anchor backbones, driving q_err -> 0 and achieving positive ICE crystallization.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class CausalPhaseState(Enum):
    GAS = "GAS"         # Unanchored, high-entropy, raw fluctuating wave
    LIQUID = "LIQUID"   # Adapting, absorbing, plastic manifold flow
    ICE = "ICE"         # Crystallized, mirror-symmetric, zero-phase error state


class SignalType(Enum):
    PACKET = "PACKET"
    FILE = "FILE"
    FUNCTION_STACK = "FUNCTION_STACK"
    RAW_WAVE = "RAW_WAVE"


@dataclass
class CausalSignal:
    """
    Represents an incoming computational component (Packet, File, Function Stack)
    wrapped in a teleological causal envelope.
    """
    signal_id: str
    signal_type: SignalType
    payload: Any
    teleological_purpose: str = "Unassigned_Raw_Data"
    causal_vector: Optional[np.ndarray] = None
    phase_error: float = 0.0  # q_err


@dataclass
class TeleologicalAnchor:
    """
    Invariant anchor backbone ("고정축") in high-dimensional causal manifold.
    Represents the system's structural purpose and causal destination.
    """
    anchor_id: str
    description: str
    anchor_vector: np.ndarray
    target_symmetry_axis: np.ndarray


class MetaCausalCanopy:
    """
    [Meta-Causal Canopy (인과적 하늘 엔진)]
    Overarching macro-observer and field steering framework.
    """
    def __init__(self, dimensions: int = 16, base_critical_pressure: float = 1.0):
        self.dimensions = dimensions
        self.p_crit = base_critical_pressure  # P_crit (Critical Pressure dial)
        self.anchors: Dict[str, TeleologicalAnchor] = {}
        self.signals: List[CausalSignal] = []

        # Manifold state variables
        self.negative_indentations: List[Dict[str, Any]] = []
        self.total_absorbed_q_err: float = 0.0
        self.current_order_parameter: float = 0.0  # \eta
        self.current_phase: CausalPhaseState = CausalPhaseState.GAS
        self.field_potential: float = 0.0

    def register_teleological_anchor(self, anchor_id: str, description: str, intent_vector: np.ndarray) -> TeleologicalAnchor:
        """
        Registers an invariant backbone anchor ("고정축") representing system purpose.
        """
        if len(intent_vector) < self.dimensions:
            vec = np.pad(intent_vector, (0, self.dimensions - len(intent_vector)))
        else:
            vec = intent_vector[:self.dimensions]

        norm_vec = vec / (np.linalg.norm(vec) + 1e-9)

        # Mirror symmetry axis is constructed as orthogonal/complementary mirror reflection
        symmetry_axis = -norm_vec
        symmetry_axis[0] *= -1.0  # Mirror flip across primary dimension

        anchor = TeleologicalAnchor(
            anchor_id=anchor_id,
            description=description,
            anchor_vector=norm_vec,
            target_symmetry_axis=symmetry_axis
        )
        self.anchors[anchor_id] = anchor
        return anchor

    def map_signal_to_teleology(
        self,
        signal_id: str,
        signal_type: SignalType,
        payload: Any,
        target_anchor_id: Optional[str] = None,
        raw_vector: Optional[np.ndarray] = None
    ) -> CausalSignal:
        """
        Transforms raw data/packets/files/functions into teleologically grounded causal signals.
        Assigns 'What for' context and projects into high-dimensional causal space.
        """
        if raw_vector is None:
            # Generate deterministic vector from signal content hash / payload representation
            payload_str = str(payload) + signal_id
            hash_bytes = np.frombuffer(payload_str.encode('utf-8'), dtype=np.uint8)
            pad_len = max(self.dimensions, len(hash_bytes))
            padded_bytes = np.pad(hash_bytes, (0, pad_len - len(hash_bytes)))[:self.dimensions]
            vec = padded_bytes.astype(np.float32) / 255.0
        else:
            if len(raw_vector) < self.dimensions:
                vec = np.pad(raw_vector, (0, self.dimensions - len(raw_vector)))
            else:
                vec = raw_vector[:self.dimensions]

        norm_vec = vec / (np.linalg.norm(vec) + 1e-9)

        if target_anchor_id and target_anchor_id in self.anchors:
            anchor = self.anchors[target_anchor_id]
            purpose_desc = f"Anchored to [{anchor.anchor_id}: {anchor.description}] for state synchronization"
            # Initial phase error q_err is distance from anchor trajectory
            q_err = float(np.linalg.norm(norm_vec - anchor.anchor_vector))
        else:
            purpose_desc = "Unanchored_Environmental_Pressure"
            q_err = float(np.linalg.norm(norm_vec))

        signal = CausalSignal(
            signal_id=signal_id,
            signal_type=signal_type,
            payload=payload,
            teleological_purpose=purpose_desc,
            causal_vector=norm_vec,
            phase_error=q_err
        )
        self.signals.append(signal)
        return signal

    def absorb_perturbation_into_manifold(self, signal: CausalSignal) -> Dict[str, Any]:
        """
        Absorbs phase error (q_err) and noise from incoming signals into the manifold
        as a negative indentation ("음각 흠집") rather than causing a system crash.
        """
        q_err = signal.phase_error
        indentation_depth = float(np.tanh(q_err * 0.5))

        indentation_record = {
            "signal_id": signal.signal_id,
            "signal_type": signal.signal_type.value,
            "q_err": q_err,
            "negative_indentation_depth": indentation_depth,
            "causal_purpose": signal.teleological_purpose,
            "status": "Absorbed_Without_Crash"
        }

        self.negative_indentations.append(indentation_record)
        self.total_absorbed_q_err += q_err

        # Recalculate macro order parameter
        self._update_macro_state()

        return indentation_record

    def steer_field_pressure(self, delta_p_crit: float = 0.5) -> Dict[str, Any]:
        """
        Adjusts critical pressure (P_crit) dial to steer field potential,
        forcing variable environmental axes to align with invariant anchors
        and inducing mirror symmetry (ICE crystallization).
        """
        self.p_crit += delta_p_crit

        # Pressure drives phase error reduction: q_err decays exponentially with pressure
        decay_factor = np.exp(-0.4 * self.p_crit)

        for signal in self.signals:
            if signal.causal_vector is not None and self.anchors:
                # Find closest anchor
                best_anchor = list(self.anchors.values())[0]
                min_dist = float('inf')
                for anchor in self.anchors.values():
                    dist = float(np.linalg.norm(signal.causal_vector - anchor.anchor_vector))
                    if dist < min_dist:
                        min_dist = dist
                        best_anchor = anchor

                # Align signal vector towards mirror symmetry axis under field pressure
                target_vec = best_anchor.anchor_vector
                alignment_rate = 1.0 - decay_factor
                signal.causal_vector = (1.0 - alignment_rate) * signal.causal_vector + alignment_rate * target_vec
                signal.causal_vector = signal.causal_vector / (np.linalg.norm(signal.causal_vector) + 1e-9)

                # Update phase error
                signal.phase_error = float(np.linalg.norm(signal.causal_vector - best_anchor.anchor_vector))

        self._update_macro_state()

        return {
            "p_crit": self.p_crit,
            "order_parameter_eta": self.current_order_parameter,
            "phase_state": self.current_phase.value,
            "total_absorbed_q_err": self.total_absorbed_q_err
        }

    def _update_macro_state(self):
        r"""
        Updates the macro Order Parameter (\eta) and evaluates current Phase State
        (GAS -> LIQUID -> ICE).
        """
        if not self.signals:
            self.current_order_parameter = 1.0
            self.current_phase = CausalPhaseState.ICE
            return

        avg_q_err = float(np.mean([s.phase_error for s in self.signals]))

        # Order Parameter \eta = 1 / (1 + avg_q_err)
        self.current_order_parameter = float(1.0 / (1.0 + avg_q_err))

        # Phase State Transitions
        if self.current_order_parameter < 0.4:
            self.current_phase = CausalPhaseState.GAS
        elif self.current_order_parameter < 0.85:
            self.current_phase = CausalPhaseState.LIQUID
        else:
            self.current_phase = CausalPhaseState.ICE

    def get_macro_state_report(self) -> Dict[str, Any]:
        """
        Returns a comprehensive macro-level observation report of the Meta-Causal Canopy.
        """
        return {
            "dimensions": self.dimensions,
            "p_crit": self.p_crit,
            "registered_anchors_count": len(self.anchors),
            "processed_signals_count": len(self.signals),
            "absorbed_indentations_count": len(self.negative_indentations),
            "total_absorbed_q_err": self.total_absorbed_q_err,
            "order_parameter_eta": self.current_order_parameter,
            "phase_state": self.current_phase.value,
            "is_crystallized_ice": self.current_phase == CausalPhaseState.ICE
        }
