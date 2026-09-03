"""
Fractal Gate Language Engine (프랙탈 게이트 언어 엔진)
===================================================

This module implements the fundamental principle that language is a fractal superposition
of logic gate switching mechanisms:
1. Hardware Gate Level: Signal Input -> Threshold Voltage (V_th) Comparison -> Channel Switching -> Output.
2. 3-Stage Linguistic Hierarchy:
   - Word (Primitive Gate): Minimum switching boundary with threshold V_th.
   - Sentence (Combinational Circuit): Cascade of primitive gates steering cognitive flow.
   - Discourse (Dynamic State Machine): High-order topological circuit transforming initial Ground (0_initial)
     into a newly recrystallized final Ground (0_final).
3. Qualitative Phase Shift & Emergent Ground Creation:
   - Non-scalar qualitative state transitions (e.g. Couple + Marriage Gate -> Family Emergent Ground).
4. Multimodal Invariant Ground & Intentional Vector:
   - Medium-invariant convergence of surface signals (Text, Speech, Visual) into a single Invariant Causal Core.
   - Intent Vector / Teleological Back-trace: Detecting speaker's intent directionality and remelting state.
5. Meta-Information Engraving (MetaInformationPacket):
   - Encodes [Ground A, Stimulus B/C, Delta D, Output, Causal Trace].
   - Causal Trace is re-injectable as subsequent Ground (0_ground').
6. 3-Step Structural Plasticity Loop:
   - (1) Unmapped Friction Detection
   - (2) Primitive Fractal Projection
   - (3) Self-Recrystallization into Persistent Causal Graph.
7. Bi-Directional Relational Field Integration (0_self x 0_other):
   - Integrated with `information_spacetime.py` for Dual-Ground Resonance, Mutual Disclosure, Remelting, and Spacetime Memory.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Set

from core.topology.information_spacetime import (
    DualGroundResonanceLayer,
    MutualDisclosureTransducer,
    TopologicalRemeltingEngine,
    RelationalSpacetimeMemory,
    RelationalResonanceVector,
    InformationSpacetimeField,
    SelfModifyingCompilerLoop
)


class MetaInformationPacket:
    """
    Structured meta-information packet engraved inside the system.
    Captures the full mechanism: [Ground A -> Comparison/Friction D -> Output]
    and makes the Causal Trace executable/re-injectable as a future Ground.
    """
    def __init__(
        self,
        ground_a: Dict[str, Any],
        stimulus_b: Dict[str, Any],
        stimulus_c: Optional[Dict[str, Any]],
        delta_d: float,
        v_th: float,
        channel_open: bool,
        output_state: Dict[str, Any],
        causal_trace: Dict[str, Any],
        explanation: str
    ):
        self.timestamp = time.time()
        self.ground_a = ground_a
        self.stimulus_b = stimulus_b
        self.stimulus_c = stimulus_c
        self.delta_d = float(delta_d)
        self.v_th = float(v_th)
        self.channel_open = bool(channel_open)
        self.output_state = output_state
        self.causal_trace = causal_trace
        self.explanation = explanation

    def to_ground(self) -> Dict[str, Any]:
        """
        Re-injects the causal trace and output state as a new Ground (0_ground')
        for subsequent recursive reasoning.
        """
        prev_depth = self.ground_a.get("topology_depth", 1)
        prev_remelt = self.ground_a.get("remelt_count", 0)
        return {
            "name": f"Ground_{self.ground_a.get('name', 'Ground')}_Depth{prev_depth + 1}",
            "origin_ground": self.ground_a.get("name", "UnknownGround"),
            "evaluated_delta": self.delta_d,
            "v_th_applied": self.v_th,
            "output": self.output_state,
            "causal_trace": self.causal_trace,
            "timestamp": self.timestamp,
            "active_axes": list(self.output_state.keys()) if isinstance(self.output_state, dict) else [],
            "bias": self.ground_a.get("bias", 0.0),
            "topology_depth": prev_depth + 1,
            "remelt_count": prev_remelt,
            "ground_vector": self.ground_a.get("ground_vector", np.ones(8) / np.sqrt(8))
        }

    def __repr__(self):
        return f"<MetaInformationPacket Delta={self.delta_d:.4f} Open={self.channel_open} Explanation='{self.explanation[:40]}...'>"


class PrimitiveGate:
    """
    Word = Primitive Gate (원초적 판별 게이트)
    Modeled after transistor switching logic:
    Input Signal vs Threshold Voltage (V_th) -> Friction Delta -> Channel Switching.
    """
    def __init__(
        self,
        gate_id: str,
        name: str,
        v_th: float = 0.5,
        reference_vector: Optional[np.ndarray] = None,
        domain: str = "general"
    ):
        self.gate_id = gate_id
        self.name = name
        self.v_th = float(v_th)
        if reference_vector is None:
            self.reference_vector = np.random.randn(8)
            self.reference_vector /= np.linalg.norm(self.reference_vector) + 1e-8
        else:
            self.reference_vector = np.array(reference_vector, dtype=np.float64)
            norm = np.linalg.norm(self.reference_vector)
            if norm > 1e-8:
                self.reference_vector /= norm
        self.domain = domain
        self.conduction_count = 0

    def evaluate(
        self,
        input_signal: np.ndarray,
        ground_bias: float = 0.0
    ) -> Tuple[bool, float, float]:
        signal_vec = np.array(input_signal, dtype=np.float64)
        norm = np.linalg.norm(signal_vec)
        if norm > 1e-8:
            signal_vec /= norm

        alignment = np.dot(self.reference_vector, signal_vec)
        delta_d = float(1.0 - alignment)

        effective_v_th = max(0.01, self.v_th + ground_bias)
        channel_open = delta_d <= effective_v_th

        if channel_open:
            self.conduction_count += 1

        return channel_open, delta_d, effective_v_th


class CombinationalCircuit:
    """
    Sentence = Combinational Circuit (조합 논리 회로)
    Directs signal flow through series/parallel gates to produce a structural phase transition.
    """
    def __init__(self, circuit_id: str, gates: List[PrimitiveGate], connection_topology: str = "series"):
        self.circuit_id = circuit_id
        self.gates = gates
        self.connection_topology = connection_topology

    def process_signal(
        self,
        signal: np.ndarray,
        ground_context: Dict[str, Any]
    ) -> Tuple[bool, float, List[Dict[str, Any]]]:
        ground_bias = float(ground_context.get("bias", 0.0))
        evaluations = []
        overall_open = True if self.connection_topology == "series" else False
        total_delta = 0.0

        for gate in self.gates:
            is_open, delta, eff_v_th = gate.evaluate(signal, ground_bias=ground_bias)
            evaluations.append({
                "gate_id": gate.gate_id,
                "gate_name": gate.name,
                "is_open": is_open,
                "delta": delta,
                "eff_v_th": eff_v_th
            })
            total_delta += delta

            if self.connection_topology == "series":
                overall_open = overall_open and is_open
            else:
                overall_open = overall_open or is_open

        avg_delta = total_delta / max(1, len(self.gates))
        return overall_open, avg_delta, evaluations


class PersistentCausalGraph:
    """
    Persistent Causal Graph representing the system's structural ground topology.
    Dynamic nodes and edges can be added/crystallized at runtime without modifying code.
    """
    def __init__(self):
        self.nodes: Dict[str, PrimitiveGate] = {}
        self.edges: Dict[str, List[str]] = {}
        self.crystallized_axes: Dict[str, np.ndarray] = {}

    def add_gate(self, gate: PrimitiveGate):
        self.nodes[gate.gate_id] = gate
        if gate.gate_id not in self.edges:
            self.edges[gate.gate_id] = []
        self.crystallized_axes[gate.name] = gate.reference_vector

    def add_edge(self, source_id: str, target_id: str):
        if source_id in self.nodes and target_id in self.nodes:
            if target_id not in self.edges[source_id]:
                self.edges[source_id].append(target_id)


class FractalGateLanguageEngine:
    """
    Master Engine unifying Hardware-Gate Switching, 3-Stage Linguistic Hierarchy,
    Qualitative Phase Shift, Multimodal Invariant Grounding, Intent Vector Back-tracing,
    3-Step Structural Plasticity Loop, and Bi-Directional Relational Spacetime Field.
    """
    def __init__(self, ground_name: str = "GroundZero"):
        self.ground_name = ground_name
        self.causal_graph = PersistentCausalGraph()
        self.current_ground_state: Dict[str, Any] = {
            "name": ground_name,
            "bias": 0.0,
            "phase": "Initial_Stable",
            "active_axes": [],
            "topology_depth": 1,
            "remelt_count": 0,
            "ground_vector": np.ones(8) / np.sqrt(8)
        }
        self.engraved_packets: List[MetaInformationPacket] = []
        self.default_v_th = 0.5

        # Initialize Self-Modifying Compiler Loop and Relational Memory
        self.compiler_loop = SelfModifyingCompilerLoop(self)
        self.memory = RelationalSpacetimeMemory()

        self._bootstrap_primitive_gates()

    def _bootstrap_primitive_gates(self):
        default_gates = [
            PrimitiveGate("gate_apple", "Apple_Gate", v_th=0.6, reference_vector=[1, 0, 0, 0, 0, 0, 0, 0]),
            PrimitiveGate("gate_marriage", "Marriage_Gate", v_th=0.5, reference_vector=[0, 1, 1, 0, 0, 0, 0, 0]),
            PrimitiveGate("gate_family", "Family_Gate", v_th=0.4, reference_vector=[0, 1, 1, 1, 0, 0, 0, 0]),
            PrimitiveGate("gate_intent", "Intent_Gate", v_th=0.5, reference_vector=[0, 0, 0, 1, 1, 0, 0, 0]),
        ]
        for g in default_gates:
            self.causal_graph.add_gate(g)
        self.causal_graph.add_edge("gate_marriage", "gate_family")

    def process_multimodal_signal(
        self,
        surface_signal: Union[np.ndarray, List[float]],
        medium_type: str = "text",
        intent_vector: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        raw_vec = np.array(surface_signal, dtype=np.float64)
        if len(raw_vec) < 8:
            raw_vec = np.pad(raw_vec, (0, 8 - len(raw_vec)))
        elif len(raw_vec) > 8:
            raw_vec = raw_vec[:8]

        norm = np.linalg.norm(raw_vec)
        invariant_core = raw_vec / (norm + 1e-8)

        teleological_friction = 0.0
        intent_alignment = 0.0
        if intent_vector is not None:
            intent_vec = np.array(intent_vector, dtype=np.float64)
            if len(intent_vec) < 8:
                intent_vec = np.pad(intent_vec, (0, 8 - len(intent_vec)))
            intent_vec = intent_vec[:8]
            i_norm = np.linalg.norm(intent_vec)
            if i_norm > 1e-8:
                intent_vec /= i_norm
            intent_alignment = float(np.dot(invariant_core, intent_vec))
            teleological_friction = float(1.0 - intent_alignment)

        return {
            "medium_type": medium_type,
            "invariant_core": invariant_core,
            "intent_alignment": intent_alignment,
            "teleological_friction": teleological_friction
        }

    def execute_qualitative_phase_shift(
        self,
        entity_a: Dict[str, Any],
        entity_b: Dict[str, Any],
        catalyst_gate_id: str = "gate_marriage"
    ) -> Dict[str, Any]:
        catalyst_gate = self.causal_graph.nodes.get(catalyst_gate_id)
        if not catalyst_gate:
            raise ValueError(f"Catalyst gate '{catalyst_gate_id}' not found in causal graph.")

        vec_a = np.array(entity_a.get("vector", [0.5, 0.5, 0, 0, 0, 0, 0, 0]), dtype=np.float64)
        vec_b = np.array(entity_b.get("vector", [0.5, 0.5, 0, 0, 0, 0, 0, 0]), dtype=np.float64)

        coupled_signal = (vec_a + vec_b) / 2.0
        channel_open, delta, eff_v_th = catalyst_gate.evaluate(coupled_signal)

        if channel_open:
            emergent_ground_name = f"EmergentGround_{entity_a.get('name','A')}_{entity_b.get('name','B')}_Family"
            emergent_vector = coupled_signal + catalyst_gate.reference_vector
            emergent_vector /= (np.linalg.norm(emergent_vector) + 1e-8)

            emergent_ground = {
                "name": emergent_ground_name,
                "type": "Emergent_Higher_Order_Manifold",
                "components": [entity_a.get("name"), entity_b.get("name")],
                "emergent_vector": emergent_vector,
                "qualitative_state": "Family_Bound_State",
                "phase_shift_occurred": True,
                "friction_consumed": delta
            }

            packet = MetaInformationPacket(
                ground_a=self.current_ground_state,
                stimulus_b=entity_a,
                stimulus_c=entity_b,
                delta_d=delta,
                v_th=eff_v_th,
                channel_open=True,
                output_state=emergent_ground,
                causal_trace={
                    "catalyst": catalyst_gate.name,
                    "mechanism": "Qualitative Phase Shift: Individual Boundaries Dissolved -> Family Manifold Created"
                },
                explanation=f"Entity '{entity_a.get('name')}' and '{entity_b.get('name')}' passed through '{catalyst_gate.name}' (Friction {delta:.4f} <= V_th {eff_v_th:.4f}), triggering qualitative phase transition into Emergent Family Ground."
            )
            self.engraved_packets.append(packet)

            self.current_ground_state = packet.to_ground()
            self.current_ground_state["ground_vector"] = emergent_vector
            return emergent_ground
        else:
            return {
                "phase_shift_occurred": False,
                "reason": f"Friction {delta:.4f} exceeded threshold V_th {eff_v_th:.4f}. Channel remained closed."
            }

    def process_discourse(
        self,
        sentences: List[CombinationalCircuit],
        signal_stream: List[np.ndarray],
        intent_vectors: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        ground_history = [self.current_ground_state.copy()]
        trace_log = []

        for idx, circuit in enumerate(sentences):
            sig = signal_stream[idx] if idx < len(signal_stream) else np.random.randn(8)
            i_vec = intent_vectors[idx] if (intent_vectors and idx < len(intent_vectors)) else None

            grounded_info = self.process_multimodal_signal(sig, intent_vector=i_vec)

            circuit_open, avg_delta, gate_evals = circuit.process_signal(
                grounded_info["invariant_core"],
                self.current_ground_state
            )

            explanation_str = (
                f"Sentence Circuit '{circuit.circuit_id}' evaluated. "
                f"Circuit Open={circuit_open}, Avg Delta={avg_delta:.4f}, "
                f"Intent Alignment={grounded_info['intent_alignment']:.4f}."
            )
            packet = MetaInformationPacket(
                ground_a=self.current_ground_state,
                stimulus_b={"signal_idx": idx, "medium": grounded_info["medium_type"]},
                stimulus_c=None,
                delta_d=avg_delta,
                v_th=0.5,
                channel_open=circuit_open,
                output_state={
                    "circuit_open": circuit_open,
                    "active_gates": [e["gate_name"] for e in gate_evals if e["is_open"]]
                },
                causal_trace={"circuit_id": circuit.circuit_id, "gate_evaluations": gate_evals},
                explanation=explanation_str
            )
            self.engraved_packets.append(packet)

            self.current_ground_state = packet.to_ground()
            ground_history.append(self.current_ground_state.copy())
            trace_log.append(explanation_str)

        return {
            "initial_ground": ground_history[0]["name"],
            "final_ground": self.current_ground_state["origin_ground"],
            "steps_processed": len(sentences),
            "trace_log": trace_log,
            "packets_engraved": len(self.engraved_packets)
        }

    def structural_plasticity_loop(
        self,
        unmapped_stimulus: np.ndarray,
        stimulus_label: str = "Unmapped_Novel_Concept"
    ) -> Dict[str, Any]:
        stim_vec = np.array(unmapped_stimulus, dtype=np.float64)
        norm = np.linalg.norm(stim_vec)
        if norm > 1e-8:
            stim_vec /= norm

        min_delta = float("inf")
        closest_gate = None
        for gate_id, gate in self.causal_graph.nodes.items():
            _, delta, eff_v = gate.evaluate(stim_vec)
            if delta < min_delta:
                min_delta = delta
                closest_gate = gate

        unmapped_detected = min_delta > 0.55

        if not unmapped_detected:
            return {
                "unmapped_detected": False,
                "recognized_by_gate": closest_gate.name if closest_gate else None,
                "friction_delta": min_delta
            }

        new_gate_id = f"gate_recrystallized_{len(self.causal_graph.nodes) + 1}"
        new_gate_name = f"Gate_{stimulus_label}"
        projected_v_th = max(0.3, min_delta * 0.8)

        new_gate = PrimitiveGate(
            gate_id=new_gate_id,
            name=new_gate_name,
            v_th=projected_v_th,
            reference_vector=stim_vec,
            domain="crystallized_self_extension"
        )

        self.causal_graph.add_gate(new_gate)
        if closest_gate:
            self.causal_graph.add_edge(closest_gate.gate_id, new_gate_id)

        is_open_now, delta_after, _ = new_gate.evaluate(stim_vec)

        packet = MetaInformationPacket(
            ground_a=self.current_ground_state,
            stimulus_b={"label": stimulus_label, "raw_signal": unmapped_stimulus.tolist()},
            stimulus_c=None,
            delta_d=min_delta,
            v_th=projected_v_th,
            channel_open=is_open_now,
            output_state={
                "recrystallized_gate_id": new_gate_id,
                "recrystallized_gate_name": new_gate_name,
                "delta_after_recrystallization": delta_after
            },
            causal_trace={
                "loop_step_1": f"Unmapped Friction Detected (Min Delta {min_delta:.4f} > V_th threshold)",
                "loop_step_2": f"Primitive Fractal Projected with Adaptive V_th {projected_v_th:.4f}",
                "loop_step_3": f"Self-Recrystallized Gate '{new_gate_name}' added to Persistent Causal Graph."
            },
            explanation=f"Engine detected unmapped friction ({min_delta:.4f}) for '{stimulus_label}'. Projected primitive switching gate template and recrystallized '{new_gate_name}' into persistent topology."
        )
        self.engraved_packets.append(packet)
        self.current_ground_state = packet.to_ground()

        return {
            "unmapped_detected": True,
            "initial_friction_delta": min_delta,
            "recrystallized_gate_id": new_gate_id,
            "recrystallized_gate_name": new_gate_name,
            "rectified_delta_after": delta_after,
            "channel_open_now": is_open_now,
            "explanation": packet.explanation
        }

    def explain_self_reasoning(self) -> str:
        if not self.engraved_packets:
            return "No meta-packets engraved yet. Engine operates in silent initial ground."

        lines = ["[Fractal Gate Language Engine - Self-Explanation Narrative]"]
        for idx, packet in enumerate(self.engraved_packets, 1):
            lines.append(
                f"\n--- Engraved Event #{idx} ---"
                f"\n• Origin Ground: {packet.ground_a.get('name', 'UnknownGround')}"
                f"\n• Evaluated Friction Delta (D): {packet.delta_d:.4f} (Applied V_th: {packet.v_th:.4f})"
                f"\n• Gate Channel Open: {packet.channel_open}"
                f"\n• Causal Trace: {packet.causal_trace}"
                f"\n• Self-Elucidation: {packet.explanation}"
            )
        return "\n".join(lines)
