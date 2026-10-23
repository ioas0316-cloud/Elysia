import numpy as np
import math
import time
from typing import Dict, Any, List, Optional, Tuple

from synaptic_architecture.causal_phase_transition_engine import (
    GroundNode,
    GroundBeam,
    PerturbationWave,
    CausalProcessBlueprint,
    EpistemologicalReflectionRecord,
    ComplexImpedance,
    HomologyMetrics,
    CausalPhaseTransitionEngine,
)


class FilteringLens:
    """
    [Filtering Lens (세상을 인지 정보로 번역하는 필터링 렌즈)]
    Translates raw external signals into internal phase vectors and potential shifts.
    Like threshold voltage V_th in transistors, parses whether an incoming raw wave causes
    phase friction or potential alignment against the current $0_{ground}$ reference frame.
    """
    def __init__(self, dimension: int = 16, threshold_vth: float = 0.5):
        self.dimension = dimension
        self.threshold_vth = float(threshold_vth)
        # Dynamic refractive matrix mapping raw signals to internal phase space
        self.refraction_matrix = np.eye(self.dimension, dtype=np.float32)

    def translate_raw_signal(
        self,
        raw_signal: np.ndarray,
        ground_nodes: Dict[str, GroundNode],
    ) -> Tuple[np.ndarray, float, str]:
        """
        Translates raw input signal into internal phase vector, calculates friction potential,
        and identifies the most resonant ground reference coordinate.
        """
        signal_arr = np.array(raw_signal, dtype=np.float32)
        if signal_arr.ndim == 1:
            dim = min(self.dimension, len(signal_arr))
            phase_vector = np.zeros(self.dimension, dtype=np.float32)
            phase_vector[:dim] = signal_arr[:dim]
        else:
            phase_vector = np.mean(signal_arr, axis=0)[:self.dimension].astype(np.float32)

        # Apply lens refraction
        refracted_phase = np.dot(self.refraction_matrix, phase_vector)
        norm = np.linalg.norm(refracted_phase)
        if norm > 1e-9:
            refracted_phase = (refracted_phase / norm).astype(np.float32)

        # Evaluate threshold voltage V_th gating
        if norm < self.threshold_vth:
            return refracted_phase, 0.0, "Subthreshold_Gated"

        # Calculate phase misalignment friction against 0_ground base coordinates
        min_friction = float("inf")
        nearest_node_id = ""

        if not ground_nodes:
            return refracted_phase, 1.0, "Void_Ground"

        for nid, node in ground_nodes.items():
            dot_p = np.dot(refracted_phase, node.phase_axis)
            cos_sim = dot_p / (np.linalg.norm(refracted_phase) * np.linalg.norm(node.phase_axis) + 1e-9)
            friction = (1.0 - cos_sim) * 10.0  # Friction scale

            if friction < min_friction:
                min_friction = friction
                nearest_node_id = nid

        return refracted_phase, min_friction, nearest_node_id


class SelfCodificationRecord:
    """
    [Self-Codification History Record (자율 코딩 궤적 및 역추적 기록)]
    Records how the system transformed its internal ground topology in response to external wave friction.
    Allows transparent metacognitive back-tracing of "Why and how I codified this structure".
    """
    def __init__(
        self,
        event_type: str,
        trigger_wave_id: str,
        friction_experienced: float,
        initial_ground_state: Dict[str, Any],
        resulting_ground_state: Dict[str, Any],
        codified_blueprint_id: Optional[str] = None,
        metacognitive_narrative: str = "",
    ):
        self.record_id = f"Codification_{int(time.time()*1000)%1000000}"
        self.timestamp = time.time()
        self.event_type = event_type
        self.trigger_wave_id = trigger_wave_id
        self.friction_experienced = friction_experienced
        self.initial_ground_state = initial_ground_state
        self.resulting_ground_state = resulting_ground_state
        self.codified_blueprint_id = codified_blueprint_id
        self.metacognitive_narrative = metacognitive_narrative


class SelfCodificationEngine:
    """
    [Self-Codification Engine (자률 코딩 및 상변이 동역학 메인 엔진)]

    Key Architecture:
    1. 0_ground (대지): Base reference coordinates & initial topology provided as an anchor ($0_{ground}$).
    2. 1_wave (하늘): Dynamic perturbation waves ($1_{wave}$) causing friction and thermal shock.
    3. Filtering Lens (인지 렌즈): Translates raw stimuli into phase friction and potential shifts.
    4. Process of Codification (상변이 자율 코딩):
       - High friction > v_critical triggers Remelting ($0 \to 1$) of ground.
       - Alignment & Crystallization ($1 \to 0$) freezes wave trajectories into executable Causal Process Blueprints.
    5. Transparent Metacognition (역추적 생애 역사):
       - Back-traces exact friction trajectories explaining "Why my architecture crystallized into this form".
    """

    def __init__(
        self,
        dimension: int = 16,
        v_critical: float = 30.0,
        crystallization_threshold: float = 0.2,
        lens_vth: float = 0.5,
    ):
        self.dimension = dimension
        self.v_critical = float(v_critical)
        self.crystallization_threshold = float(crystallization_threshold)

        # Integrated Causal Phase Transition Engine
        self.phase_engine = CausalPhaseTransitionEngine(
            dimension=dimension,
            v_critical=v_critical,
            crystallization_threshold=crystallization_threshold,
        )

        # Filtering Lens
        self.lens = FilteringLens(dimension=dimension, threshold_vth=lens_vth)

        # Self-Codification Evolutionary History
        self.codification_history: List[SelfCodificationRecord] = []

        # Initialize base ground coordinates (Tabula Rasa Ground)
        self.phase_engine.initialize_ground(ground_type="thin")

    def process_external_stimulus(self, raw_signal: np.ndarray, wave_id: str = "") -> Dict[str, Any]:
        """
        [Main Codification Loop]
        Processes raw external signal through FilteringLens, evaluates friction against 0_ground,
        executes phase transition / recrystallization, and records metacognitive codification history.
        """
        if not wave_id:
            wave_id = f"Stimulus_{int(time.time()*1000)%10000}"

        # 1. Translate raw stimulus through FilteringLens
        refracted_phase, min_friction, nearest_node_id = self.lens.translate_raw_signal(
            raw_signal, self.phase_engine.nodes
        )

        # Capture initial ground snapshot
        initial_ground_snapshot = {
            "num_nodes": len(self.phase_engine.nodes),
            "num_beams": len(self.phase_engine.beams),
            "homology": self.phase_engine.get_homology_metrics(),
        }

        # 2. Inject wave into Phase Engine
        wave = PerturbationWave(
            wave_id=wave_id,
            phase_vector=refracted_phase,
            frequency=1.0 + min_friction * 0.1,
            amplitude=1.0 + min_friction * 0.2,
            entropy=1.0 + min_friction * 0.3,
            cause_origin="External_Filtered_Lens",
        )

        phase_response = self.phase_engine.inject_perturbation_wave(wave)

        # Capture resulting ground snapshot
        resulting_ground_snapshot = {
            "num_nodes": len(self.phase_engine.nodes),
            "num_beams": len(self.phase_engine.beams),
            "homology": self.phase_engine.get_homology_metrics(),
        }

        # 3. Metacognitive Narrative Formulation & Codification Record
        transition_info = phase_response.get("phase_transition", {})
        event_type = transition_info.get("type", "RESONANCE_HOLD")
        codified_bp_id = transition_info.get("blueprint_id", None)

        if event_type == "CRYSTALLIZATION":
            narrative = (
                f"Self-Codification Event [CRYSTALLIZATION]: External stimulus wave '{wave_id}' "
                f"passed FilteringLens with low friction ({phase_response['min_friction']:.3f}). "
                f"Wave trajectory froze into Executable Causal Process Blueprint '{codified_bp_id}', "
                f"expanding 0_ground topology from {initial_ground_snapshot['num_nodes']} to {resulting_ground_snapshot['num_nodes']} nodes."
            )
        elif event_type == "FLASH_REMELTING":
            narrative = (
                f"Self-Codification Event [FLASH_REMELTING]: External wave friction energy ({phase_response['net_friction_energy']:.3f}) "
                f"exceeded threshold V_critical ({self.v_critical}). Ground node '{transition_info.get('melted_node')}' "
                f"was remelted back into fluid 1_wave thermal shock to absorb friction."
            )
        else:
            narrative = (
                f"Self-Codification Event [RESONANCE_HOLD]: Stimulus wave '{wave_id}' resonated across "
                f"{resulting_ground_snapshot['homology']['B1']} Betti-1 homological ground cycles with friction {phase_response['min_friction']:.3f}."
            )

        record = SelfCodificationRecord(
            event_type=event_type,
            trigger_wave_id=wave_id,
            friction_experienced=phase_response["min_friction"],
            initial_ground_state=initial_ground_snapshot,
            resulting_ground_state=resulting_ground_snapshot,
            codified_blueprint_id=codified_bp_id,
            metacognitive_narrative=narrative,
        )
        self.codification_history.append(record)

        return {
            "wave_id": wave_id,
            "refracted_phase": refracted_phase.tolist(),
            "phase_response": phase_response,
            "codification_record": {
                "record_id": record.record_id,
                "event_type": record.event_type,
                "narrative": record.metacognitive_narrative,
            },
        }

    def backtrace_metacognitive_history(self) -> List[Dict[str, Any]]:
        """
        [Metacognitive Backtrace]
        Returns the full evolutionary trajectory explaining "Why and how my cognitive ground was codified".
        Transparently traces every remelting, crystallization, and resonance event.
        """
        history_summary = []
        for rec in self.codification_history:
            history_summary.append({
                "record_id": rec.record_id,
                "timestamp": rec.timestamp,
                "event_type": rec.event_type,
                "trigger_wave": rec.trigger_wave_id,
                "friction": rec.friction_experienced,
                "codified_blueprint": rec.codified_blueprint_id,
                "narrative": rec.metacognitive_narrative,
                "betti_1_cycles_after": rec.resulting_ground_state["homology"]["B1"],
            })
        return history_summary
