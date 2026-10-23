"""
Enactive Boundary Layer (실재적 마찰 경계층)
==============================================
Provides the interface between internal causal field projections and non-negotiable
external environmental constraints. Rejects scalar loss functions and backpropagation.
Operates on wave phase coherence, phase lag, and relational edge impedance recalibration.

Modules:
1. PerceptualProjectionModule: Projects intentional causal prediction waves Psi_pred(t).
2. EnvironmentalConstraintReceiver: Receives non-negotiable external reaction waves Psi_ext(t).
3. PhaseFrictionSensor: Evaluates complex phase coherence, friction F, and phase lag Delta_phi.
4. FrictionSensorLensCalibrator: Recalibrates lens angles phi and updates relational edge impedance Z.
5. EnactiveBoundaryLayer: High-level orchestrator integrating projection, reception, friction sensing, and lens recalibration.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import networkx as nx
import math

from core.lens.cognitive_lens_engine import CognitiveLensEngine, ContextualDimension, RefractedObservation


@dataclass
class WaveSignal:
    """Complex causal wave representation Psi(t) = A * exp(i * (omega * t + phi))."""
    domain: str
    time_axis: np.ndarray
    wave_data: np.ndarray  # Complex 1D array
    amplitude: float
    frequency: float
    phase_angle: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrictionEvaluation:
    """Quantitative phase friction & discrepancy metrics."""
    friction_factor: float          # F = 1.0 - coherence (0.0 to 1.0)
    coherence: float                # Phase coherence (1.0 = resonance, 0.0 = orthogonal)
    phase_lag_rad: float            # Delta_phi in radians
    requires_recalibration: bool    # True if friction > threshold
    recalibration_signal: Dict[str, Any] = field(default_factory=dict)


class PerceptualProjectionModule:
    """Projects internal causal field expectations into complex wave signals Psi_pred(t)."""

    def __init__(self, sample_points: int = 100, duration: float = 1.0):
        self.sample_points = sample_points
        self.duration = duration
        self.t = np.linspace(0.0, duration, sample_points)

    def project_wave(self, domain: str, frequency: float, phase_angle: float, amplitude: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> WaveSignal:
        """Generates complex wave projection Psi_pred(t) = A * exp(i * (omega * t + phi))."""
        omega = 2.0 * np.pi * frequency
        complex_wave = amplitude * np.exp(1j * (omega * self.t + phase_angle))
        return WaveSignal(
            domain=domain,
            time_axis=self.t.copy(),
            wave_data=complex_wave,
            amplitude=amplitude,
            frequency=frequency,
            phase_angle=phase_angle,
            metadata=metadata or {}
        )


class EnvironmentalConstraintReceiver:
    """Receives non-negotiable external reaction waves Psi_ext(t) from real-world constraints."""

    def __init__(self, sample_points: int = 100, duration: float = 1.0):
        self.sample_points = sample_points
        self.duration = duration
        self.t = np.linspace(0.0, duration, sample_points)

    def receive_reaction(self, domain: str, frequency: float, phase_angle: float, amplitude: float = 1.0, noise_level: float = 0.0, metadata: Optional[Dict[str, Any]] = None) -> WaveSignal:
        """Captures ground reality wave response Psi_ext(t)."""
        omega = 2.0 * np.pi * frequency
        base_wave = amplitude * np.exp(1j * (omega * self.t + phase_angle))

        if noise_level > 0.0:
            noise_real = np.random.normal(0.0, noise_level, size=self.sample_points)
            noise_imag = np.random.normal(0.0, noise_level, size=self.sample_points)
            base_wave = base_wave + (noise_real + 1j * noise_imag)

        return WaveSignal(
            domain=domain,
            time_axis=self.t.copy(),
            wave_data=base_wave,
            amplitude=amplitude,
            frequency=frequency,
            phase_angle=phase_angle,
            metadata=metadata or {}
        )


class PhaseFrictionSensor:
    """Measures complex phase coherence, friction factor F, and phase lag Delta_phi."""

    def __init__(self, tolerance_threshold: float = 0.15):
        self.threshold = tolerance_threshold

    def evaluate(self, pred_wave: WaveSignal, ext_wave: WaveSignal) -> FrictionEvaluation:
        """Computes phase alignment coherence, friction F, and phase lag Delta_phi."""
        w_pred = pred_wave.wave_data
        w_ext = ext_wave.wave_data

        # Complex cross-correlation real part (phase alignment)
        real_inner_prod = np.real(np.sum(w_pred * np.conj(w_ext)))
        norm_pred = np.sum(np.abs(w_pred) ** 2)
        norm_ext = np.sum(np.abs(w_ext) ** 2)
        norm = np.sqrt(norm_pred * norm_ext) + 1e-9

        raw_coherence = real_inner_prod / norm
        coherence = float(np.clip(raw_coherence, 0.0, 1.0))
        friction_factor = float(1.0 - coherence)

        # Extract mean phase lag angle difference
        angle_pred = np.angle(w_pred)
        angle_ext = np.angle(w_ext)
        phase_diff = np.abs(angle_pred - angle_ext)
        phase_diff = np.mod(phase_diff, 2.0 * np.pi)
        phase_diff = np.minimum(phase_diff, 2.0 * np.pi - phase_diff)
        mean_phase_lag = float(np.mean(phase_diff))

        requires_recalibration = friction_factor > self.threshold

        recal_signal = {
            "domain": pred_wave.domain,
            "friction_level": friction_factor,
            "coherence": coherence,
            "phase_lag_rad": mean_phase_lag,
            "requires_recalibration": requires_recalibration,
            "suggested_phase_shift": mean_phase_lag if angle_pred.mean() > angle_ext.mean() else -mean_phase_lag
        }

        return FrictionEvaluation(
            friction_factor=friction_factor,
            coherence=coherence,
            phase_lag_rad=mean_phase_lag,
            requires_recalibration=requires_recalibration,
            recalibration_signal=recal_signal
        )


class FrictionSensorLensCalibrator:
    """Recalibrates lens phase angles and updates relational edge impedance Z in NetworkX topology."""

    def __init__(self, alpha: float = 0.4, beta: float = 0.15, max_impedance: float = 0.8, min_impedance: float = 0.01):
        self.alpha = alpha            # Friction impedance amplification coefficient
        self.beta = beta              # Resonance impedance consolidation coefficient
        self.max_impedance = max_impedance
        self.min_impedance = min_impedance

    def recalibrate_node_phase(self, graph: nx.DiGraph, node_name: str, phase_lag: float) -> float:
        """Adjusts node phase angle by subtracting phase lag: phi_new = phi_old - phase_lag."""
        if node_name not in graph.nodes:
            raise ValueError(f"Node '{node_name}' not in graph.")

        old_phase = graph.nodes[node_name]["phase"]
        new_phase = old_phase - phase_lag
        # Wrap to [-pi, pi]
        new_phase = math.atan2(math.sin(new_phase), math.cos(new_phase))
        graph.nodes[node_name]["phase"] = new_phase
        return new_phase

    def update_edge_impedance(self, graph: nx.DiGraph, source: str, target: str, friction_factor: float, threshold: float = 0.15) -> float:
        """
        Dynamically updates relational edge impedance Z:
        - High friction (F > threshold): Z_new = Z_old + alpha * F
        - Resonance (F <= threshold):    Z_new = max(min_impedance, Z_old * (1 - beta))
        """
        if not graph.has_edge(source, target):
            graph.add_edge(source, target, impedance=0.1)

        current_z = graph.edges[source, target].get("impedance", 0.1)

        if friction_factor > threshold:
            new_z = current_z + (self.alpha * friction_factor)
            graph.edges[source, target]["impedance"] = new_z
        else:
            new_z = max(self.min_impedance, current_z * (1.0 - self.beta))
            graph.edges[source, target]["impedance"] = new_z

        return new_z


class EnactiveBoundaryLayer:
    """
    High-Level Enactive Boundary Layer (실재적 마찰 경계층).
    Integrates projection, environment reception, phase friction measurement, and topological recalibration.
    """

    def __init__(self, lens_engine: Optional[CognitiveLensEngine] = None, alpha: float = 0.4, beta: float = 0.15, threshold: float = 0.15):
        self.projector = PerceptualProjectionModule()
        self.receiver = EnvironmentalConstraintReceiver()
        self.friction_sensor = PhaseFrictionSensor(tolerance_threshold=threshold)
        self.calibrator = FrictionSensorLensCalibrator(alpha=alpha, beta=beta)
        self.lens_engine = lens_engine or CognitiveLensEngine()

        self.graph = nx.DiGraph()
        self.threshold = threshold

    def add_causal_node(self, name: str, frequency: float, phase: float, dimension: Optional[ContextualDimension] = None):
        """Registers a causal lens node in the boundary graph."""
        self.graph.add_node(name, freq=frequency, phase=phase, dimension=dimension)

    def add_causal_edge(self, source: str, target: str, initial_impedance: float = 0.1):
        """Registers a relational edge with impedance Z."""
        self.graph.add_edge(source, target, impedance=initial_impedance)

    def enact_step(self, source_node: str, external_frequency: float, external_phase: float, target_node: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes a 4-step Enactive Recalibration Cycle:
        1. Intentional Projection: Psi_pred(t) generated from source_node.
        2. Dynamic Impact: Psi_ext(t) captured from external reality.
        3. Friction Extraction: Measure phase friction F and phase lag Delta_phi.
        4. Lens Recalibration: Recalibrate node phase angle and edge impedance Z.
        """
        if source_node not in self.graph.nodes:
            raise ValueError(f"Source node '{source_node}' not found in boundary graph.")

        node_data = self.graph.nodes[source_node]
        freq = node_data["freq"]
        phase = node_data["phase"]

        # Step 1: Project expectation
        pred_wave = self.projector.project_wave(
            domain=source_node,
            frequency=freq,
            phase_angle=phase,
            amplitude=1.0
        )

        # Step 2: Receive environment reaction
        ext_wave = self.receiver.receive_reaction(
            domain=source_node,
            frequency=external_frequency,
            phase_angle=external_phase,
            amplitude=1.0
        )

        # Step 3: Extract friction
        friction_eval = self.friction_sensor.evaluate(pred_wave, ext_wave)

        # Step 4: Recalibrate
        target = target_node
        if not target:
            out_edges = list(self.graph.out_edges(source_node))
            if out_edges:
                target = out_edges[0][1]

        updated_z = None
        if target:
            updated_z = self.calibrator.update_edge_impedance(
                self.graph, source_node, target, friction_eval.friction_factor, self.threshold
            )

        phase_recalibrated = False
        if friction_eval.requires_recalibration:
            new_phase = self.calibrator.recalibrate_node_phase(
                self.graph, source_node, friction_eval.phase_lag_rad
            )
            phase_recalibrated = True

            # If node is associated with a CognitiveLens dimension, update its curvature
            dim = node_data.get("dimension")
            if dim and dim in self.lens_engine.lenses:
                current_curv = self.lens_engine.lenses[dim].curvature
                new_curv = max(0.1, current_curv + 0.1 * friction_eval.friction_factor)
                self.lens_engine.adjust_lens_curvature(dim, new_curv)

        path_blocked = (updated_z is not None and updated_z > self.calibrator.max_impedance)

        return {
            "source_node": source_node,
            "target_node": target,
            "friction_factor": friction_eval.friction_factor,
            "coherence": friction_eval.coherence,
            "phase_lag_rad": friction_eval.phase_lag_rad,
            "phase_recalibrated": phase_recalibrated,
            "new_source_phase": self.graph.nodes[source_node]["phase"],
            "updated_edge_impedance": updated_z,
            "path_blocked": path_blocked,
            "status": "BLOCKED" if path_blocked else ("RECALIBRATED" if phase_recalibrated else "RESONANCE")
        }
