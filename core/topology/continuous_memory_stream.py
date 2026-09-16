"""
Continuous Memory Impedance Stream & Dimensionally Isomorphic Spatiotemporal Engine
===================================================================================
Elysia Core Topology Module

This module implements a pure, continuous causal structure where data and computation
are unified into a continuous byte-phase stream across isomorphic topological dimensions:
- 0D Point manifold (Discrete state values)
- 1D Vector manifold (Data momentum & velocity streams)
- 2D+ Tensor field manifold (Relational/Spatial memory topography)
- 3D/4D Spatiotemporal Manifold (Phase-Lock Spatiotemporal Axis)

Core Principles Implemented:
1. Primordial Potential Flow Gradient (Initial voltage/current vectors V_potential & I_flow).
2. Continuous Byte-Phase Stream without destructive 1D flattening overhead.
3. First Inflection Point Detection: Phase Derivative Asymmetry (grad^2 phi) and
   Chromatic Entropy Perturbation (delta Y).
4. Dynamic Impedance Damping Regulator: Auto-tunes resistance R(T) to converge
   causal tension T -> 0 (Principle of Least Action / Minimum Phase-Friction).
5. Non-Invasive Phase Flow Observatory & Proprioceptive Self-Correction.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union


@dataclass
class ChromaticSignature:
    """
    Chromatic Interconnectedness Signature (Flux, Order, Entropy)
    Representing physical-informational properties of the causal flow.
    """
    flux: float = 1.0     # Red: Energy, momentum, flow
    order: float = 1.0    # Blue: Axiomatic order, structural stability
    entropy: float = 0.0  # Yellow: Phase friction, noise, uncertainty

    def to_array(self) -> np.ndarray:
        return np.array([self.flux, self.order, self.entropy], dtype=np.float32)

    def perturb(self, delta_entropy: float, delta_flux: float = 0.0):
        self.entropy = max(0.0, self.entropy + delta_entropy)
        self.flux = max(0.0, self.flux + delta_flux)


@dataclass
class DimensionalIsomorphicNode:
    """
    A dimensionally isomorphic node maintaining topological dimension integrity.
    - 0D: Scalar/Point state
    - 1D: Vector stream / Momentum
    - 2D+: Spatial/Relational Tensor
    - 4D: Spatiotemporal Phase-Lock manifold
    """
    node_id: str
    dimension_type: str  # "0D_point", "1D_vector", "2D_field", "4D_spatiotemporal"
    raw_shape: Tuple[int, ...]
    phase_tensor: np.ndarray
    chromatic: ChromaticSignature = field(default_factory=ChromaticSignature)
    potential_gradient: float = 1.0  # Primordial potential differential
    phase_angle: float = 0.0         # Current phase angle phi
    phase_velocity: float = 1.0      # d_phi / dt


@dataclass
class FlowInflectionMetrics:
    """
    Real-time metrics for detecting the first inflection point of causal breakdown.
    """
    phase_derivative_asymmetry: float  # grad^2 phi = d^2 phi / dt^2
    chromatic_entropy_perturbation: float  # delta Y
    causal_tension: float  # Internal tension T
    impedance: float       # Dynamic impedance R(T)
    is_inflection_detected: bool


class ContinuousMemoryImpedanceStream:
    """
    Continuous Memory Stream & Dimensionally Isomorphic Spatiotemporal Engine.
    """

    def __init__(self, target_dimension: int = 8, initial_voltage: float = 1.0, initial_current: float = 1.0):
        self.target_dimension = target_dimension
        self.v_potential = float(initial_voltage)
        self.i_flow = float(initial_current)
        self.impedance = 0.1  # Initial dynamic impedance R
        self.tension = 0.0    # Causal tension T
        self.phase_lock_axis = np.ones(self.target_dimension, dtype=np.float32) / np.sqrt(self.target_dimension)

        # History for second derivative & entropy perturbation tracking
        self.phase_history: List[float] = []
        self.entropy_history: List[float] = []
        self.nodes: Dict[str, DimensionalIsomorphicNode] = {}

    def initialize_primordial_flow(self, voltage: float = 1.0, current: float = 1.0):
        """
        Initialize the primordial potential flow gradient upon system power-on.
        """
        self.v_potential = float(voltage)
        self.i_flow = float(current)
        self.impedance = self.v_potential / (self.i_flow + 1e-6)
        self.tension = 0.0

    def register_isomorphic_node(
        self,
        node_id: str,
        data: Union[int, float, List, np.ndarray],
        dimension_type: str = "auto",
        chromatic: Optional[ChromaticSignature] = None
    ) -> DimensionalIsomorphicNode:
        """
        Registers data preserving its native topological dimension (0D point, 1D vector, 2D field, etc.)
        without lossy 1D flattening.
        """
        arr = np.asarray(data, dtype=np.float32)
        if dimension_type == "auto":
            if arr.ndim == 0 or arr.size == 1:
                dimension_type = "0D_point"
            elif arr.ndim == 1:
                dimension_type = "1D_vector"
            elif arr.ndim == 2:
                dimension_type = "2D_field"
            else:
                dimension_type = "4D_spatiotemporal"

        if chromatic is None:
            chromatic = ChromaticSignature(flux=1.0, order=1.0, entropy=0.0)

        # Create phase tensor retaining raw shape
        node = DimensionalIsomorphicNode(
            node_id=node_id,
            dimension_type=dimension_type,
            raw_shape=arr.shape,
            phase_tensor=arr,
            chromatic=chromatic,
            potential_gradient=self.v_potential,
            phase_angle=0.0,
            phase_velocity=self.i_flow
        )
        self.nodes[node_id] = node
        return node

    def detect_first_inflection_point(self, node: DimensionalIsomorphicNode) -> FlowInflectionMetrics:
        """
        Detects the first subtle inflection point of breakdown in the continuous stream:
        1. Phase Derivative Asymmetry: grad^2 phi = d^2 phi / dt^2
        2. Chromatic Entropy Perturbation: delta Y
        """
        # Calculate current phase angle phi
        flat_phase = node.phase_tensor.flatten()
        norm_val = np.linalg.norm(flat_phase)
        current_phi = float(np.angle(np.sum(flat_phase) + 1j * norm_val))

        self.phase_history.append(current_phi)
        self.entropy_history.append(node.chromatic.entropy)

        if len(self.phase_history) > 30:
            self.phase_history.pop(0)
            self.entropy_history.pop(0)

        # Compute d^2 phi / dt^2
        if len(self.phase_history) >= 3:
            d1 = np.diff(self.phase_history)
            d2 = np.diff(d1)
            phase_der_asymmetry = float(abs(d2[-1]))
        else:
            phase_der_asymmetry = 0.0

        # Compute delta Y
        if len(self.entropy_history) >= 2:
            delta_y = float(abs(self.entropy_history[-1] - self.entropy_history[-2]))
        else:
            delta_y = node.chromatic.entropy

        # Causal tension T = grad^2 phi * (1 + delta Y)
        causal_tension = phase_der_asymmetry * (1.0 + delta_y) + node.chromatic.entropy * 0.5
        is_inflection = bool(phase_der_asymmetry > 0.15 or delta_y > 0.2)

        return FlowInflectionMetrics(
            phase_derivative_asymmetry=phase_der_asymmetry,
            chromatic_entropy_perturbation=delta_y,
            causal_tension=causal_tension,
            impedance=self.impedance,
            is_inflection_detected=is_inflection
        )

    def apply_dynamic_impedance_damping(self, metrics: FlowInflectionMetrics) -> float:
        """
        Dynamic Impedance Damping Regulator:
        Adjusts R(T) based on causal tension T to drive T -> 0 (Principle of Least Action).
        """
        self.tension = metrics.causal_tension
        if metrics.causal_tension > 1e-4:
            # Increase damping impedance R(T) proportional to tension to dissipate noise
            damping_factor = 0.2 * metrics.causal_tension
            self.impedance += damping_factor
            # System self-corrects: reduces tension by damping
            self.tension = max(0.0, self.tension - 0.5 * self.impedance)
        else:
            # System is in zero-tension equilibrium: relax impedance back to baseline
            self.impedance = max(0.05, self.impedance * 0.9)

        return self.impedance

    def propagate_spatiotemporal_phase_lock(self, dt: float = 0.1) -> Dict[str, FlowInflectionMetrics]:
        """
        Propagates continuous phase waves across all isomorphic nodes locked onto the spatiotemporal axis.
        Performs real-time self-correction driving tension toward zero.
        """
        metrics_dict = {}
        for node_id, node in self.nodes.items():
            # Update phase angle phi along primordial flow
            node.phase_angle = (node.phase_angle + node.phase_velocity * dt) % (2.0 * np.pi)

            # Phase wave transformation maintaining dimensional shape
            wave_factor = np.cos(node.phase_angle) + 1j * np.sin(node.phase_angle)
            node.phase_tensor = node.phase_tensor * float(np.abs(wave_factor))

            # Check inflection point
            metrics = self.detect_first_inflection_point(node)

            # Apply dynamic impedance damping self-correction
            if metrics.is_inflection_detected or metrics.causal_tension > 0.05:
                new_impedance = self.apply_dynamic_impedance_damping(metrics)
                # Damping absorbs entropy perturbation in chromatic signature
                node.chromatic.entropy = max(0.0, node.chromatic.entropy - 0.3 * new_impedance)
                # Recalculate metrics post-correction
                metrics = self.detect_first_inflection_point(node)

            metrics_dict[node_id] = metrics

        return metrics_dict

    def align_contextual_phase_axis(self, target_context_vector: np.ndarray):
        """
        Re-aligns the phase gradient axis G_s when the system's intentional context shifts
        (e.g., from ultra-low latency interactive response to data integrity).
        """
        vec = np.asarray(target_context_vector, dtype=np.float32).flatten()
        if len(vec) != self.target_dimension:
            indices = np.linspace(0, len(vec) - 1, self.target_dimension)
            vec = np.interp(indices, np.arange(len(vec)), vec).astype(np.float32)

        norm_val = np.linalg.norm(vec)
        if norm_val > 1e-8:
            self.phase_lock_axis = vec / norm_val

        # Context shift temporarily resets tension and aligns all node velocities
        self.tension = 0.0
        for node in self.nodes.values():
            node.phase_velocity = float(np.dot(self.phase_lock_axis, np.ones(self.target_dimension, dtype=np.float32) / np.sqrt(self.target_dimension)))
