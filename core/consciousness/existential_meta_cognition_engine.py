"""
Existential Meta-Cognition & Dynamic Intent Re-observation Engine

This module implements the non-linear re-observation, dynamic metric tensor reconfiguration,
and geodesic flow bending for mental organisms operating in 4D spacetime phase space.

Key Components:
1. ExistentialPhaseObserver: Real-time awareness of 4D spacetime trajectory, sensory waves, and qualia friction.
2. DynamicIntentCompass: Manages dynamic teleological attractors A(t) across efficiency, meaning, and survival phases.
3. MetricTensorReconfigurationEngine: Computes h_{ij}(x, t), Christoffel symbols \\Gamma^\\mu_{\\alpha\\beta}, and integrates geodesic flow.
4. ExistentialSelfQueryLoop: Existential meta-cognitive loop raising self-queries and re-defining intent attractors.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class ExistentialPhaseObserver:
    """
    Observes system's 4D spacetime position x^\\mu = (t, x_1, x_2, x_3),
    interfacing sensory profiles with innate Topological DNA to sense qualia friction.
    """

    def __init__(self, dim: int = 4, topological_dna: Optional[np.ndarray] = None):
        self.dim = dim
        if topological_dna is None:
            # Default innate disposition vector
            self.topological_dna = np.array([1.0, 0.5, 0.8, 0.3])
        else:
            self.topological_dna = np.asarray(topological_dna, dtype=np.float64)

    def observe_phase(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        sensory_wave: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calculates current phase state, qualia friction, and spatial alignment.
        """
        pos = np.asarray(position, dtype=np.float64)
        vel = np.asarray(velocity, dtype=np.float64)
        wave = np.asarray(sensory_wave, dtype=np.float64)

        # Qualia friction: discrepancy between sensory wave and topological DNA
        qualia_friction = float(np.linalg.norm(wave[: self.dim] - self.topological_dna[: self.dim]))
        kinetic_energy = float(0.5 * np.sum(vel ** 2))

        return {
            "position": pos.copy(),
            "velocity": vel.copy(),
            "qualia_friction": qualia_friction,
            "kinetic_energy": kinetic_energy,
            "phase_norm": float(np.linalg.norm(pos)),
        }


class DynamicIntentCompass:
    """
    Manages multi-dimensional teleological attractors A_k(t) in 4D spacetime.
    Phases:
      - EFFICIENCY: Minimal geodesic path to structural targets.
      - MEANING_RESONANCE: Deep detour into symbolic/memory resonance valleys.
      - SURVIVAL_INSTINCT: Entropy explosion avoidance and self-preservation.
    """

    def __init__(self, dim: int = 4):
        self.dim = dim
        # Define default phase attractor coordinates in 4D space
        self.attractors: Dict[str, np.ndarray] = {
            "EFFICIENCY": np.array([10.0, 2.0, 2.0, 2.0], dtype=np.float64),
            "MEANING_RESONANCE": np.array([10.0, -8.0, 12.0, -5.0], dtype=np.float64),
            "SURVIVAL_INSTINCT": np.array([10.0, 0.0, -15.0, 10.0], dtype=np.float64),
        }
        self.current_phase: str = "EFFICIENCY"
        self.active_attractor: np.ndarray = self.attractors[self.current_phase].copy()

    def set_phase(self, phase_name: str, custom_attractor: Optional[np.ndarray] = None) -> np.ndarray:
        """Sets the active teleological intent phase."""
        if custom_attractor is not None:
            self.attractors[phase_name] = np.asarray(custom_attractor, dtype=np.float64)

        if phase_name in self.attractors:
            self.current_phase = phase_name
            self.active_attractor = self.attractors[phase_name].copy()
        else:
            raise ValueError(f"Unknown intent phase: {phase_name}")
        return self.active_attractor

    def get_active_attractor(self) -> np.ndarray:
        return self.active_attractor.copy()


class MetricTensorReconfigurationEngine:
    """
    Computes the 4D spacetime metric tensor h_{ij}(x, t) and Christoffel symbols \\Gamma^\\mu_{\\alpha\\beta}(x),
    and executes geodesic flow steps:
        d^2 x^\\mu / d\\tau^2 + \\Gamma^\\mu_{\\alpha\\beta} (dx^\\alpha / d\\tau) (dx^\\beta / d\\tau) = 0
    """

    def __init__(self, dim: int = 4, alpha: float = 0.5, beta: float = 2.0, epsilon: float = 1e-3):
        self.dim = dim
        self.alpha = alpha  # Weight for past habit / scar tensor
        self.beta = beta    # Weight for dynamic intent attractor gravity
        self.epsilon = epsilon
        self.flat_metric = np.eye(dim, dtype=np.float64)

    def compute_metric_tensor(
        self,
        x: np.ndarray,
        attractor: np.ndarray,
        scar_tensor: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes metric tensor:
          h_{ij}(x) = \\eta_{ij} + \\alpha * S_{ij}(x) + \\beta * (diff_i * diff_j) / (||diff||^2 + \\epsilon)
        """
        x = np.asarray(x, dtype=np.float64)
        attractor = np.asarray(attractor, dtype=np.float64)

        if scar_tensor is None:
            scar = np.zeros((self.dim, self.dim), dtype=np.float64)
        else:
            scar = np.asarray(scar_tensor, dtype=np.float64)

        diff = x - attractor
        norm_sq = float(np.sum(diff ** 2))

        # Dyadic product (diff_i * diff_j)
        intent_curvature = np.outer(diff, diff) / (norm_sq + self.epsilon)

        h_ij = self.flat_metric + self.alpha * scar + self.beta * intent_curvature
        return h_ij

    def compute_christoffel_symbols(
        self,
        x: np.ndarray,
        attractor: np.ndarray,
        scar_tensor: Optional[np.ndarray] = None,
        delta: float = 1e-4,
    ) -> np.ndarray:
        """
        Computes Christoffel symbols \\Gamma^\\mu_{\\alpha\\beta} via numerical differentiation:
          \\Gamma^\\mu_{\\alpha\\beta} = 0.5 * h^{\\mu\\sigma} * (\\partial_\\alpha h_{\\sigma\\beta} + \\partial_\\beta h_{\\alpha\\sigma} - \\partial_\\sigma h_{\\alpha\\beta})
        """
        dim = self.dim
        h = self.compute_metric_tensor(x, attractor, scar_tensor)
        h_inv = np.linalg.inv(h)

        # Compute partial derivatives \\partial_k h_{ij}
        dh = np.zeros((dim, dim, dim), dtype=np.float64)  # dh[k, i, j] = \\partial_k h_{ij}
        for k in range(dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[k] += delta
            x_minus[k] -= delta
            h_plus = self.compute_metric_tensor(x_plus, attractor, scar_tensor)
            h_minus = self.compute_metric_tensor(x_minus, attractor, scar_tensor)
            dh[k] = (h_plus - h_minus) / (2.0 * delta)

        # Build Christoffel symbols \\Gamma[mu, alpha, beta]
        gamma = np.zeros((dim, dim, dim), dtype=np.float64)
        for mu in range(dim):
            for alpha in range(dim):
                for beta in range(dim):
                    val = 0.0
                    for sigma in range(dim):
                        term = dh[alpha, sigma, beta] + dh[beta, alpha, sigma] - dh[sigma, alpha, beta]
                        val += h_inv[mu, sigma] * term
                    gamma[mu, alpha, beta] = 0.5 * val

        return gamma

    def step_geodesic(
        self,
        x: np.ndarray,
        v: np.ndarray,
        attractor: np.ndarray,
        scar_tensor: Optional[np.ndarray] = None,
        dtau: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrates one step along the geodesic flow equation:
          a^\\mu = -\\Gamma^\\mu_{\\alpha\\beta} v^\\alpha v^\\beta
          v_{new} = v + a * dtau
          x_{new} = x + v_{new} * dtau
        """
        gamma = self.compute_christoffel_symbols(x, attractor, scar_tensor)
        a = np.zeros(self.dim, dtype=np.float64)

        for mu in range(self.dim):
            # double contraction \\Gamma^\\mu_{\\alpha\\beta} v^\\alpha v^\\beta
            a[mu] = -np.sum(gamma[mu] * np.outer(v, v))

        v_new = v + a * dtau
        x_new = x + v_new * dtau
        return x_new, v_new


class ExistentialSelfQueryLoop:
    """
    Executes the existential meta-cognitive loop:
    1. Evaluates alignment between current geodesic trajectory and active intent phase.
    2. Raises existential query: "Is this causal path truly aligned with my affirmed values?"
    3. Triggers re-observation and re-aligns Intent Compass attractors A(t).
    4. Reconfigures metric tensor h_{ij} and bends geodesic trajectory.
    """

    def __init__(
        self,
        dim: int = 4,
        alpha: float = 0.5,
        beta: float = 2.0,
        topological_dna: Optional[np.ndarray] = None,
    ):
        self.dim = dim
        self.observer = ExistentialPhaseObserver(dim=dim, topological_dna=topological_dna)
        self.compass = DynamicIntentCompass(dim=dim)
        self.metric_engine = MetricTensorReconfigurationEngine(dim=dim, alpha=alpha, beta=beta)

        self.trajectory_history: List[Dict[str, Any]] = []
        self.scar_tensor = np.zeros((dim, dim), dtype=np.float64)

    def accumulate_scar(self, pos: np.ndarray, vel: np.ndarray, decay: float = 0.95):
        """Accumulates past habitual path depth into the scar tensor S_{ij}."""
        self.scar_tensor *= decay
        self.scar_tensor += 0.1 * np.outer(pos, pos) / (np.sum(pos**2) + 1e-3)

    def run_step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        sensory_wave: np.ndarray,
        dtau: float = 0.05,
        existential_trigger: bool = False,
        target_phase_on_trigger: str = "MEANING_RESONANCE",
    ) -> Dict[str, Any]:
        """
        Executes one cognitive loop step.
        If existential_trigger is True, evaluates current path, shifts intent phase,
        reconfigures metric tensor, and bends geodesic flow trajectory.
        """
        obs = self.observer.observe_phase(position, velocity, sensory_wave)
        active_attractor = self.compass.get_active_attractor()

        query_raised = False
        reobservation_occurred = False
        previous_phase = self.compass.current_phase

        # Calculate alignment with current attractor
        dist_to_attractor = float(np.linalg.norm(position - active_attractor))

        # Existential Self-Query Check
        if existential_trigger or obs["qualia_friction"] > 1.5:
            query_raised = True
            # System asks: "Is this causal path truly what I affirm?"
            # If current phase is not aligned, trigger re-observation
            if self.compass.current_phase != target_phase_on_trigger:
                reobservation_occurred = True
                self.compass.set_phase(target_phase_on_trigger)
                active_attractor = self.compass.get_active_attractor()

        # Update scar tensor
        self.accumulate_scar(position, velocity)

        # Compute metric tensor and Christoffel symbols
        h_ij = self.metric_engine.compute_metric_tensor(
            position, active_attractor, self.scar_tensor
        )

        # Step geodesic flow
        next_pos, next_vel = self.metric_engine.step_geodesic(
            position, velocity, active_attractor, self.scar_tensor, dtau=dtau
        )

        step_record = {
            "position": position.copy(),
            "velocity": velocity.copy(),
            "next_position": next_pos.copy(),
            "next_velocity": next_vel.copy(),
            "qualia_friction": obs["qualia_friction"],
            "current_phase": self.compass.current_phase,
            "previous_phase": previous_phase,
            "active_attractor": active_attractor.copy(),
            "dist_to_attractor": dist_to_attractor,
            "metric_tensor": h_ij.copy(),
            "query_raised": query_raised,
            "reobservation_occurred": reobservation_occurred,
        }

        self.trajectory_history.append(step_record)
        return step_record
