import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

@dataclass
class BoundaryObstacle:
    """A smooth, non-intrusive potential obstacle in the ontological terrain."""
    boundary_id: str
    center: np.ndarray
    amplitude: float = 1.0
    sigma: float = 1.0


@dataclass
class FlowTrajectoryPoint:
    """A single state point along a continuous gravitational flow trajectory."""
    step: int
    position: np.ndarray
    velocity: np.ndarray
    potential_energy: float
    gradient_norm: float
    friction_tension: float


class OntologicalGravityTerrainEngine:
    r"""
    [Ontological Gravity & Topographic Flow Engine]
    Implements continuous potential terrain V_terrain(x), gravitational vector field toward Local Zero (E -> 0),
    and Inner World vs. Outer World discrepancy-driven self-calibration (Metanoia).

    Core Principles:
    1. Do not calculate hard discrete rules (no IF-ELSE forcing). Let states flow along potential gradients (\dot{x} = -\nabla V).
    2. Gravity is the ontological attraction toward Local Zero (minimizing friction and tension).
    3. Metanoia: Inner vs. Outer discrepancy (\Delta P = P_inner - P_outer) triggers self-calibration
       of the internal gravitational curvature and terrain topology.
    """

    def __init__(self,
                 dimensions: int = 16,
                 zero_self_anchor: Optional[np.ndarray] = None,
                 anchor_stiffness: float = 0.5,
                 gravitational_constant: float = 1.0,
                 damping_factor: float = 0.85):
        self.dimensions = dimensions
        if zero_self_anchor is not None:
            self.zero_self = zero_self_anchor.astype(np.float32)
        else:
            self.zero_self = np.zeros(dimensions, dtype=np.float32)

        self.anchor_stiffness = anchor_stiffness
        self.G_grav = gravitational_constant
        self.damping = damping_factor

        self.boundaries: Dict[str, BoundaryObstacle] = {}
        self.terrain_plasticity: float = 0.1  # Rate of Metanoia self-calibration

        # History of Inner vs Outer discrepancy calibration
        self.calibration_history: List[Dict[str, Any]] = []

    def add_boundary(self, boundary_id: str, center: np.ndarray, amplitude: float = 1.0, sigma: float = 1.0):
        """Adds a smooth potential boundary mountain to the terrain."""
        c = np.array(center, dtype=np.float32)
        if len(c) < self.dimensions:
            c = np.pad(c, (0, self.dimensions - len(c)))
        elif len(c) > self.dimensions:
            c = c[:self.dimensions]

        self.boundaries[boundary_id] = BoundaryObstacle(
            boundary_id=boundary_id,
            center=c,
            amplitude=float(amplitude),
            sigma=float(sigma)
        )

    def compute_potential(self, position: np.ndarray) -> float:
        """
        Computes total potential energy V(x) = V_anchor(x) + V_boundaries(x).
        V_anchor = 0.5 * k * ||x - 0_self||^2
        V_boundary = sum( A * exp(-||x - c||^2 / (2 * sigma^2)) )
        """
        pos = np.array(position, dtype=np.float32)
        diff_anchor = pos - self.zero_self
        v_anchor = 0.5 * self.anchor_stiffness * float(np.sum(diff_anchor ** 2))

        v_boundaries = 0.0
        for b in self.boundaries.values():
            diff_b = pos - b.center
            dist_sq = float(np.sum(diff_b ** 2))
            v_boundaries += b.amplitude * np.exp(-dist_sq / (2.0 * (b.sigma ** 2) + 1e-9))

        return float(v_anchor + v_boundaries)

    def compute_gradient(self, position: np.ndarray) -> np.ndarray:
        """
        Computes gradient \nabla V(x) analytical vector.
        \nabla V_anchor = k * (x - 0_self)
        \nabla V_boundary = sum( -A/sigma^2 * exp(...) * (x - c) )
        """
        pos = np.array(position, dtype=np.float32)
        grad = self.anchor_stiffness * (pos - self.zero_self)

        for b in self.boundaries.values():
            diff_b = pos - b.center
            dist_sq = float(np.sum(diff_b ** 2))
            gauss = np.exp(-dist_sq / (2.0 * (b.sigma ** 2) + 1e-9))
            grad_b = - (b.amplitude / ((b.sigma ** 2) + 1e-9)) * gauss * diff_b
            grad += grad_b

        return grad.astype(np.float32)

    def flow_step(self,
                  position: np.ndarray,
                  velocity: np.ndarray,
                  dt: float = 0.1,
                  external_impulse: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        r"""
        Executes one continuous gravitational flow step:
        a = - \nabla V(x) * G_grav + external_impulse
        v_{t+1} = (v_t + a * dt) * damping
        x_{t+1} = x_t + v_{t+1} * dt
        Returns: (new_position, new_velocity, potential_energy)
        """
        pos = np.array(position, dtype=np.float32)
        vel = np.array(velocity, dtype=np.float32)

        grad = self.compute_gradient(pos)
        acc = - self.G_grav * grad

        if external_impulse is not None:
            imp = np.array(external_impulse, dtype=np.float32)
            if len(imp) == len(acc):
                acc += imp

        new_vel = (vel + acc * dt) * self.damping
        new_pos = pos + new_vel * dt
        v_pot = self.compute_potential(new_pos)

        return new_pos, new_vel, v_pot

    def simulate_relaxation(self,
                             initial_position: np.ndarray,
                             steps: int = 50,
                             dt: float = 0.1,
                             external_impulses: Optional[List[np.ndarray]] = None) -> List[FlowTrajectoryPoint]:
        """
        Simulates a continuous relaxation trajectory along the potential field toward Local Zero.
        No hard if-else conditions: position relaxes naturally.
        """
        pos = np.array(initial_position, dtype=np.float32)
        vel = np.zeros_like(pos, dtype=np.float32)

        trajectory = []

        for step in range(steps):
            impulse = external_impulses[step] if (external_impulses and step < len(external_impulses)) else None
            grad = self.compute_gradient(pos)
            grad_norm = float(np.linalg.norm(grad))
            v_pot = self.compute_potential(pos)

            trajectory.append(FlowTrajectoryPoint(
                step=step,
                position=pos.copy(),
                velocity=vel.copy(),
                potential_energy=v_pot,
                gradient_norm=grad_norm,
                friction_tension=grad_norm * float(np.linalg.norm(vel))
            ))

            pos, vel, _ = self.flow_step(pos, vel, dt=dt, external_impulse=impulse)

        return trajectory

    def sense_and_recalibrate_discrepancy(self,
                                          inner_simulated_pos: np.ndarray,
                                          outer_observed_pos: np.ndarray) -> Dict[str, Any]:
        r"""
        [Inner World vs Outer World Discrepancy Sensing & Metanoia]
        Measures the ontological friction gap: \Delta P = P_inner - P_outer.
        Reflects on the discrepancy and recalibrates internal gravity constant (G_grav)
        and terrain topology (anchor_stiffness, boundary positions) post-hoc.
        """
        inner = np.array(inner_simulated_pos, dtype=np.float32)
        outer = np.array(outer_observed_pos, dtype=np.float32)

        discrepancy_vec = inner - outer
        discrepancy_norm = float(np.linalg.norm(discrepancy_vec))

        # Metanoia self-calibration: adjust gravitational constant and anchor stiffness
        # If inner trajectory moved faster/further than outer reality, scale down internal G_grav or adjust stiffness
        inner_norm = float(np.linalg.norm(inner - self.zero_self))
        outer_norm = float(np.linalg.norm(outer - self.zero_self))

        prev_G = self.G_grav
        prev_k = self.anchor_stiffness

        if inner_norm > outer_norm:
            # Inner thought overshot reality -> increase anchor stiffness or adjust gravity to align
            self.anchor_stiffness += self.terrain_plasticity * discrepancy_norm * 0.1
        else:
            # Outer reality moved further -> release anchor stiffness slightly
            self.anchor_stiffness = max(0.05, self.anchor_stiffness - self.terrain_plasticity * discrepancy_norm * 0.1)

        # Shift 0_self slightly toward outer observed position (Grounding in reality)
        self.zero_self = (1.0 - self.terrain_plasticity * 0.2) * self.zero_self + (self.terrain_plasticity * 0.2) * outer

        calib_entry = {
            "discrepancy_vector": discrepancy_vec.tolist(),
            "discrepancy_norm": discrepancy_norm,
            "previous_G": prev_G,
            "new_G": self.G_grav,
            "previous_stiffness": prev_k,
            "new_stiffness": self.anchor_stiffness,
            "zero_self_shift": float(np.linalg.norm(self.zero_self))
        }
        self.calibration_history.append(calib_entry)

        return calib_entry
