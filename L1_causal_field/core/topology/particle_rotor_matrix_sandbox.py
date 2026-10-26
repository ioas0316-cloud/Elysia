"""
particle_rotor_matrix_sandbox.py: Micro-Rotor Particle Matrix Sandbox & Causal Timeline Scrubber.

Embodying the continuous, organic causality of Elysia:
- Micro-rotor particles act as microscopic cognitive/physical cells with position, velocity,
  phase angle (theta), angular velocity (omega), and somatic friction/mass.
- Local interactions follow potential fields and phase-locking forces (Kuramoto-like coupling
  driven by spatial proximity), giving rise to emergent macro-order (homeostasis).
- CausalTimelineScrubber enables backward/forward time scrubbing and counterfactual parallel
  trajectory simulation (tracing phase divergence origins).
"""

from dataclasses import dataclass, field
import copy
import numpy as np
from typing import List, Dict, Any, Optional, Callable


@dataclass
class MicroRotorParticle:
    """
    Micro-Rotor Particle acting as a microscopic cognitive-physical cell.
    """
    id: int
    position: np.ndarray  # (D,) position vector
    velocity: np.ndarray  # (D,) velocity vector
    mass: float = 1.0     # Semantic/physical mass (Ms)
    friction: float = 0.1 # Somatic resistance/damping
    theta: float = 0.0    # Rotor phase angle in radians [0, 2*pi)
    omega: float = 1.0    # Intrinsic angular velocity
    type_id: int = 0      # Particle flavor / chromatic classification
    phase_lock_degree: float = 0.0 # Degree of phase synchronization with neighbors
    sensory_profile: Dict[str, float] = field(default_factory=dict) # Physical sensations (pressure, weight, texture)

    def __post_init__(self):
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=float)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=float)
        if not self.sensory_profile:
            self.sensory_profile = {
                "pressure": float(self.mass * 9.81),
                "weight": float(self.mass),
                "texture": float(self.friction),
            }


class CausalTimelineScrubber:
    """
    Records causal reel history of the sandbox and allows forward/backward time scrubbing,
    state retrieval, and counterfactual trajectory branching.
    """

    def __init__(self, max_history: int = 2000):
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []

    def record_frame(self, step_idx: int, timestamp: float, particles: List[MicroRotorParticle], macro_order: float, total_energy: float) -> None:
        """
        Record snapshot of particles and macro order parameters.
        """
        frame = {
            "step_idx": step_idx,
            "timestamp": timestamp,
            "macro_order": float(macro_order),
            "total_energy": float(total_energy),
            "particles": [
                {
                    "id": p.id,
                    "position": p.position.copy(),
                    "velocity": p.velocity.copy(),
                    "mass": p.mass,
                    "friction": p.friction,
                    "theta": p.theta,
                    "omega": p.omega,
                    "type_id": p.type_id,
                    "phase_lock_degree": p.phase_lock_degree,
                    "sensory_profile": copy.deepcopy(p.sensory_profile),
                }
                for p in particles
            ]
        }
        self.history.append(frame)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def scrub_to(self, step_idx: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve frame at specified step_idx.
        """
        for frame in reversed(self.history):
            if frame["step_idx"] == step_idx:
                return frame
        return None

    def restore_particles_from_frame(self, frame: Dict[str, Any]) -> List[MicroRotorParticle]:
        """
        Reconstruct MicroRotorParticle list from snapshot frame.
        """
        particles = []
        for pdata in frame["particles"]:
            p = MicroRotorParticle(
                id=pdata["id"],
                position=pdata["position"].copy(),
                velocity=pdata["velocity"].copy(),
                mass=pdata["mass"],
                friction=pdata["friction"],
                theta=pdata["theta"],
                omega=pdata["omega"],
                type_id=pdata["type_id"],
                phase_lock_degree=pdata["phase_lock_degree"],
                sensory_profile=copy.deepcopy(pdata["sensory_profile"]),
            )
            particles.append(p)
        return particles

    def counterfactual_trace(
        self,
        start_step: int,
        steps_forward: int,
        perturbation_fn: Callable[[List[MicroRotorParticle]], None],
        sandbox_factory: Callable[[List[MicroRotorParticle]], Any],
    ) -> Dict[str, Any]:
        """
        Fork timeline at start_step, apply perturbation_fn, run forward steps_forward,
        and compare phase divergence with original history.
        """
        frame = self.scrub_to(start_step)
        if frame is None:
            raise ValueError(f"Step {start_step} not found in history.")

        forked_particles = self.restore_particles_from_frame(frame)
        perturbation_fn(forked_particles)

        forked_sandbox = sandbox_factory(forked_particles)
        forked_order_history = []
        for _ in range(steps_forward):
            forked_sandbox.step()
            forked_order_history.append(forked_sandbox.get_macro_order_parameter())

        # Measure divergence from baseline
        baseline_order = []
        for idx in range(start_step + 1, start_step + 1 + steps_forward):
            orig_frame = self.scrub_to(idx)
            if orig_frame:
                baseline_order.append(orig_frame["macro_order"])
            else:
                break

        min_len = min(len(baseline_order), len(forked_order_history))
        if min_len > 0:
            order_divergence = float(np.mean(np.abs(
                np.array(baseline_order[:min_len]) - np.array(forked_order_history[:min_len])
            )))
        else:
            order_divergence = 0.0

        return {
            "start_step": start_step,
            "steps_forward": steps_forward,
            "forked_order_history": forked_order_history,
            "baseline_order_history": baseline_order,
            "order_divergence": order_divergence,
        }


class ParticleRotorMatrixSandbox:
    """
    Organic Matrix Sandbox where micro-rotor particles interact via
    spatial potential fields and phase synchronization dynamics.
    """

    def __init__(
        self,
        particles: Optional[List[MicroRotorParticle]] = None,
        dimensions: int = 2,
        bounds: float = 10.0,
        phase_coupling_strength: float = 2.0,
        interaction_radius: float = 3.0,
        dt: float = 0.05,
    ):
        self.dimensions = dimensions
        self.bounds = bounds
        self.coupling_k = phase_coupling_strength
        self.interaction_radius = interaction_radius
        self.dt = dt
        self.step_count = 0
        self.time = 0.0

        self.scrubber = CausalTimelineScrubber()

        if particles is not None:
            self.particles = particles
        else:
            self.particles = []

    def add_particle(self, particle: MicroRotorParticle) -> None:
        self.particles.append(particle)

    def get_macro_order_parameter(self) -> float:
        """
        Calculate Kuramoto order parameter R = | (1/N) * sum_k e^{i * theta_k} |.
        R ~ 1 indicates coherent phase locking; R ~ 0 indicates incoherent phase distribution.
        """
        if not self.particles:
            return 0.0
        phases = np.array([p.theta for p in self.particles])
        complex_order = np.mean(np.exp(1j * phases))
        return float(np.abs(complex_order))

    def get_total_energy(self) -> float:
        """
        Calculate total kinetic + rotational energy in the sandbox.
        """
        ke = sum(0.5 * p.mass * float(np.sum(p.velocity**2)) for p in self.particles)
        re = sum(0.5 * p.mass * (p.omega**2) for p in self.particles)
        return ke + re

    def compute_forces_and_coupling(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute spatial forces (attraction/repulsion) and phase-coupling torques.
        Returns (force_vectors, phase_torques).
        """
        N = len(self.particles)
        forces = np.zeros((N, self.dimensions), dtype=float)
        torques = np.zeros(N, dtype=float)

        if N == 0:
            return forces, torques

        positions = np.array([p.position for p in self.particles])
        phases = np.array([p.theta for p in self.particles])
        types = np.array([p.type_id for p in self.particles])

        for i in range(N):
            p_i = self.particles[i]
            diff = positions - p_i.position  # (N, D)
            dist = np.linalg.norm(diff, axis=1) # (N,)

            # Exclude self
            mask = (dist > 1e-6) & (dist < self.interaction_radius)
            if not np.any(mask):
                continue

            for j in np.where(mask)[0]:
                r = dist[j]
                r_hat = diff[j] / r
                p_j = self.particles[j]

                # Type-based interaction multiplier
                type_factor = 1.0 if types[i] == types[j] else -0.5

                # Phase difference effect on force: co-phased particles attract, anti-phased repel
                phase_cos = np.cos(phases[j] - phases[i])
                spatial_mag = (1.0 / (r**2 + 0.1) - 0.5 / (r + 0.1)) * type_factor * (0.5 + 0.5 * phase_cos)
                forces[i] += spatial_mag * r_hat

                # Kuramoto phase-locking torque attenuated by distance
                spatial_weight = np.exp(-r / self.interaction_radius)
                phase_sin = np.sin(phases[j] - phases[i])
                torques[i] += (self.coupling_k / p_i.mass) * spatial_weight * phase_sin

        return forces, torques

    def step(self) -> Dict[str, Any]:
        """
        Advance physical and phase state by dt.
        """
        N = len(self.particles)
        if N == 0:
            self.step_count += 1
            self.time += self.dt
            return {"macro_order": 0.0, "total_energy": 0.0}

        forces, torques = self.compute_forces_and_coupling()

        for i, p in enumerate(self.particles):
            # Update physical position & velocity with somatic friction
            acceleration = (forces[i] - p.friction * p.velocity) / p.mass
            p.velocity += acceleration * self.dt
            p.position += p.velocity * self.dt

            # Periodic boundary wrapping
            p.position = np.mod(p.position + self.bounds, 2 * self.bounds) - self.bounds

            # Update phase angle and angular velocity with phase torque
            angular_acc = (torques[i] - p.friction * (p.omega - 1.0)) / p.mass
            p.omega += angular_acc * self.dt
            p.theta = (p.theta + p.omega * self.dt) % (2 * np.pi)

            # Measure neighbor phase locking
            # Lock degree increases when torques stabilize
            p.phase_lock_degree = float(np.clip(1.0 - np.abs(torques[i]), 0.0, 1.0))

            # Update somatosensory profile with physical friction & kinetic resistance
            kinetic_resistance = 0.5 * p.mass * float(np.linalg.norm(p.velocity))
            p.sensory_profile["pressure"] = float(p.mass * 9.81 + kinetic_resistance)
            p.sensory_profile["texture"] = float(p.friction + 0.1 * np.abs(p.omega))

        self.step_count += 1
        self.time += self.dt

        macro_order = self.get_macro_order_parameter()
        total_energy = self.get_total_energy()

        # Record to timeline scrubber
        self.scrubber.record_frame(
            step_idx=self.step_count,
            timestamp=self.time,
            particles=self.particles,
            macro_order=macro_order,
            total_energy=total_energy,
        )

        return {
            "step_idx": self.step_count,
            "timestamp": self.time,
            "macro_order": macro_order,
            "total_energy": total_energy,
        }
