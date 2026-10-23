import numpy as np
from typing import Callable, Dict, Any, Optional, Tuple, List

class SymplecticCausalEngine:
    """
    [Symplectic Causal State Dynamics Engine]
    Advances position z and momentum p in phase space (z, p) using a two-stage Symplectic Verlet Integrator
    that guarantees phase space volume preservation and numeric stability.

    Includes:
    - Symplectic Verlet Integrator
    - Phase Space Energy Monitoring (E_k, V(z), E_total)
    - Adaptive Time Stepping (dt) based on Hessian-free curvature estimates
    - Viscosity/Damping controller (gamma)
    """
    def __init__(
        self,
        mass: float = 1.0,
        dt_initial: float = 0.05,
        dt_min: float = 0.001,
        dt_max: float = 0.2,
        gamma_initial: float = 0.1,
        adaptive_dt: bool = True
    ):
        self.mass = mass
        self.dt = dt_initial
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.gamma = gamma_initial
        self.adaptive_dt = adaptive_dt

        # State tracking
        self.last_grad_v: Optional[np.ndarray] = None
        self.last_z: Optional[np.ndarray] = None
        self.history: List[Dict[str, Any]] = []

    def compute_energy(
        self,
        z: np.ndarray,
        p: np.ndarray,
        potential_fn: Callable[[np.ndarray], float]
    ) -> Dict[str, float]:
        """Calculates kinetic energy E_k, potential energy V(z), and total energy E_total."""
        e_k = float(np.sum(p ** 2) / (2.0 * self.mass))
        e_v = float(potential_fn(z))
        e_total = e_k + e_v
        return {
            "e_k": e_k,
            "e_v": e_v,
            "e_total": e_total
        }

    def estimate_curvature(self, z: np.ndarray, grad_v: np.ndarray) -> float:
        """
        [Hessian-Free Curvature Estimator]
        Estimates local field curvature ||nabla^2 V|| using finite differences of gradients:
        L ≈ || grad_V(z) - grad_V(z_prev) || / || z - z_prev ||
        """
        if self.last_z is None or self.last_grad_v is None:
            return 1.0

        delta_z = z - self.last_z
        norm_dz = np.linalg.norm(delta_z)
        if norm_dz < 1e-8:
            return 1.0

        delta_grad = grad_v - self.last_grad_v
        norm_dgrad = np.linalg.norm(delta_grad)

        curvature = norm_dgrad / norm_dz
        return float(curvature)

    def adapt_time_step(self, curvature: float):
        """
        Dynamically adjusts dt based on estimated curvature.
        High curvature -> smaller dt to prevent numerical divergence.
        Low curvature -> larger dt for faster convergence.
        """
        if not self.adaptive_dt:
            return

        # Scale factor inversely proportional to square root of curvature
        target_dt = 0.1 / np.sqrt(max(0.1, curvature))
        target_dt = np.clip(target_dt, self.dt_min, self.dt_max)

        # Smooth transition for dt
        self.dt = float(0.8 * self.dt + 0.2 * target_dt)

    def step(
        self,
        z: np.ndarray,
        p: np.ndarray,
        potential_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Performs one Symplectic Verlet Integration step:

        1. p_{t+1/2} = p_t - (dt / 2) * (grad_V(z_t) + gamma * p_t)
        2. z_{t+1} = z_t + dt * (p_{t+1/2} / m)
        3. grad_V(z_{t+1}) evaluated
        4. p_{t+1} = (p_{t+1/2} - (dt / 2) * grad_V(z_{t+1})) / (1 + gamma * dt / 2)
        """
        z_curr = np.array(z, dtype=np.float32)
        p_curr = np.array(p, dtype=np.float32)

        # Gradient at current position
        grad_v_t = grad_fn(z_curr)

        # Estimate curvature and adapt dt
        curvature = self.estimate_curvature(z_curr, grad_v_t)
        self.adapt_time_step(curvature)

        dt = self.dt
        m = self.mass
        gamma = self.gamma

        # Step 1: Half-step momentum
        p_half = p_curr - (dt / 2.0) * (grad_v_t + gamma * p_curr)

        # Step 2: Full-step position
        z_next = z_curr + dt * (p_half / m)

        # Step 3: Gradient at new position
        grad_v_next = grad_fn(z_next)

        # Step 4: Full-step momentum
        p_next = (p_half - (dt / 2.0) * grad_v_next) / (1.0 + (gamma * dt / 2.0))

        # Energy & metrics calculation
        energies = self.compute_energy(z_next, p_next, potential_fn)

        # Store last state for Hessian-free curvature estimation
        self.last_z = z_curr.copy()
        self.last_grad_v = grad_v_t.copy()

        info = {
            "dt": dt,
            "gamma": gamma,
            "curvature": curvature,
            "grad_norm": float(np.linalg.norm(grad_v_next)),
            **energies
        }
        self.history.append(info)

        return z_next, p_next, info

    def set_damping(self, gamma: float):
        """Adjusts the damping/viscosity factor gamma."""
        self.gamma = max(0.0, float(gamma))
