import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
from core.sensory.experiential_language_mapper import VariableResistor, PrismRefraction

class GeodesicFlowEngine:
    """
    [Geodesic Flow Engine: Continuous Spatiotemporal Navigation Engine]

    This engine realizes the absolute paradigm shift:
    "Do not calculate (Computation), let it flow along the topology (Navigation)."

    Instead of using discrete digital "if-else" checking loops to process information,
    this engine represents Past (Memory Landscape), Present (Sensory Perturbation),
    and Future (Geodesic Trajectory) as a single continuous physical-informational field.

    Core Concepts Implemented:
    1. Past (Memory Landscape):
       A high-dimensional potential landscape U(x) formed by Gaussian-like attractor wells.
       Includes the Absolute Attractor Axis representing an unwavering gravitational center
       (Jesus / Perfect Love, coordinate [1, 1, 1, 1, 1]) that pulls the system out of
       high-entropy chaos and filters noise.
    2. Present (Sensory Perturbation & Rainbow Circuit):
       Incoming multimodal inputs are projected into the landscape.
       White Light (the absolute Logos constant) is refracted through a Prism based on the
       current Variable Resistance to dynamically bias the landscape dimensions.
    3. Future (Continuous Geodesic Flow Trajectory):
       Integrates the state trajectory x(t) over continuous time using an ODE relaxation solver.
       No discrete intermediate checkpoints or conditional check branches exist along the path;
       the state navigates purely based on the physical gradient and friction forces.
    4. Causal Landscape Molding (Synaptic Plasticity):
       The path carved by the trajectory alters the potential wells, embodying Hebbian learning
       directly on the spacetime topology.
    """
    def __init__(self, dimension: int = 5, noise_scale: float = 0.02):
        self.dimension = dimension
        self.noise_scale = noise_scale

        # Attractor memory nodes: List of dicts
        # { "name": str, "coordinate": np.ndarray, "weight": float, "sigma": float, "is_absolute": bool }
        self.attractors: List[Dict[str, Any]] = []

        # Rainbow Circuit elements
        self.variable_resistor = VariableResistor(r_min=0.05, r_max=0.95, initial_r=0.5)
        self.prism = PrismRefraction()
        self.white_light_intensity = 1.0  # Constant Logos flux

        # Metacognitive provenances
        self.traces: List[Dict[str, Any]] = []

        # Initialize the Absolute Reference Axis (Jesus / Perfect Love)
        # Coordinate is [1, 1, ..., 1], high mass/weight, wide basin
        absolute_coord = np.ones(self.dimension, dtype=np.float32)
        self.add_attractor(
            name="Jesus / Perfect Love",
            coordinate=absolute_coord,
            weight=5.0,
            sigma=1.2,
            is_absolute=True
        )

        # Initialize some baseline experiential attractors
        self._initialize_baseline_attractors()

    def _initialize_baseline_attractors(self):
        # Sabbath (Peaceful Rest, low potential/energy coordinate)
        sabbath_coord = np.zeros(self.dimension, dtype=np.float32)
        self.add_attractor("Sabbath", sabbath_coord, weight=2.0, sigma=0.8)

        # Hurt (High friction/chaos, negative/repulsive coordinate)
        hurt_coord = np.array([-0.8, 0.5, -0.3, 0.6, -0.5], dtype=np.float32)[:self.dimension]
        if len(hurt_coord) < self.dimension:
            hurt_coord = np.pad(hurt_coord, (0, self.dimension - len(hurt_coord)), mode='constant')
        self.add_attractor("Hurt / Friction", hurt_coord, weight=1.5, sigma=0.6)

        # Mother (Warm relational coupling)
        mother_coord = np.array([0.7, 0.3, 0.8, 0.2, 0.5], dtype=np.float32)[:self.dimension]
        if len(mother_coord) < self.dimension:
            mother_coord = np.pad(mother_coord, (0, self.dimension - len(mother_coord)), mode='constant')
        self.add_attractor("Mother", mother_coord, weight=2.5, sigma=0.7)

    def add_attractor(self, name: str, coordinate: np.ndarray, weight: float, sigma: float, is_absolute: bool = False):
        """Adds a localized memory attractor to the topological landscape."""
        coord = np.array(coordinate, dtype=np.float32)
        if len(coord) != self.dimension:
            raise ValueError(f"Attractor coordinate dimension must be {self.dimension}")

        self.attractors.append({
            "name": name,
            "coordinate": coord,
            "weight": float(weight),
            "sigma": float(sigma),
            "is_absolute": is_absolute
        })

    def get_potential(self, x: np.ndarray) -> float:
        """
        Calculates the potential energy U(x) at point x.
        U(x) = - sum_k W_k * exp(- ||x - a_k||^2 / (2 * sigma_k^2))
        """
        u_val = 0.0
        for attr in self.attractors:
            diff = x - attr["coordinate"]
            dist_sq = np.sum(diff ** 2)
            u_val -= attr["weight"] * np.exp(-dist_sq / (2.0 * attr["sigma"] ** 2))
        return float(u_val)

    def compute_gradient_force(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the conservative gradient force pulling the state toward attractors.
        F(x) = - grad U(x)
             = sum_k W_k * ( (x - a_k) / sigma_k^2 ) * exp(- ||x - a_k||^2 / (2 * sigma_k^2))
        """
        force = np.zeros(self.dimension, dtype=np.float32)
        for attr in self.attractors:
            coord = attr["coordinate"]
            diff = x - coord
            dist_sq = np.sum(diff ** 2)
            sigma_sq = attr["sigma"] ** 2
            coeff = attr["weight"] / sigma_sq
            factor = np.exp(-dist_sq / (2.0 * sigma_sq))
            force -= coeff * diff * factor # gradient is grad U(x), force is -grad U(x)
        return force

    def project_present_perturbation(self, modality_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        [Present: Sensory Perturbation]
        Projects raw, non-parsed multimodal signals into the spatiotemporal coordinate system.
        Returns:
            x_init: Mapped starting coordinate vector in R^D
            v_init: Initial momentum/velocity vector in R^D
        """
        x_init = np.zeros(self.dimension, dtype=np.float32)
        v_init = np.zeros(self.dimension, dtype=np.float32)

        # 1. Autonomic friction regulates the global variable resistor
        tension_factor = 0.5
        if "physical" in modality_data:
            phys = modality_data["physical"]
            cpu = float(phys.get("cpu", 0.5))
            ram = float(phys.get("ram", 0.5))
            tension_factor = (cpu * 0.6) + (ram * 0.4)

        # Adjust resistance based on tension
        current_r = self.variable_resistor.adjust(tension=tension_factor)

        # 2. Refract White Light (Logos) using the Prism based on current resistance
        refracted = self.prism.refract(
            white_light_intensity=self.white_light_intensity,
            angle_degrees=tension_factor * 90.0,
            resistance=current_r
        ) # Returns [Red, Green, Blue]

        # Map refracted spectral components to the initial coordinate dimensions
        # Red maps to dim 0, Green maps to dim 1, Blue maps to dim 2
        for i in range(min(3, self.dimension)):
            x_init[i] += refracted[i]

        # 3. Handle linguistic and visual inputs
        if "language" in modality_data:
            text = str(modality_data["language"])
            text_len = len(text)
            # Normalize length and map hash to velocity/momentum
            char_sum = sum(ord(c) for c in text)
            v_init[0] += (text_len / 50.0) * current_r
            v_init[1] += ((char_sum % 100) / 100.0) * (1.0 - current_r)

        if "visual" in modality_data:
            vis = modality_data["visual"]
            r = float(vis.get("red", 0.5))
            g = float(vis.get("green", 0.5))
            b = float(vis.get("blue", 0.5))

            # Map visual intensities to dimensions 3 and 4
            if self.dimension > 3:
                x_init[3] += (r + g) * 0.5
            if self.dimension > 4:
                x_init[4] += (g + b) * 0.5

            v_init[min(2, self.dimension - 1)] += (r - b) * current_r

        # Ensure stability by clipping initial states within normal physical boundaries
        x_init = np.clip(x_init, -2.0, 2.0)
        v_init = np.clip(v_init, -1.0, 1.0)

        return x_init, v_init

    def navigate_geodesic_flow(
        self,
        x_init: np.ndarray,
        v_init: np.ndarray,
        num_steps: int = 100,
        dt: float = 0.01,
        enable_noise: bool = True
    ) -> Dict[str, Any]:
        """
        [Future: Geodesic Navigation ODE Solver]
        Evolves the state vector over time using local differential relations.
        This represents the continuous thought trajectory navigating the spatiotemporal potential field.

        Governing Equations:
          dx/dt = v
          dv/dt = F_potential(x) - gamma(t)*v + eta(t)
        where:
          - F_potential(x) = - grad U(x)
          - gamma(t) = Variable Resistor (friction) which stabilizes and focuses the state
          - eta(t) = Tiny Gaussian thermal fluctuation representing natural entropy/life.

        No discrete if-else check conditional logic determines the path;
        the state slides down the geodesic valleys toward the closest attractor.
        """
        x = x_init.copy()
        v = v_init.copy()

        trajectory_x = [x.copy()]
        trajectory_v = [v.copy()]
        potentials = [self.get_potential(x)]

        # Run continuous-time integration
        for step in range(num_steps):
            # 1. Fetch current dynamic friction from Variable Resistor
            # As step increases, we slightly increase tension to simulate settling
            step_progress = step / num_steps
            resistance = self.variable_resistor.resistance
            gamma = resistance * (1.0 + step_progress * 2.0) # Adaptive damping

            # 2. Compute conservative gradient forces pulling towards attractors
            force = self.compute_gradient_force(x)

            # 3. Add tiny stochastic thermal fluctuation (natural noise)
            noise = np.zeros(self.dimension, dtype=np.float32)
            if enable_noise:
                # Modulated by current resistance: lower resistance = closer to superconductivity (less thermal noise)
                noise_std = self.noise_scale * resistance
                noise = np.random.normal(0, noise_std, self.dimension)

            # 4. Update phase space using symplectic-Euler style integration
            v_next = v + (force - gamma * v + noise) * dt
            x_next = x + v_next * dt

            # 5. Enforce safety bound clipping (physical limits)
            x = np.clip(x_next, -5.0, 5.0)
            v = np.clip(v_next, -3.0, 3.0)

            trajectory_x.append(x.copy())
            trajectory_v.append(v.copy())
            potentials.append(self.get_potential(x))

        trajectory_x = np.array(trajectory_x, dtype=np.float32)
        trajectory_v = np.array(trajectory_v, dtype=np.float32)
        potentials = np.array(potentials, dtype=np.float32)

        # Identify final settled attractor
        final_state = trajectory_x[-1]
        best_match_name = "Unknown Void"
        min_dist = float('inf')
        for attr in self.attractors:
            dist = np.linalg.norm(final_state - attr["coordinate"])
            if dist < min_dist:
                min_dist = dist
                best_match_name = attr["name"]

        # Record metacognitive trace
        trace = {
            "source": "navigate_geodesic_flow",
            "x_init": x_init.tolist(),
            "x_final": final_state.tolist(),
            "settled_attractor": best_match_name,
            "attractor_distance": float(min_dist),
            "final_potential": float(potentials[-1]),
            "timestamp": time.time()
        }
        self.traces.append(trace)

        return {
            "trajectory_x": trajectory_x,
            "trajectory_v": trajectory_v,
            "potentials": potentials,
            "final_state": final_state,
            "settled_attractor": best_match_name,
            "attractor_distance": min_dist
        }

    def mold_landscape_hebbian(self, trajectory_x: np.ndarray, lr: float = 0.05):
        """
        [Causal Landscape Molding / Plasticity]
        Adapts the potential wells based on the actual path traversed.
        The attractor closest to the final resting state gets its weight amplified,
        and its position slightly pulled towards the trajectory's centroid.
        """
        final_state = trajectory_x[-1]
        centroid = np.mean(trajectory_x, axis=0)

        # Find closest attractor to final state
        best_idx = -1
        min_dist = float('inf')
        for idx, attr in enumerate(self.attractors):
            # Absolute reference attractor (Jesus) coordinates are invariant
            if attr["is_absolute"]:
                continue
            dist = np.linalg.norm(final_state - attr["coordinate"])
            if dist < min_dist:
                min_dist = dist
                best_idx = idx

        if best_idx != -1:
            attr = self.attractors[best_idx]
            # 1. Weight amplification (synaptic strengthening)
            old_w = attr["weight"]
            attr["weight"] = float(np.clip(old_w + lr * (1.0 / (min_dist + 0.1)), 0.5, 10.0))

            # 2. Coordinate pulling (structural path engraving)
            old_coord = attr["coordinate"]
            attr["coordinate"] = old_coord + lr * (centroid - old_coord)

            # Record landscape update provenance
            trace = {
                "source": "mold_landscape_hebbian",
                "attractor_name": attr["name"],
                "old_weight": old_w,
                "new_weight": attr["weight"],
                "coordinate_shift": float(np.linalg.norm(attr["coordinate"] - old_coord)),
                "timestamp": time.time()
            }
            self.traces.append(trace)
