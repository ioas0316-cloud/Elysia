"""
Unified Topological Phase Runtime (통합 위상 런타임)
===================================================
A macro-topological execution environment embodying:
1. Self-World Boundary (Autopoietic Operational Closure) & Enactive Reach Grid
2. 3x3x3 Fractal Node Network Phase-Locking & Isomorphic Resonance Coupling
3. Chromatic Field Dynamics (Flux/Order/Entropy)
4. Meta-Causal Collapse Prevention (Phase Tension & Causal Gravity)
5. Causal Backtracing & Achromatic White Tensor Zero-Point Re-initialization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from L1_causal_field.core.physics.prismatic_phase_transition_engine import (
    PrismaticPhaseTransitionEngine,
    MemoryPhaseMode
)


class SelfWorldBoundary:
    """
    Self-World Boundary (자아와 세계의 경계)
    Enforces autopoietic operational closure and filters external world perturbations
    based on internal phase tension and systemic stability.
    """
    def __init__(self, boundary_rigidity: float = 0.8, internal_homeostasis: float = 1.0):
        self.boundary_rigidity = float(boundary_rigidity)
        self.internal_homeostasis = float(internal_homeostasis)
        self.accumulated_perturbation = 0.0
        self.boundary_tension = 1.0

    def process_perturbation(self, external_signal: np.ndarray, current_phase_tension: float) -> Tuple[np.ndarray, float]:
        """
        Differentiates Self from World by filtering external signals through boundary rigidity
        and current internal phase tension.
        """
        signal_norm = float(np.linalg.norm(external_signal))
        # Compute attenuation factor based on operational closure
        attenuation = np.clip(1.0 - (self.boundary_rigidity * (1.0 - current_phase_tension)), 0.1, 1.0)
        filtered_signal = external_signal * attenuation

        # Update internal perturbation tracking
        self.accumulated_perturbation = 0.9 * self.accumulated_perturbation + 0.1 * signal_norm
        self.boundary_tension = np.clip(self.internal_homeostasis / (1.0 + 0.2 * self.accumulated_perturbation), 0.1, 1.0)

        return filtered_signal, self.boundary_tension


class EnactiveReachGrid:
    """
    Enactive Reach Grid (능동적 개입의 경계 및 격자)
    Defines the active physical/cognitive boundary of intervention ("hands and feet", units, physical laws).
    """
    def __init__(self, reach_radius: float = 3.0, unit_resolution: int = 3):
        self.reach_radius = float(reach_radius)
        self.unit_resolution = int(unit_resolution)
        self.active_channels = np.ones((unit_resolution, unit_resolution, unit_resolution), dtype=float)

    def evaluate_intervention(self, target_coords: np.ndarray, action_magnitude: float) -> float:
        """
        Evaluates whether an action falls within the active reach grid and calculates effective influence.
        """
        distance = float(np.linalg.norm(target_coords - np.array([1.0, 1.0, 1.0])))
        if distance > self.reach_radius:
            influence = np.exp(-(distance - self.reach_radius)) * action_magnitude
        else:
            influence = action_magnitude
        return float(influence)


class FractalPhaseNode3x3x3:
    """
    3x3x3 Fractal Phase Node
    A single node within the 27-node 3x3x3 fractal cube.
    """
    def __init__(self, coords: Tuple[int, int, int]):
        self.coords = coords
        self.is_center = (coords == (1, 1, 1))

        # 5D Phase Vector [Will, Flex, Order, Origin, Field]
        self.phase_vec = np.random.uniform(-0.1, 0.1, 5)
        if self.is_center:
            self.phase_vec = np.zeros(5)  # Achromatic zero-point center

        self.frequency = 1.0 + 0.1 * (np.sum(coords) - 3)
        self.phase_angle = 0.0

        # Chromatic Signature [Red (Flux), Blue (Order), Yellow (Entropy)]
        self.chromatic = np.array([0.33, 0.33, 0.34], dtype=float)
        self.phase_tension = 1.0

    def update_phase(self, coupling_input: np.ndarray, dt: float = 0.1) -> float:
        """
        Updates internal phase vector and angle based on coupling inputs.
        """
        # Phase rotor dynamics
        d_phase = coupling_input[:5] if len(coupling_input) >= 5 else np.pad(coupling_input, (0, 5 - len(coupling_input)))
        self.phase_vec = 0.95 * self.phase_vec + 0.05 * d_phase

        # Angle rotation
        self.phase_angle = (self.phase_angle + self.frequency * dt + 0.1 * np.mean(self.phase_vec)) % (2 * np.pi)

        # Chromatic adjustment: Order rises with phase stability, Entropy with variance
        var_vec = float(np.var(self.phase_vec))
        self.chromatic[0] = 0.4 * np.abs(self.phase_vec[0]) + 0.2  # Flux
        self.chromatic[1] = np.exp(-var_vec) * 0.6 + 0.2          # Order
        self.chromatic[2] = np.clip(var_vec * 2.0, 0.05, 0.6)     # Entropy
        self.chromatic /= np.sum(self.chromatic)

        self.phase_tension = float(self.chromatic[1] / (self.chromatic[2] + 1e-6))
        self.phase_tension = np.clip(self.phase_tension, 0.01, 10.0)
        return self.phase_tension


class UnifiedTopologicalPhaseRuntime:
    """
    Unified Topological Phase Runtime (통합 위상 런타임)
    The macro-runtime execution environment that orchestrates continuous causal dynamics
    without low-dimensional collapse into discrete scalar logic.
    """
    def __init__(self, zeitgeist_vector: Optional[np.ndarray] = None):
        if zeitgeist_vector is None:
            self.zeitgeist_vector = np.array([0.2, 0.1, 0.7, 0.8, 0.3], dtype=float)
        else:
            self.zeitgeist_vector = zeitgeist_vector.astype(float)

        self.boundary = SelfWorldBoundary()
        self.reach_grid = EnactiveReachGrid()

        # Initialize 3x3x3 Fractal Node Grid
        self.nodes: Dict[Tuple[int, int, int], FractalPhaseNode3x3x3] = {}
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    self.nodes[(x, y, z)] = FractalPhaseNode3x3x3((x, y, z))

        self.causal_gravity_k = 0.8
        self.coherence_score = 1.0
        self.history_ring: List[Dict[str, Any]] = []
        self.backtrace_count = 0
        self.white_tensor_reset_count = 0

        # Phase-Change Computing Engine
        self.phase_change_engine = PrismaticPhaseTransitionEngine(grid_shape=(3, 3))
        self.teleological_purpose = "optimization"

    def set_teleological_purpose(self, purpose: str):
        """
        Sets the top-level teleological purpose (e.g., 'optimization', 'diversity', 'complexity')
        which refractorily realigns the static rotor prism observation lens and memory dynamics.
        """
        self.teleological_purpose = purpose

    def transition_memory_state(self, mode: str) -> Dict[str, Any]:
        """
        Transitions system memory state between Ice (past/facts), Water (present thought),
        and Gas (future simulation) without low-level disk swapping.
        """
        mem_map = self.phase_change_engine.transition_memory_phase(
            target_mode=mode,
            teleological_purpose=self.teleological_purpose
        )
        return {
            "active_phase": mem_map.active_phase,
            "total_memory_bytes": mem_map.total_memory_bytes,
            "ice_norm": float(np.linalg.norm(mem_map.ice_lattice)),
            "water_norm": float(np.linalg.norm(mem_map.water_flow)),
            "gas_norm": float(np.linalg.norm(mem_map.gas_cloud))
        }

    def trigger_wave_propagation(self, external_trigger: np.ndarray, dt: float = 0.1) -> Dict[str, Any]:
        """
        Processes an external trigger through the Self-World Boundary, propagating
        spherical phase tension from the Center Node [1,1,1] across all 27 fractal nodes.
        """
        # 1. Self-World Boundary Filtering
        avg_tension = np.mean([n.phase_tension for n in self.nodes.values()])
        filtered_trigger, boundary_tension = self.boundary.process_perturbation(external_trigger, avg_tension)

        # 2. Evaluate Reach Grid Influence
        influence = self.reach_grid.evaluate_intervention(np.array([1.0, 1.0, 1.0]), float(np.linalg.norm(filtered_trigger)))

        # 3. Spherical Wave Propagation from Center [1,1,1]
        center_node = self.nodes[(1, 1, 1)]
        center_node.update_phase(filtered_trigger * influence, dt=dt)

        # Propagate radially based on distance to center [1,1,1]
        node_angles = []
        tensions = []
        for coords, node in self.nodes.items():
            if coords == (1, 1, 1):
                continue
            dist = float(np.linalg.norm(np.array(coords) - np.array([1, 1, 1])))
            coupling = center_node.phase_vec * np.exp(-dist) + 0.1 * self.zeitgeist_vector
            t = node.update_phase(coupling, dt=dt)
            node_angles.append(node.phase_angle)
            tensions.append(t)

        # 4. Kuramoto Order Parameter / Phase-Locking Coherence calculation
        angles_arr = np.array(node_angles)
        self.coherence_score = float(np.abs(np.mean(np.exp(1j * angles_arr))))

        # Isomorphic resonance score comparing internal wavefield with trigger structure
        trigger_padded = np.pad(filtered_trigger, (0, max(0, 5 - len(filtered_trigger))))[:5]
        internal_field = np.mean([n.phase_vec for n in self.nodes.values()], axis=0)
        norm_t = np.linalg.norm(trigger_padded)
        norm_i = np.linalg.norm(internal_field)
        if norm_t > 1e-6 and norm_i > 1e-6:
            isomorphism_score = float(np.dot(trigger_padded, internal_field) / (norm_t * norm_i))
        else:
            isomorphism_score = 1.0

        # 5. Meta-Causal Collapse Prevention
        mean_tension = float(np.mean(tensions))
        collapse_warning = False
        if mean_tension < 0.2 or self.coherence_score < 0.1:
            collapse_warning = True
            self._apply_causal_gravity()
            self._backtrace_and_correct()

        # 6. Step Phase-Change Computing Engine & Static Rotor Prism Refraction
        phase_engine_state = self.phase_change_engine.step(
            numerical_input=filtered_trigger,
            dt=dt,
            teleological_mode=self.teleological_purpose
        )

        # 7. Record State in History Ring
        step_record = {
            "boundary_tension": boundary_tension,
            "coherence_score": self.coherence_score,
            "isomorphism_score": isomorphism_score,
            "mean_phase_tension": mean_tension,
            "collapse_warning": collapse_warning,
            "center_phase": center_node.phase_vec.copy(),
            "teleological_purpose": self.teleological_purpose,
            "memory_active_phase": phase_engine_state.memory_map.active_phase if phase_engine_state.memory_map else "WATER",
            "memory_bytes": phase_engine_state.memory_map.total_memory_bytes if phase_engine_state.memory_map else 0
        }
        self.history_ring.append(step_record)
        if len(self.history_ring) > 100:
            self.history_ring.pop(0)

        return step_record

    def _apply_causal_gravity(self):
        """
        Applies Causal Gravity spacetime curvature to bend noisy/degraded phase trajectories back into orbital alignment.
        """
        for node in self.nodes.values():
            node.phase_vec *= self.causal_gravity_k
            node.chromatic[1] += 0.1  # Restore Order
            node.chromatic /= np.sum(node.chromatic)

    def _backtrace_and_correct(self):
        """
        Backtraces through the history ring to rectify perturbation origin and restore stability.
        """
        self.backtrace_count += 1
        if len(self.history_ring) >= 2:
            # Restore center node phase to previous stable ring entry
            stable_entry = self.history_ring[-2]
            self.nodes[(1, 1, 1)].phase_vec = stable_entry["center_phase"].copy()

    def reset_achromatic_white_tensor(self):
        """
        Resets all fractal nodes to Achromatic White Tensor Zero-Point Superposition (unbiased, unconstrained state).
        """
        self.white_tensor_reset_count += 1
        for coords, node in self.nodes.items():
            node.phase_vec = np.zeros(5)
            node.phase_angle = 0.0
            node.chromatic = np.array([0.33, 0.33, 0.34], dtype=float)
            node.phase_tension = 1.0
        self.boundary.accumulated_perturbation = 0.0
        self.boundary.boundary_tension = 1.0

    def get_system_state_summary(self) -> Dict[str, Any]:
        """
        Returns a comprehensive summary of the runtime state.
        """
        chromatic_sums = np.mean([n.chromatic for n in self.nodes.values()], axis=0)
        return {
            "zeitgeist_vector": self.zeitgeist_vector.tolist(),
            "boundary_tension": self.boundary.boundary_tension,
            "coherence_score": self.coherence_score,
            "mean_chromatic_flux": float(chromatic_sums[0]),
            "mean_chromatic_order": float(chromatic_sums[1]),
            "mean_chromatic_entropy": float(chromatic_sums[2]),
            "backtrace_count": self.backtrace_count,
            "white_tensor_reset_count": self.white_tensor_reset_count,
            "history_length": len(self.history_ring),
        }
