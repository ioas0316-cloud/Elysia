import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class AttractorWell:
    """Represents an attractor well (-V) or repulsion barrier (+V) in potential space."""
    name: str
    position: np.ndarray  # Coordinate in vector space
    depth: float         # Positive for attractor (pull), negative for barrier (push)
    width: float         # Radial dispersion width


class CausalDoOperator:
    """
    Subsystem for variable isolation and intentional intervention (do-operator).
    Slices high-dimensional potential fields along specific variable axes,
    clamping chosen variables while fixing orthogonal dimensions to observe
    pure causal curvature changes on target axes.
    """

    def __init__(self, vector_dim: int = 8):
        self.vector_dim = vector_dim

    def slice_and_clamp(
        self,
        potential_field_func,
        clamped_axis: int,
        clamped_val: float,
        state_point: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Executes do(X_i = clamped_val) on state_point, freezing orthogonal axes.
        Returns:
            - Intervened state vector
            - Pure potential value at the intervened state
        """
        intervened_state = state_point.copy()
        intervened_state[clamped_axis] = clamped_val
        val = potential_field_func(intervened_state)
        return intervened_state, val

    def observe_pure_causal_curvature(
        self,
        potential_field_func,
        clamped_axis: int,
        target_axis: int,
        state_point: np.ndarray,
        range_vals: np.ndarray
    ) -> np.ndarray:
        """
        Measures the pure causal curvature along target_axis under do(X_clamped = val).
        Returns potential response profile along target_axis as target_axis values vary across range_vals.
        """
        responses = []
        for v in range_vals:
            test_state = state_point.copy()
            test_state[target_axis] = v
            responses.append(potential_field_func(test_state))
        return np.array(responses)


class CounterfactualMetaField:
    """
    Parallel virtual space simulation for retrospective reasoning.
    Clones past state T_past, runs counterfactual intervention do(A'),
    calculates metacognitive torque from actual vs. meta phase discrepancy,
    and executes plastic deformation on the top-down intent field.
    """

    def __init__(self, vector_dim: int = 8):
        self.vector_dim = vector_dim
        self.past_states: List[Dict[str, Any]] = []
        self.metacognitive_torque: float = 0.0

    def record_past_snapshot(self, timestamp: float, state_point: np.ndarray, attractors: List[AttractorWell], friction: float):
        """Clones past state T_past for retrospective analysis."""
        self.past_states.append({
            "timestamp": timestamp,
            "state_point": state_point.copy(),
            "attractors": [AttractorWell(a.name, a.position.copy(), a.depth, a.width) for a in attractors],
            "friction": friction
        })

    def run_counterfactual_simulation(
        self,
        snapshot_index: int,
        counterfactual_attractor: AttractorWell,
        steps: int = 15
    ) -> Tuple[np.ndarray, List[AttractorWell]]:
        """
        Runs counterfactual simulation from past snapshot with virtual intervention do(A').
        Returns simulated meta state point and resulting meta attractor topology.
        """
        if not self.past_states or snapshot_index >= len(self.past_states):
            raise ValueError("Invalid snapshot index for counterfactual simulation.")

        snapshot = self.past_states[snapshot_index]
        meta_state = snapshot["state_point"].copy()
        meta_attractors = [AttractorWell(a.name, a.position.copy(), a.depth, a.width) for a in snapshot["attractors"]]

        # Inject counterfactual intervention (e.g. activate console/switch, disable barrier)
        existing_names = [a.name for a in meta_attractors]
        if counterfactual_attractor.name in existing_names:
            idx = existing_names.index(counterfactual_attractor.name)
            meta_attractors[idx] = counterfactual_attractor
        else:
            meta_attractors.append(counterfactual_attractor)

        # Simulate trajectory under meta potential field
        for _ in range(steps):
            grad = self.calculate_gradient(meta_state, meta_attractors)
            meta_state -= 0.1 * grad

        return meta_state, meta_attractors

    def calculate_gradient(self, pos: np.ndarray, attractors: List[AttractorWell]) -> np.ndarray:
        """Calculates total potential gradient at pos for given attractors."""
        grad = np.zeros_like(pos)
        eps = 1e-5
        for i in range(len(pos)):
            pos_plus = pos.copy()
            pos_plus[i] += eps
            pos_minus = pos.copy()
            pos_minus[i] -= eps
            v_plus = sum(self._attractor_potential(pos_plus, a) for a in attractors)
            v_minus = sum(self._attractor_potential(pos_minus, a) for a in attractors)
            grad[i] = (v_plus - v_minus) / (2 * eps)
        return grad

    def _attractor_potential(self, pos: np.ndarray, attractor: AttractorWell) -> float:
        dist_sq = np.sum((pos - attractor.position) ** 2)
        # Depth > 0 means attraction (-V), depth < 0 means barrier (+V)
        return -attractor.depth * np.exp(-dist_sq / (2 * attractor.width ** 2))

    def compute_metacognitive_torque(
        self,
        actual_grad: np.ndarray,
        meta_grad: np.ndarray
    ) -> float:
        """
        Computes metacognitive torque τ_meta = || ∇V_actual × ∇V_meta || (or norm of outer gradient discrepancy).
        Quantifies the rotational/restorative tension between actual failed path and successful counterfactual path.
        """
        discrepancy = np.linalg.norm(actual_grad - meta_grad)
        if len(actual_grad) >= 3 and len(meta_grad) >= 3:
            cross_prod = np.cross(actual_grad[:3], meta_grad[:3])
            cross_norm = np.linalg.norm(cross_prod)
            if cross_norm > 0:
                discrepancy = cross_norm

        self.metacognitive_torque = float(discrepancy)
        return self.metacognitive_torque

    def apply_plastic_deformation(
        self,
        intent_attractors: List[AttractorWell],
        failed_attractor_name: str,
        successful_counterfactual_attractor: AttractorWell,
        torque_threshold: float = 0.001
    ) -> List[AttractorWell]:
        """
        Permanently reshapes the top-down intent field (Plastic Deformation) if metacognitive torque exceeds threshold.
        Erases/flattens failed paths and carves deep permanent attractors for counterfactual paths.
        """
        if self.metacognitive_torque < torque_threshold:
            return [AttractorWell(a.name, a.position.copy(), a.depth, a.width) for a in intent_attractors]  # Elastic region, no permanent deformation

        deformed_attractors = []
        for a in intent_attractors:
            if a.name == failed_attractor_name:
                # Flatten or turn failed path into a barrier
                deformed_attractors.append(AttractorWell(a.name, a.position.copy(), depth=-abs(a.depth), width=a.width))
            else:
                deformed_attractors.append(AttractorWell(a.name, a.position.copy(), a.depth, a.width))

        # Carve new permanent attractor well for counterfactual route
        # Avoid duplication if already present
        existing_names = [a.name for a in deformed_attractors]
        if successful_counterfactual_attractor.name in existing_names:
            idx = existing_names.index(successful_counterfactual_attractor.name)
            deformed_attractors[idx] = AttractorWell(
                successful_counterfactual_attractor.name,
                successful_counterfactual_attractor.position.copy(),
                successful_counterfactual_attractor.depth,
                successful_counterfactual_attractor.width
            )
        else:
            deformed_attractors.append(AttractorWell(
                successful_counterfactual_attractor.name,
                successful_counterfactual_attractor.position.copy(),
                successful_counterfactual_attractor.depth,
                successful_counterfactual_attractor.width
            ))

        return deformed_attractors


class TopologicalCausalEngine:
    """
    Topological Causal Engine.
    Executes reasoning and action synthesis entirely through continuous potential field dynamics
    (V_total = V_intent + V_sensor) and Principle of Least Action gradient flows (-∇V).
    Includes Standing Wave resonance detection, Dynamic Relaxation, CausalDoOperator, and CounterfactualMetaField.
    """

    def __init__(self, vector_dim: int = 8):
        self.vector_dim = vector_dim
        self.state_point = np.zeros(vector_dim)
        self.attractors: List[AttractorWell] = []
        self.do_operator = CausalDoOperator(vector_dim)
        self.meta_field = CounterfactualMetaField(vector_dim)

        self.standing_wave_active = False
        self.topological_friction = 0.0
        self.step_count = 0

    def add_attractor(self, attractor: AttractorWell):
        """Adds or updates an attractor well / barrier in the field."""
        existing = [a.name for a in self.attractors]
        if attractor.name in existing:
            idx = existing.index(attractor.name)
            self.attractors[idx] = attractor
        else:
            self.attractors.append(attractor)

    def compute_total_potential(self, pos: np.ndarray) -> float:
        """
        Evaluates V_total(pos) = sum of all potential contributions.
        Attractors lower potential (-V), barriers raise potential (+V).
        """
        total_v = 0.0
        for a in self.attractors:
            dist_sq = np.sum((pos - a.position) ** 2)
            # depth > 0 -> attraction (-V), depth < 0 -> barrier (+V)
            total_v += -a.depth * np.exp(-dist_sq / (2 * a.width ** 2))
        return total_v

    def compute_gradient(self, pos: np.ndarray) -> np.ndarray:
        """Computes gradient ∇V_total at pos using central differences."""
        grad = np.zeros_like(pos)
        eps = 1e-5
        for i in range(len(pos)):
            pos_p = pos.copy()
            pos_p[i] += eps
            pos_m = pos.copy()
            pos_m[i] -= eps
            grad[i] = (self.compute_total_potential(pos_p) - self.compute_total_potential(pos_m)) / (2 * eps)
        return grad

    def step(self, learning_rate: float = 0.1) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Executes one dynamic step along the gradient of least action (-∇V).
        Returns:
            - Updated state_point
            - Action vector (motion vector = -∇V)
            - Current topological friction
        """
        grad = self.compute_gradient(self.state_point)
        action_vector = -grad
        self.state_point += learning_rate * action_vector

        # Calculate topological friction (magnitude of gradient & obstacle proximity)
        self.topological_friction = float(np.linalg.norm(grad))

        # Check Standing Wave formation (resonance when state hits attractor basin center)
        self._update_standing_wave_status()

        self.step_count += 1
        return self.state_point.copy(), action_vector, self.topological_friction

    def _update_standing_wave_status(self):
        """
        Checks if state point is at a stable minimum (Standing Wave Node),
        where topological friction drops near zero.
        """
        min_dist = float('inf')
        closest_attractor = None
        for a in self.attractors:
            if a.depth > 0:  # Attractor
                dist = np.linalg.norm(self.state_point - a.position)
                if dist < min_dist:
                    min_dist = dist
                    closest_attractor = a

        if closest_attractor is not None and min_dist < closest_attractor.width * 0.3:
            self.standing_wave_active = True
        else:
            self.standing_wave_active = False

    def dynamic_relaxation(self, achieved_attractor_name: str):
        """
        Dynamic Relaxation: When a goal attractor is achieved and standing wave is formed,
        the attractor well is flattened (depth -> 0), allowing the state point to flow
        naturally to the next deepest attractor well.
        """
        for a in self.attractors:
            if a.name == achieved_attractor_name:
                a.depth = 0.0  # Flatten well
                break
        self.standing_wave_active = False
