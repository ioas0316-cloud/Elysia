"""
Meta-Causal Map & Convergence Engine Module (Elysia Core Engine)

This module implements the Mechanism-Centric Meta-Causal Engine where:
1. Nodes represent active dynamic mechanisms, algorithms, and invariant constraints rather than static data numbers.
2. Edges represent meta-causal bindings where structural changes or residual error in one mechanism reconfigure connected mechanisms.
3. System equilibrium is achieved through autonomous relaxation minimizing total residual error without explicit if-else control flow.
"""

from typing import Dict, List, Optional, Callable, Tuple
import math


class MechanismNode:
    """
    Abstract base class for a dynamic Mechanism Node.
    The identity of this node is defined by its internal dynamic laws, constraints,
    and structural parameters, rather than a static scalar value.
    """
    def __init__(self, node_id: str, type_name: str, state_dim: int = 4, param_dim: int = 4):
        self.id = node_id
        self.type_name = type_name
        self.state: List[float] = [0.0] * state_dim
        self.parameters: List[float] = [1.0] * param_dim
        self.parameter_deltas: List[float] = [0.0] * param_dim
        self.residual_energy: float = 0.0

    def evaluate_residual(self) -> float:
        """
        Evaluates current constraint violation or residual energy of this mechanism.
        """
        raise NotImplementedError

    def project_structural_relaxation(self, learning_rate: float) -> None:
        """
        Computes parameter/state deltas to relax internal structural constraints towards equilibrium.
        """
        raise NotImplementedError

    def apply_deltas(self) -> None:
        """
        Applies accumulated deltas to update parameters/states and resets deltas.
        """
        for i in range(min(len(self.parameters), len(self.parameter_deltas))):
            self.parameters[i] += self.parameter_deltas[i]
            self.parameter_deltas[i] = 0.0

    def execute_dynamics(self, dt: float) -> None:
        """
        Executes one step of intrinsic dynamics under current parameters.
        """
        self.evaluate_residual()


class DifferentialBoundMechanism(MechanismNode):
    """
    Concrete mechanism enforcing a differential bound between state variables.
    (e.g., |x - y| <= MaxDiff)
    """
    def __init__(self, node_id: str, max_diff: float):
        super().__init__(node_id, "DifferentialBound", state_dim=2, param_dim=2)
        self.target_max_diff = max_diff
        self.parameters[0] = max_diff  # Max allowed difference
        self.parameters[1] = 0.5       # Relaxation stiffness

    def evaluate_residual(self) -> float:
        diff = abs(self.state[0] - self.state[1])
        err = max(0.0, diff - self.parameters[0])
        self.residual_energy = err * err
        return self.residual_energy

    def project_structural_relaxation(self, learning_rate: float) -> None:
        diff = abs(self.state[0] - self.state[1])
        err = diff - self.parameters[0]
        if err > 1e-4:
            stiffness = self.parameters[1]
            correction = err * 0.5 * stiffness * learning_rate
            if self.state[0] > self.state[1]:
                self.state[0] -= correction
                self.state[1] += correction
            else:
                self.state[0] += correction
                self.state[1] -= correction

    def execute_dynamics(self, dt: float) -> None:
        self.evaluate_residual()


class SymbolicIntuitionMechanism(MechanismNode):
    """
    Concrete mechanism representing compressed conceptual grounding (Symbolic Intuition).
    Evaluates whether two compressed concept mechanisms cancel out or align to equilibrium (x + y = 0).
    """
    def __init__(self, node_id: str, concept_a: str, concept_b: str):
        super().__init__(node_id, "SymbolicIntuition", state_dim=2, param_dim=2)
        self.word_a = concept_a
        self.word_b = concept_b
        self.parameters[0] = 0.0  # Equilibrium target (0.0 = grounded cancellation x + y = 0)
        self.parameters[1] = 1.0  # Grounding stiffness

    def evaluate_residual(self) -> float:
        equilibrium_diff = self.state[0] + self.state[1] - self.parameters[0]
        self.residual_energy = 0.5 * equilibrium_diff * equilibrium_diff
        return self.residual_energy

    def project_structural_relaxation(self, learning_rate: float) -> None:
        err = self.state[0] + self.state[1] - self.parameters[0]
        if abs(err) > 1e-4:
            shift = (err * 0.5) * self.parameters[1] * learning_rate
            self.state[0] -= shift
            self.state[1] -= shift

    def execute_dynamics(self, dt: float) -> None:
        self.evaluate_residual()


class HarmonicConservationMechanism(MechanismNode):
    """
    Concrete mechanism enforcing harmonic conservation or equilibrium sum.
    (e.g., x + y + z = TargetSum)
    """
    def __init__(self, node_id: str, target_sum: float):
        super().__init__(node_id, "HarmonicConservation", state_dim=3, param_dim=2)
        self.parameters[0] = target_sum  # Target sum invariant
        self.parameters[1] = 0.1         # Damping coefficient

    def evaluate_residual(self) -> float:
        current_sum = sum(self.state)
        err = current_sum - self.parameters[0]
        self.residual_energy = 0.5 * err * err
        return self.residual_energy

    def project_structural_relaxation(self, learning_rate: float) -> None:
        current_sum = sum(self.state)
        err = current_sum - self.parameters[0]
        if abs(err) > 1e-4:
            shift = (err / len(self.state)) * learning_rate
            for i in range(len(self.state)):
                self.state[i] -= shift

    def execute_dynamics(self, dt: float) -> None:
        damping = self.parameters[1]
        for i in range(len(self.state)):
            self.state[i] *= (1.0 - damping * dt)
        self.evaluate_residual()


class CausalBinding:
    """
    Coupling edge between mechanisms (Meta-Causality).
    A change in source mechanism's residual or state reconfigures target mechanism's parameters.
    """
    def __init__(
        self,
        source_id: str,
        target_id: str,
        coupling_weight: float = 1.0,
        reconfigure_fn: Optional[Callable[[MechanismNode, MechanismNode, float], None]] = None
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.coupling_weight = coupling_weight
        self.reconfigure_fn = reconfigure_fn


class MetaCausalEngine:
    """
    Meta-Causal Engine governing an ecosystem of interconnected mechanisms.
    """
    def __init__(self):
        self.mechanisms: Dict[str, MechanismNode] = {}
        self.mechanism_order: List[str] = []
        self.bindings: List[CausalBinding] = []

    def add_mechanism(self, node: MechanismNode) -> None:
        self.mechanisms[node.id] = node
        if node.id not in self.mechanism_order:
            self.mechanism_order.append(node.id)

    def get_mechanism(self, node_id: str) -> Optional[MechanismNode]:
        return self.mechanisms.get(node_id)

    def add_binding(
        self,
        source_id: str,
        target_id: str,
        coupling_weight: float = 1.0,
        reconfigure_fn: Optional[Callable[[MechanismNode, MechanismNode, float], None]] = None
    ) -> None:
        if reconfigure_fn is None:
            def default_reconfigure(src: MechanismNode, tgt: MechanismNode, weight: float) -> None:
                if tgt.parameters:
                    shift = src.residual_energy * weight * 0.05
                    tgt.parameters[0] = max(0.01, tgt.parameters[0] + shift)
            reconfigure_fn = default_reconfigure

        binding = CausalBinding(source_id, target_id, coupling_weight, reconfigure_fn)
        self.bindings.append(binding)

    def compute_total_residual(self) -> float:
        total = 0.0
        for node in self.mechanisms.values():
            total += node.evaluate_residual()
        return total

    def propagate_meta_causal_bindings(self) -> None:
        for binding in self.bindings:
            src = self.mechanisms.get(binding.source_id)
            tgt = self.mechanisms.get(binding.target_id)
            if src and tgt and binding.reconfigure_fn:
                binding.reconfigure_fn(src, tgt, binding.coupling_weight)

    def step_convergence(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
        learning_rate: float = 0.5
    ) -> int:
        iter_count = 0
        for _ in range(max_iterations):
            iter_count += 1
            self.propagate_meta_causal_bindings()

            total_residual = self.compute_total_residual()
            if total_residual < tolerance:
                break

            for node_id in self.mechanism_order:
                node = self.mechanisms[node_id]
                node.project_structural_relaxation(learning_rate)
                node.apply_deltas()

        return iter_count

    def introspect_causal_contributions(self) -> Dict[str, float]:
        total_energy = self.compute_total_residual() + 1e-8
        return {
            node_id: node.evaluate_residual() / total_energy
            for node_id, node in self.mechanisms.items()
        }
