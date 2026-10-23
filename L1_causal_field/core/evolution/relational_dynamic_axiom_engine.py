"""
Relational Dynamic Axiom Engine & Embodied Virtual Environment

Implements continuous causal field dynamics where axioms are treated as
'Relative Reference Frames (Pivots)' rather than fixed absolute truths.

Core Mechanisms:
1. Embodied Virtual Environment: Physical simulation supporting do-interventions
   and environmental rule shifts (e.g. mass-spring-damper, variable circuit).
2. Locality Constraint: Localizes tension/friction upon do-intervention,
   unlocking ONLY affected local edges/nodes into variable resistor $x$ dials,
   keeping unaffected axioms anchored.
3. Principle of Least Action Re-crystallization: Balances error reduction
   against information complexity cost before freezing variable $x$ back into
   a condensed Axiomatic Axis.
4. Back-trace Causal Projection: Translates $x$-dial displacements and new
   relational axioms into human-interpretable symbolic / mathematical expressions.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class AxiomNode:
    """
    Represents a relational concept, physical variable, or mathematical operator
    within the causal field topology.
    """
    def __init__(self, node_id: str, name: str, value: float = 0.0, is_axis: bool = True, resistor_x: float = 0.001):
        self.node_id = node_id
        self.name = name
        self.value = value
        self.is_axis = is_axis          # True if currently locked as a Relational Axis Pivot
        self.resistor_x = resistor_x    # Variable resistor $x$ (impedance / degree of freedom)
        self.invariance_score = 0.95 if is_axis else 0.5
        self.edges: Dict[str, float] = {}  # Connected node_id -> edge coupling strength

class EmbodiedVirtualEnvironment:
    """
    Virtual Embodied Environment (Mass-Spring Circuit Sandbox).
    Allows do-interventions and environmental rule shifts.
    """
    def __init__(self, mass: float = 1.0, stiffness: float = 10.0, damping: float = 0.5):
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping
        self.position = 0.0
        self.velocity = 0.0
        self.time = 0.0
        self.dt = 0.01

    def step(self, external_force: float = 0.0) -> Dict[str, float]:
        """Advance physical state by 1 timestep."""
        # F_spring = -k * x
        # F_damping = -c * v
        f_spring = -self.stiffness * self.position
        f_damping = -self.damping * self.velocity
        total_force = f_spring + f_damping + external_force

        acceleration = total_force / self.mass
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt
        self.time += self.dt

        return {
            "position": self.position,
            "velocity": self.velocity,
            "acceleration": acceleration,
            "force": total_force,
            "energy": 0.5 * self.mass * (self.velocity**2) + 0.5 * self.stiffness * (self.position**2)
        }

    def do_intervention(self, target_var: str, value: float):
        """Perform do-operation (e.g. do(stiffness=20.0) or do(position=5.0))."""
        if target_var == "stiffness":
            self.stiffness = value
        elif target_var == "damping":
            self.damping = value
        elif target_var == "mass":
            self.mass = max(0.01, value)
        elif target_var == "position":
            self.position = value
        elif target_var == "velocity":
            self.velocity = value


class RelationalDynamicAxiomEngine:
    """
    Relational Dynamic Axiom Engine managing the transition between
    Relative Axioms (Axis) and Variable Resistors ($x$).
    """
    def __init__(
        self,
        relativization_threshold: float = 0.6,
        condensation_threshold: float = 0.85,
        complexity_penalty: float = 0.1,
        min_resistor_x: float = 0.001,
        max_resistor_x: float = 50.0,
    ):
        self.relativization_threshold = relativization_threshold
        self.condensation_threshold = condensation_threshold
        self.complexity_penalty = complexity_penalty
        self.min_resistor_x = min_resistor_x
        self.max_resistor_x = max_resistor_x

        self.nodes: Dict[str, AxiomNode] = {}
        self.causal_trace: List[Dict[str, Any]] = []
        self._initialize_primary_axioms()

    def _initialize_primary_axioms(self):
        """Initialize primary relative axioms (relational reference frames)."""
        # Physical / Mathematical Relational Primitives
        axioms = [
            ("hooke_law", "F = -k * x", 0.0, True, 0.001),
            ("mass", "Mass (m)", 1.0, True, 0.001),
            ("stiffness", "Stiffness (k)", 10.0, True, 0.001),
            ("damping", "Damping (c)", 0.5, True, 0.001),
            ("position", "Position (x)", 0.0, False, 1.0),
            ("velocity", "Velocity (v)", 0.0, False, 1.0),
            ("energy_cons", "dE/dt <= 0 (Energy Dissipation)", 0.0, True, 0.001)
        ]

        for node_id, name, val, is_axis, res_x in axioms:
            node = AxiomNode(node_id, name, val, is_axis, res_x)
            self.nodes[node_id] = node

        # Build causal topological edges
        self.add_coupling("hooke_law", "stiffness", 0.9)
        self.add_coupling("hooke_law", "position", 0.9)
        self.add_coupling("position", "velocity", 0.8)
        self.add_coupling("energy_cons", "damping", 0.7)

    def add_coupling(self, node_a: str, node_b: str, strength: float):
        """Add bidirectional causal coupling between nodes."""
        if node_a in self.nodes and node_b in self.nodes:
            self.nodes[node_a].edges[node_b] = strength
            self.nodes[node_b].edges[node_a] = strength

    def process_observation(
        self,
        env_state: Dict[str, float],
        prediction: Dict[str, float],
        intervention_node: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process incoming state observation vs expected prediction.
        Calculates local tension, applies Locality Constraint,
        adjusts $x$ resistors, and checks for Re-crystallization via
        Principle of Least Action.
        """
        # 1. Compute prediction error / tension
        pos_err = abs(env_state["position"] - prediction.get("position", env_state["position"]))
        vel_err = abs(env_state["velocity"] - prediction.get("velocity", env_state["velocity"]))
        total_tension = pos_err + vel_err

        unlocked_nodes = []
        condensed_nodes = []

        # 2. LOCALITY CONSTRAINT:
        # Identify directly affected nodes if intervention or local error occurred.
        # If tension exceeds relativization_threshold, unlock ONLY local edges/nodes!
        if total_tension > self.relativization_threshold:
            affected_nodes = set()
            if intervention_node and intervention_node in self.nodes:
                affected_nodes.add(intervention_node)
                # Include immediate 1-hop local neighbors
                for neighbor in self.nodes[intervention_node].edges:
                    affected_nodes.add(neighbor)
            else:
                # If no specific intervention node given, pick local node with highest edge weight to position/velocity
                affected_nodes.update(["stiffness", "hooke_law"])

            # Relativize ONLY affected local nodes into variable resistor x!
            for node_id in affected_nodes:
                node = self.nodes[node_id]
                if node.is_axis:
                    node.is_axis = False
                    node.resistor_x = min(self.max_resistor_x, node.resistor_x + 2.0 * total_tension)
                    node.invariance_score = max(0.1, node.invariance_score - 0.25)
                    unlocked_nodes.append(node_id)
                else:
                    # Increase impedance x with tension
                    node.resistor_x = min(self.max_resistor_x, node.resistor_x + 0.5 * total_tension)

        # 3. PRINCIPLE OF LEAST ACTION RE-CRYSTALLIZATION:
        # Check variable $x$ nodes for re-crystallization into Axiom Axis.
        # Action Cost S = Error_Reduction - (Complexity_Penalty * Info_Volume)
        for node_id, node in self.nodes.items():
            if not node.is_axis:
                if total_tension < 0.2:
                    # Small error -> increase invariance score
                    node.invariance_score = min(1.0, node.invariance_score + 0.08)
                    node.resistor_x = max(self.min_resistor_x, node.resistor_x * 0.8)

                    # Compute Principle of Least Action score
                    error_reduction = 1.0 - total_tension
                    info_complexity = len(node.edges) * 0.05 + 0.1  # cost of freezing new axiom
                    net_action_gain = error_reduction - (self.complexity_penalty * info_complexity)

                    if node.invariance_score >= self.condensation_threshold and net_action_gain > 0.5:
                        node.is_axis = True
                        node.resistor_x = self.min_resistor_x
                        condensed_nodes.append(node_id)

        # Log trace
        trace_record = {
            "time": env_state.get("time", 0.0),
            "tension": total_tension,
            "unlocked_nodes": unlocked_nodes,
            "condensed_nodes": condensed_nodes,
            "axes_count": sum(1 for n in self.nodes.values() if n.is_axis),
            "vars_count": sum(1 for n in self.nodes.values() if not n.is_axis)
        }
        self.causal_trace.append(trace_record)

        return trace_record

    def backtrace_projection(self) -> str:
        """
        Back-trace causal state into human-interpretable mathematical/symbolic expression.
        """
        axis_names = [n.name for n in self.nodes.values() if n.is_axis]
        var_status = [f"{n.node_id}(x={n.resistor_x:.3f})" for n in self.nodes.values() if not n.is_axis]

        projection = (
            f"=== RELATIONAL CAUSAL PROJECTION ===\n"
            f"Locked Relational Axiom Axes [{len(axis_names)}]: {', '.join(axis_names)}\n"
            f"Dynamic Variable Resistors [{len(var_status)}]: {', '.join(var_status)}\n"
            f"Causal Trace Length: {len(self.causal_trace)} steps"
        )
        return projection
