"""
Static Causal Graph: Topological Memory and Wave Propagation System.

This module unifies code execution and memory topology. Code structure itself
forms a static causal graph (Static Memory), where execution is represented
as wave energy propagation along topological gradients, and friction/resonance
dynamically alters connection tension (Dynamic Plasticity Loop).
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any, Tuple


class BoundaryType(Enum):
    """4-layer self-awareness boundary classification."""
    EXTERNAL_WORLD   = 0  # External Environment / Non-Self (Uncontrollable)
    PERCEPTION_LAYER = 1  # Perception Boundary (Sensory friction surface)
    PROCESSING_LAYER = 2  # Dynamic Thinking & Processing Layer
    MEMORY_LAYER     = 3  # Static Identity Core & Attractor Well


@dataclass(frozen=True)
class LayerMetadata:
    """Self-referential metadata attached to system boundary layers."""
    boundary: BoundaryType
    label: str
    description: str
    controllability: float  # 0.0 (External: Uncontrollable) ~ 1.0 (Memory Core: Fully Controllable)

    @property
    def is_self(self) -> bool:
        """Determines if the boundary is within Self (anything other than EXTERNAL_WORLD)."""
        return self.boundary != BoundaryType.EXTERNAL_WORLD


@dataclass
class CausalSignal:
    """Causal wave signal transmitted across internal/external boundaries."""
    origin_boundary: BoundaryType
    payload: Dict[str, Any]
    energy: float


@dataclass
class CausalNode:
    """
    Causal Node: A point in topological memory that accumulates energy and
    emits action vectors upon reaching potential threshold.
    """
    node_id: str
    threshold: float
    boundary: BoundaryType = BoundaryType.PROCESSING_LAYER
    potential_depth: float = 1.0  # Depth of attractor potential well (deepest in MEMORY_LAYER)
    action_vector: Optional[Callable] = None
    potential: float = 0.0

    def receive_energy(self, energy: float) -> float:
        """
        Receives incoming energy wave. If energy exceeds threshold,
        triggers action_vector and returns overflow energy + base pulse.
        """
        self.potential += energy
        if self.potential >= self.threshold:
            overflow = self.potential - self.threshold
            if self.action_vector:
                self.action_vector(self.node_id, self.potential)
            self.potential = 0.0  # Reset after release
            return overflow + 1.0  # Base transmitted energy pulse
        return 0.0


@dataclass
class CausalEdge:
    """
    Causal Edge: Connection tension (Tension) and friction path in topological memory.
    Measures accumulated friction loss and scar weight for meta-observation.
    """
    source_id: str
    target_id: str
    tension: float         # Coupling strength (0.0 ~ 1.0) - Structural Memory
    resistance: float      # Causal resistance / friction (0.0 ~ 1.0)
    accumulated_friction: float = 0.0  # Accumulated friction energy loss (Meta-Metric)
    traversal_count: int = 0
    scar_weight: float = 0.0            # Recrystallized scar weight from damage absorption

    def transmit(self, energy: float) -> float:
        """
        Transmits energy across edge, computing attenuation due to tension and resistance.
        Accumulates friction loss for meta-cognition.
        """
        effective_energy = energy * self.tension * (1.0 - self.resistance)
        loss = max(0.0, energy - effective_energy)
        self.accumulated_friction += loss
        self.traversal_count += 1
        return max(0.0, effective_energy)


class StaticCausalGraph:
    """
    Topological Memory Terrain where code structure and causal logic are unified.
    Propagates energy waves, accumulates friction, and updates edge tension (plasticity).
    """

    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[str, List[CausalEdge]] = {}

    def add_node(
        self,
        node_id: str,
        threshold: float = 1.0,
        boundary: BoundaryType = BoundaryType.PROCESSING_LAYER,
        depth: float = 1.0,
        action: Optional[Callable] = None
    ):
        """Registers a CausalNode in the topological graph."""
        self.nodes[node_id] = CausalNode(
            node_id=node_id,
            threshold=threshold,
            boundary=boundary,
            potential_depth=depth,
            action_vector=action
        )
        if node_id not in self.edges:
            self.edges[node_id] = []

    def connect(
        self,
        source_id: str,
        target_id: str,
        tension: float,
        resistance: float = 0.1
    ) -> CausalEdge:
        """Establishes a causal edge connection in static memory."""
        edge = CausalEdge(source_id, target_id, tension, resistance)
        if source_id not in self.edges:
            self.edges[source_id] = []
        self.edges[source_id].append(edge)
        return edge

    def get_all_edges_flat(self) -> List[CausalEdge]:
        """Returns a flat list of all edges in the graph."""
        flat = []
        for e_list in self.edges.values():
            flat.extend(e_list)
        return flat

    def propagate(
        self,
        entry_node_id: str,
        energy: float,
        plasticity_alpha: float = 0.01
    ) -> List[str]:
        """
        Dynamic wave flow execution over topological terrain.
        Transmits energy wave, records trace, and applies dynamic plasticity reinforcement.
        """
        if entry_node_id not in self.nodes:
            return []

        active_wave: List[Tuple[str, float]] = [(entry_node_id, energy)]
        execution_trace: List[str] = []

        while active_wave:
            next_wave: List[Tuple[str, float]] = []
            for curr_id, curr_energy in active_wave:
                if curr_id not in self.nodes:
                    continue
                node = self.nodes[curr_id]
                execution_trace.append(curr_id)

                output_energy = node.receive_energy(curr_energy)

                if output_energy > 0:
                    for edge in self.edges.get(curr_id, []):
                        transmitted = edge.transmit(output_energy)
                        if transmitted > 0.05:  # Significant wave threshold
                            next_wave.append((edge.target_id, transmitted))
                            # Plasticity reinforcement
                            if plasticity_alpha > 0.0:
                                edge.tension = min(1.0, edge.tension + (transmitted * plasticity_alpha))

            active_wave = next_wave

        return execution_trace

    def inject_stimulus(
        self,
        entry_node_id: str,
        initial_energy: float,
        plasticity_alpha: float = 0.01
    ) -> List[str]:
        """Alias for propagate to simulate stimulus injection into topological memory."""
        return self.propagate(entry_node_id, initial_energy, plasticity_alpha)
