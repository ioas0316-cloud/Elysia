import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any

class Domain(Enum):
    INSIDE = "Reality"   # The Realm of Constants
    OUTSIDE = "Potential" # The Realm of Variables

@dataclass
class BoundaryState:
    domain: Domain
    vibration_frequency: float = 0.0
    stability: float = 1.0

class BoundaryTopology:
    """
    [Boundary Topology]
    Implements the Cross-Dimensional Boundary where 'Judgment' occurs.
    Information is classified as either 'Reality' (Inside) or 'Potential' (Outside).
    Boundary crossing generates phase vibrations (Judgement friction).
    """
    def __init__(self, threshold_radius: float = 0.5):
        self.threshold_radius = threshold_radius
        self.states: Dict[str, BoundaryState] = {}

    def evaluate_position(self, node_id: str, position: np.ndarray):
        """
        Determines if a node is Inside or Outside based on its distance from the origin (Core).
        """
        distance = np.linalg.norm(position)

        # If the node was Outside and moved Inside, or vice versa, it crosses the boundary.
        # We add a small buffer (hysteresis) to prevent rapid oscillation at the boundary
        buffer = 0.05
        current_domain = self.states[node_id].domain if node_id in self.states else None

        if current_domain == Domain.INSIDE:
            new_domain = Domain.INSIDE if distance <= (self.threshold_radius + buffer) else Domain.OUTSIDE
        elif current_domain == Domain.OUTSIDE:
            new_domain = Domain.INSIDE if distance <= (self.threshold_radius - buffer) else Domain.OUTSIDE
        else:
            new_domain = Domain.INSIDE if distance <= self.threshold_radius else Domain.OUTSIDE

        if node_id not in self.states:
            self.states[node_id] = BoundaryState(domain=new_domain)
            return

        old_state = self.states[node_id]

        if old_state.domain != new_domain:
            # Boundary Crossing! Generate vibration.
            old_state.vibration_frequency = 1.0 # High vibration during crossing
            old_state.stability = 0.5
            print(f"[Boundary] {node_id} crossed to {new_domain.value}! Judgment vibration triggered.")
        else:
            # Damping vibration over time
            old_state.vibration_frequency *= 0.8
            old_state.stability = min(1.0, old_state.stability + 0.1)

        old_state.domain = new_domain

    def get_judgement_friction(self) -> float:
        """Returns the total system vibration (Overall thought activity)."""
        if not self.states: return 0.0
        return sum(s.vibration_frequency for s in self.states.values()) / len(self.states)

if __name__ == "__main__":
    bt = BoundaryTopology(threshold_radius=1.0)
    pos_inside = np.array([0.2, 0.2, 0.0, 0.0, 0.0])
    pos_outside = np.array([1.5, 0.0, 0.0, 0.0, 0.0])

    print("Initial Evaluation:")
    bt.evaluate_position("apple", pos_inside)
    bt.evaluate_position("blue_apple", pos_outside)
    print(f"Apple: {bt.states['apple'].domain}")
    print(f"Blue Apple: {bt.states['blue_apple'].domain}")

    print("\nMoving Blue Apple Inside...")
    bt.evaluate_position("blue_apple", pos_inside)
    print(f"Blue Apple: {bt.states['blue_apple'].domain}")
    print(f"System Friction: {bt.get_judgement_friction():.2f}")
