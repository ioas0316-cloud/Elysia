import numpy as np
from typing import Dict, List, Any
from core.physics.causal_gravity_engine import CausalGravityEngine

class CausalGravityInference:
    """
    [Phase: Grand Leap - Field Inference]
    Replaces point-based probability with 'Causal Gravity' curvature.
    Instead of asking "What is the probability?", we ask "Where does the field suvery?"
    """
    def __init__(self, dimensions: int = 8):
        self.gravity = CausalGravityEngine(dimensions=dimensions)

    def map_engrams_to_field(self, engrams: List[Dict[str, Any]]):
        """Projects stored engrams as mass-points in the gravitational field."""
        for e in engrams:
            eid = e.get("engram_id", "unknown")
            # Extract or generate a coordinate from the data_blob
            # In a real system, the quaternion or tension_vector is used
            data = e.get("data", {})
            pos = np.zeros(self.gravity.dimensions)

            # Use 'stability' or 'emotional_value' as mass
            mass = e.get("emotional_value", 1.0)

            if "quaternion" in data:
                q = data["quaternion"] # [x, y, z, w]
                pos[:len(q)] = q
            elif "axis_vector" in data:
                v = data["axis_vector"]
                pos[:len(v)] = v

            self.gravity.add_node(eid, str(data.get("type", "engram")), [])
            self.gravity.nodes[eid].position = pos
            self.gravity.nodes[eid].mass = mass

    def infer_necessity(self, stimulus_vector: np.ndarray) -> Dict[str, Any]:
        """
        Injects a stimulus as a 'Question Mass' and observes where the field converges.
        This is 'Causal Necessity' (필연성).
        """
        q_id = "stimulus_focus"
        self.gravity.add_node(q_id, "stimulus", [])
        self.gravity.nodes[q_id].position = stimulus_vector
        self.gravity.nodes[q_id].mass = 10.0 # High mass focus

        # Evolve the field to find equilibrium
        for _ in range(50):
            self.gravity.step(dt=0.05)

        # The node closest to the focus after evolution is the 'Necessary Result'
        best_node = None
        min_dist = float('inf')

        focus_pos = self.gravity.nodes[q_id].position
        for nid, node in self.gravity.nodes.items():
            if nid == q_id: continue
            dist = np.linalg.norm(node.position - focus_pos)
            if dist < min_dist:
                min_dist = dist
                best_node = node

        return {
            "necessary_result_id": best_node.id if best_node else None,
            "convergence_distance": min_dist,
            "field_curvature": 1.0 / (1.0 + min_dist)
        }

if __name__ == "__main__":
    inference = CausalGravityInference(dimensions=4)
    # Mock engrams
    mock_engrams = [
        {"engram_id": "E1", "emotional_value": 2.0, "data": {"quaternion": [1, 0, 0, 0]}},
        {"engram_id": "E2", "emotional_value": 0.5, "data": {"quaternion": [0, 1, 0, 0]}}
    ]
    inference.map_engrams_to_field(mock_engrams)

    # Stimulus close to E1
    stim = np.array([0.9, 0.1, 0, 0])
    result = inference.infer_necessity(stim)
    print(f"Inference Result: {result}")
