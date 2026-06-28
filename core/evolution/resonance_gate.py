import numpy as np
from typing import List, Callable, Any, Dict, Union

class TrajectoryResonanceGate:
    """
    [Phase: Grand Leap - Field Evolution] Trajectory Resonance Gate
    Replaces point-based signatures with 'Causal Trajectories' (Lines/Planes).
    Resonance is achieved when the organism's movement through state-space
    aligns with the structural 'Flow' of a logic path.
    """
    def __init__(self, controller=None):
        self.paths = []
        self.controller = controller
        # Circular buffer for organism state history [N_steps, Dim]
        self.history_limit = 10
        self.state_history = []

    def register_path(self, path_id: str, logic_func: Callable, trajectory_signature: np.ndarray):
        """
        Registers a path with a 'Trajectory Signature' (Sequence of states).
        """
        self.paths.append({
            "id": path_id,
            "logic": logic_func,
            "trajectory": trajectory_signature
        })

    def update_history(self, current_tensor: np.ndarray):
        self.state_history.append(current_tensor)
        if len(self.state_history) > self.history_limit:
            self.state_history.pop(0)

    def execute_with_field_resonance(self, input_data: Any) -> Dict[str, Any]:
        if not self.paths or len(self.state_history) < 2:
            return {"error": "Insufficient history or no paths"}

        current_trajectory = np.array(self.state_history)

        best_path = None
        max_resonance = -1.0
        scores = []

        for p in self.paths:
            # Field Resonance: Use Trajectory Sameness (Gram Matrix comparison)
            # if we have a controller, use its advanced comparison logic
            if self.controller:
                # find_trajectory_sameness expects lists/arrays
                res = self.controller.find_trajectory_sameness(
                    p["trajectory"],
                    current_trajectory,
                    scale_factor=1.0
                )
                # Use the variance and min_difference to calculate a resonance score
                # Low variance and low min_diff = High resonance
                sameness = max([d['sameness_score'] for d in res['sameness_distribution']])
                resonance = sameness
            else:
                # Fallback to simple MSE between trajectories (padded)
                t1 = p["trajectory"]
                t2 = current_trajectory
                # Simple alignment
                min_len = min(len(t1), len(t2))
                dist = np.linalg.norm(t1[-min_len:] - t2[-min_len:])
                resonance = 1.0 / (1.0 + dist)

            scores.append((p["id"], float(resonance)))

            if resonance > max_resonance:
                max_resonance = resonance
                best_path = p

        print(f"[FieldResonance] Flow Selection Scores: {scores}")

        # Execute the winner
        result = best_path["logic"](input_data)

        return {
            "selected_path": best_path["id"],
            "resonance": max_resonance,
            "result": result,
            "scores": scores
        }

if __name__ == "__main__":
    # Mock test
    gate = TrajectoryResonanceGate()

    # Path: Growth Flow
    gate.register_path(
        "GROWTH_SPIRAL",
        lambda x: "Growing...",
        np.array([[0,0], [1,1], [2,2]])
    )

    # Path: Decay Flow
    gate.register_path(
        "DECAY_STASIS",
        lambda x: "Decaying...",
        np.array([[2,2], [1,1], [0,0]])
    )

    # Simulate history
    gate.update_history(np.array([0.1, 0.1]))
    gate.update_history(np.array([1.1, 1.1]))
    gate.update_history(np.array([1.9, 2.0]))

    print(gate.execute_with_field_resonance("Data"))
