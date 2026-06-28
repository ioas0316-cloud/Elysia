import numpy as np
from typing import List, Callable, Any, Dict

class ResonanceGate:
    """
    [Phase: Grand Leap] Non-linear Resonance Gate
    Replaces static if-else branches with 'Resonance-based Flow Selection'.
    Execution paths are chosen based on which logic 'resonates' most with the
    current system state (Organism Tensor).
    """
    def __init__(self):
        # paths: list of { "id": str, "logic": function, "vibration_signature": np.ndarray }
        self.paths = []

    def register_path(self, path_id: str, logic_func: Callable, signature: np.ndarray):
        """
        Registers a potential execution path with its unique 'vibration signature'.
        """
        self.paths.append({
            "id": path_id,
            "logic": logic_func,
            "signature": signature
        })

    def execute_with_resonance(self, organism_tensor: np.ndarray, input_data: Any) -> Dict[str, Any]:
        """
        Selects the execution path that has the highest resonance (lowest variance)
        with the current organism tensor.
        """
        if not self.paths:
            return {"error": "No paths registered"}

        best_path = None
        max_resonance = -1.0

        # Calculate resonance scores
        scores = []
        for p in self.paths:
            # Resonance = 1.0 / (1.0 + distance)
            # We compare the organism's current 'shape' with the path's 'signature'
            dist = np.linalg.norm(organism_tensor - p["signature"])
            resonance = 1.0 / (1.0 + dist)
            scores.append((p["id"], resonance))

            if resonance > max_resonance:
                max_resonance = resonance
                best_path = p

        print(f"[ResonanceGate] Path Selection Scores: {scores}")
        print(f"[ResonanceGate] Winner: {best_path['id']} (Resonance: {max_resonance:.4f})")

        # Execute the chosen path
        result = best_path["logic"](input_data)

        return {
            "selected_path": best_path["id"],
            "resonance": max_resonance,
            "result": result
        }

if __name__ == "__main__":
    gate = ResonanceGate()

    # Path A: Stable path
    gate.register_path("STABLE_SYNC", lambda x: f"Stable Process: {x}", np.array([0, 0, 1, 0, 0, 0]))
    # Path B: High Tension path (Crisis response)
    gate.register_path("CRISIS_MUTATION", lambda x: f"Mutation Process: {x}", np.array([5, 0, 0, 0, 0, 0]))

    # Test 1: Stable system
    print("Test 1: Stable System")
    stable_tensor = np.array([0.1, 0.1, 1.0, 0.5, 0.2, 0.05])
    print(gate.execute_with_resonance(stable_tensor, "Data_A"))

    # Test 2: High tension system
    print("\nTest 2: High Tension System")
    tension_tensor = np.array([4.8, 0.2, 0.5, 2.5, 0.8, 0.9])
    print(gate.execute_with_resonance(tension_tensor, "Data_B"))
