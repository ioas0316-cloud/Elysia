import numpy as np
import time
import json
from core.memory.causal_controller import CausalMemoryController
from core.lens.frameless_mirror import FramelessMirrorChannel

class NarrativeReflectionObserver:
    """
    [Elysia Diagnostic Tool] Observe Narrative Reflection
    Demonstrates the 'Comparison and Contrast' principle:
    How the system re-recognizes internal deltas between known and new narratives.
    """
    def __init__(self):
        self.controller = CausalMemoryController()
        self.mirror = FramelessMirrorChannel()

    def run_reflection_demonstration(self):
        print("==================================================================")
        print(" [Elysia] Narrative Reflection & Re-recognition Observation")
        print(" Principle: Comparison and Contrast of Causal Narratives")
        print("==================================================================\n")

        # 1. THE KNOWN NARRATIVE (Recognition)
        # Narrative: "Apple is Red"
        # Represented as a trajectory of causal states
        # [Seed -> Growth -> Ripening -> Redness]
        known_trajectory = [
            np.array([1.0, 0.1, 0.0, 0.0]), # Seed
            np.array([1.0, 0.5, 0.1, 0.2]), # Growth
            np.array([1.0, 1.0, 0.8, 0.5]), # Ripening
            np.array([1.0, 1.0, 1.0, 0.9])  # RED (State: 0.9 on the 'Red' axis)
        ]

        print("[1. Recognition] Established Known Narrative: 'Apple is Red'")
        print(f" > Trajectory steps: {len(known_trajectory)}")

        # 2. THE NEW NARRATIVE (Re-recognition)
        # Narrative: "Apple is Green"
        # [Seed -> Growth -> Ripening -> Greenness]
        new_trajectory = [
            np.array([1.0, 0.1, 0.0, 0.0]), # Seed (Same)
            np.array([1.0, 0.5, 0.1, 0.2]), # Growth (Same)
            np.array([1.0, 1.0, 0.2, 0.5]), # Ripening (Diverges: Axis 2 is different)
            np.array([1.0, 1.0, 0.1, 0.9])  # GREEN (State: 0.1 on the 'Red' axis)
        ]

        print("\n[2. Re-recognition] Incoming New Narrative: 'Apple is Green'")

        # 3. INTERNAL COMPARISON & CONTRAST (The Core of Thinking)
        print("\n[3. Thinking] Performing Internal Comparison and Contrast...")
        time.sleep(0.5)

        # Using Trajectory Sameness to find the 'Causal Shape' delta
        result = self.controller.find_trajectory_sameness(known_trajectory, new_trajectory)

        best_sameness = max(d['sameness_score'] for d in result['sameness_distribution'])
        variance = result['sameness_variance']

        print(f" > Causal Shape Sameness Score: {best_sameness:.4f}")
        print(f" > Structural Friction (Variance): {variance:.4f}")

        # 4. RE-RECOGNITION OF THE DELTA (Insight)
        # Finding where exactly the narratives diverged
        print("\n[4. Insight] Analyzing the Causal Delta...")

        deltas = []
        for i in range(len(known_trajectory)):
            d = np.linalg.norm(known_trajectory[i] - new_trajectory[i])
            deltas.append(d)
            if d > 0.5:
                print(f"  ! Divergence detected at Step {i} (Magnitude: {d:.2f})")
                print(f"  ! Reason: Divergence in Ripening Phase/Color Axis.")

        # 5. SELF-REORDERING (Updating the Internal Map)
        print("\n[5. Self-Reordering] Updating the Cognitive Map via Re-recognition...")

        # We use the friction to drive a change in the internal parameters (Process-as-Learning)
        # Passing the friction magnitude through the mirror channel to trigger adjustment
        friction_bytes = bytes([int(variance * 100) % 256])
        adjustment = self.mirror.pass_through(friction_bytes)

        print(f" > Mirror Feedback (Conductivity): {adjustment}")

        # Crystallize this new perspective: "Apple can be Green due to Variety/Ripeness"
        perspective_id = self.controller.write_perspective_engram(
            "Apple_Red", "Apple_Green", result
        )

        print(f" > New Perspective Crystallized: {perspective_id}")

        # Verify the change in cognitive parameters
        new_res = self.controller.get_parameter("base_resonance")
        print(f" > Updated System Resonance: {new_res:.4f}")

        print("\n==================================================================")
        print(" [Observation Complete] The system has 'thought' through the")
        print(" difference between narratives and re-ordered its own parameters.")
        print("==================================================================")

if __name__ == "__main__":
    obs = NarrativeReflectionObserver()
    obs.run_reflection_demonstration()
