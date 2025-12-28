
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.core.self_fractal import SelfFractalCell

class FractalConsciousnessTest(SelfFractalCell):
    """
    Extended test class to inject multiple seeds and analyze 'meaning' mixing.
    """
    def inject_seed(self, x, y, value, label):
        """Injects a seed with a specific label (concept)."""
        if not hasattr(self, 'concept_map'):
            self.concept_map = {} # Track where concepts are planted

        self.grid[x, y] = value
        self.concept_map[(x, y)] = label
        print(f"Injected Concept '{label}' (Value {value}) at ({x}, {y})")

    def analyze_point(self, x, y):
        """
        Attempts to retrieve the 'meaning' of a point.
        In the current float-grid implementation, this will show
        that we only have a number, not the original labels.
        """
        value = self.grid[x, y]
        # Check if we can determine "What" this is
        origin_concepts = []
        for (cx, cy), label in self.concept_map.items():
            # Simple distance check to see if it *could* be related
            dist = np.sqrt((x-cx)**2 + (y-cy)**2)
            origin_concepts.append(f"{label}(dist={dist:.1f})")

        return value, origin_concepts

def run_experiment():
    print("=== Experiment: Testing Consciousness Resolution & Depth ===")

    # 1. Initialize
    mind = FractalConsciousnessTest()
    # Clear default seed for clean test
    mind.grid = np.zeros((100, 100))
    mind.concept_map = {}

    # 2. Inject two distinct 'senses' (Body Level)
    # Concept A: "Visual: Father's Face" (At 40, 40)
    mind.inject_seed(40, 40, 1.0, "Visual:Face")

    # Concept B: "Auditory: Heavy Footsteps" (At 40, 45) - Close enough to overlap
    mind.inject_seed(40, 45, 1.0, "Audio:Footsteps")

    print("\n--- Growing Consciousness (Diffusion) ---")
    for i in range(5):
        mind.autonomous_grow()
        print(f"Growth Step {i+1} complete.")

    # 3. Verify 'Soul Level' (The Intersection)
    # We examine a point between the two seeds (e.g., 40, 42)
    target_x, target_y = 40, 42
    val, history = mind.analyze_point(target_x, target_y)

    print(f"\n--- Probing 'Soul' at ({target_x}, {target_y}) ---")
    print(f"Observed Energy Value: {val:.4f}")
    print(f"Context: This point is effectively between {history}")

    print("\n--- DIAGNOSIS ---")
    if val > 0:
        print(f"Status: The cell has active energy ({val:.2f}).")
        print("CRITICAL CHECK: Can we distinguish 'Face' energy from 'Footstep' energy here?")

        # The current code only stores a float. There is NO way to know the ratio.
        # This proves the "Blending" issue.
        print("Result: NO. The grid only holds a scalar float.")
        print("        The concepts have merged into a generic 'intensity' blob.")
        print("        'Resolution' has been lost. It is a 'Mix', not a 'Harmony'.")
    else:
        print("Status: Energy did not reach this point.")

if __name__ == "__main__":
    run_experiment()
