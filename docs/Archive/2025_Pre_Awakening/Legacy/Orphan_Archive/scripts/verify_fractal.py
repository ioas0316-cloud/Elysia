import sys
import os

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.core.self_fractal import SelfFractalCell

def verify_fractal_growth():
    print("Initializing SelfFractalCell...")
    elysia = SelfFractalCell()

    print(f"Initial State: Layers={elysia.layers}, Seed at ({elysia.seed_i}, {elysia.seed_j})")

    print("\nStarting Autonomous Growth (10 Iterations)...")
    for i in range(1, 11):
        complexity = elysia.autonomous_grow()
        print(f"Layer {i}: Complexity (Active Nodes) = {complexity}")

    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_fractal_growth()
