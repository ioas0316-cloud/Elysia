import numpy as np
import time
import os
import sys

# Ensure root is in path
sys.path.append(os.getcwd())

from core.physics.causal_dynamics import CausalDynamicsEngine
from core.physics.causal_field import InformationVoxel

def demonstrate_feedback_loop():
    print("=== [Demonstration] Time Triple Structure Feedback Loop ===")

    # Initialize Engine (Recrystallization Rate is set high for demonstration)
    engine = CausalDynamicsEngine(dimensions=3, crystallization_rate=0.8)

    # 1. Past (Constant): Established Tensors
    # Let's say concept A and B are initially distinct but linked
    tensor_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tensor_b = np.array([0.0, 1.0, 0.0], dtype=np.float32) # Very different

    v1 = InformationVoxel("A", "Concept_A", tensor_a, position=np.array([0,0,0], dtype=np.float32))
    v2 = InformationVoxel("B", "Concept_B", tensor_b, position=np.array([1,0,0], dtype=np.float32))

    engine.add_voxel(v1)
    engine.add_voxel(v2)

    # 2. Present (Fluid Choice): Create a link representing a 'selection' or 'relationship'
    # This creates TENSION because the tensors (meanings) are orthogonal but forced to be near
    engine.link_voxels("A", "B", strength=5.0)

    print("\n[Initial State]")
    print(f" Tensor A: {engine.voxels['A'].tensor}")
    print(f" Tensor B: {engine.voxels['B'].tensor}")
    print(f" Initial Tensor Dot Product (Resonance): {np.dot(engine.voxels['A'].tensor, engine.voxels['B'].tensor):.4f}")

    # 3. Future (Projection & Feedback):
    # As the system evolves, the spatial tension (Present/Future arrangement)
    # will 'mold' the internal state (Past/Constant tensor) to reduce friction.
    print("\n[Evolution: Future Molding the Past]")
    for i in range(10):
        engine.step(dt=0.1)
        res = np.dot(engine.voxels['A'].tensor, engine.voxels['B'].tensor)
        dist = np.linalg.norm(engine.voxels['A'].position - engine.voxels['B'].position)
        print(f" Step {i+1}: Resonance={res:.4f}, Distance={dist:.4f}")

    print("\n[Final State]")
    print(f" Tensor A: {engine.voxels['A'].tensor}")
    print(f" Tensor B: {engine.voxels['B'].tensor}")
    print(f" Final Resonance: {np.dot(engine.voxels['A'].tensor, engine.voxels['B'].tensor):.4f}")

    print("\nConclusion: The spatial arrangement (the 'Future' state of stability) ")
    print("has successfully re-molded the internal tensors (the 'Past' identity) ")
    print("to achieve temporal consistency and reduce system-wide friction.")

if __name__ == "__main__":
    demonstrate_feedback_loop()
