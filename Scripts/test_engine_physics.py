
import sys
import os
import torch

# Add the project root to sys.path
sys.path.append(os.getcwd())

from Core.Keystone.sovereign_math import FractalWaveEngine, SovereignVector

def test_engine_gravity():
    print("Testing FractalWaveEngine Causal Gravity...")

    engine = FractalWaveEngine(max_nodes=10)

    # Define an attractor with high mass and rigidity
    target_vec = [0.8, 0.2, -0.5, 0.1] + [0.0]*17
    engine.define_meaning_attractor("Truth", mask="Truth", target_vector=target_vec, mass=100.0, rigidity=5.0)

    node_idx = engine.concept_to_idx["Truth"]
    print(f"Node 'Truth' index: {node_idx}")

    # Inject initial pulse
    engine.inject_pulse("Truth", energy=0.1)

    # Current state
    initial_q = engine.q[node_idx, :4].clone()
    print(f"Initial Q: {initial_q}")

    # Apply holographic projection (which now includes gravity)
    engine.holographic_projection(target_vector=[0.0]*21, focus_intensity=1.0)

    # Check momentum
    momentum = engine.momentum[node_idx, :4]
    print(f"Momentum after gravity pull: {momentum}")

    # Expected: momentum should point towards target_vec
    # target_phys is [0.8, 0.2, -0.5, 0.1]
    # initial_q is [1, 0, 0, 0] approx (from get_or_create_node and pulse)

    diff = torch.tensor(target_vec[:4]) - initial_q
    print(f"Target Diff: {diff}")

    if torch.dot(momentum, diff) > 0:
        print("SUCCESS: Momentum points towards the target attractor!")
    else:
        print("FAILURE: Gravity pull not working as expected.")

if __name__ == "__main__":
    test_engine_gravity()
