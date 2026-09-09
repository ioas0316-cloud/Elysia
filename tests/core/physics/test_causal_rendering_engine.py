import numpy as np
from core.physics.causal_rendering_engine import CausalRenderingEngine

def test_causal_rendering_engine():
    renderer = CausalRenderingEngine(observer_pos=[0, 0, 0], observer_sight_axis=[0, 0, 1])

    scene = [
        {"id": "front_obj", "position": [0, 0, 10]},
        {"id": "behind_obj", "position": [0, 0, -10]},
        {"id": "side_obj", "position": [-10, 0, 0]},
        {"id": "front_right_obj", "position": [5, 0, 5]}
    ]

    result = renderer.extract_phase_boundary_tensor(scene)

    assert result["overdraw_ratio"] == 0.0
    assert result["rendered_primitives"] == 2  # front_obj and front_right_obj
    assert result["bypassed_primitives"] == 2  # behind_obj and side_obj
    assert result["ray_bounce_evaluations"] == 0
    print("CausalRenderingEngine test passed successfully.")

if __name__ == "__main__":
    test_causal_rendering_engine()
