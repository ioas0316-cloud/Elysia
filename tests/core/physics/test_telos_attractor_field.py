import numpy as np
from core.physics.telos_attractor_field import TelosAttractorField

def test_telos_attractor_field():
    field = TelosAttractorField(dim=4)
    telos = np.array([1.0, 2.0, 3.0, 4.0])
    field.set_telos(telos)

    start_state = np.array([10.0, -5.0, 8.0, 0.0])
    res = field.evaluate_flow_trajectory(start_state, max_steps=200)

    assert res["if_branch_evaluations"] == 0
    assert np.allclose(res["final_state"], telos, atol=1e-2)
    print("TelosAttractorField test passed successfully.")

if __name__ == "__main__":
    test_telos_attractor_field()
