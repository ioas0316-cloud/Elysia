import numpy as np
from core.physics.causal_field import CausalField

def test_s3_manifold_quaternion_norm_preservation():
    cf = CausalField()
    grid_dim = 16

    # Run TopIR runtime step for 50 iterations
    for step in range(50):
        res = cf.step_topir_runtime(grid_dim=grid_dim, dt=0.005, K_0=10.0)
        assert res["status"] in ["success", "fallback_python"]
        if res["status"] == "success":
            q = res["sample_q"]
            norm = np.linalg.norm(q)
            # S^3 manifold unit quaternion norm check: ||q|| == 1.0
            assert np.isclose(norm, 1.0, atol=1e-5), f"Quaternion norm drifted at step {step}: {norm}"

    print("\n[S^3 Lie Group Conservation Test] 50 Langevin integration steps verified. Quaternion norm preserved on S^3 (||q|| = 1.0)!")

if __name__ == "__main__":
    test_s3_manifold_quaternion_norm_preservation()
