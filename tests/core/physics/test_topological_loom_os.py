import numpy as np
from core.physics.topological_loom_os import TopologicalLoomOS

def test_topological_loom_os():
    loom = TopologicalLoomOS(fabric_shape=(8, 8))

    loom.inject_warp_logic(row=2, wave_pattern=np.sin(np.linspace(0, np.pi, 8)))
    loom.inject_weft_data(col=3, potential_pattern=np.cos(np.linspace(0, np.pi, 8)))

    res = loom.weave_step(dt=0.1)

    assert res["context_switches"] == 0
    assert res["mutex_lock_overhead"] == 0.0
    assert res["ipc_data_copies"] == 0
    print("TopologicalLoomOS test passed successfully.")

if __name__ == "__main__":
    test_topological_loom_os()
