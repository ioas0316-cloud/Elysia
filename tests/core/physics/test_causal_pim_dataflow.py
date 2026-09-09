import numpy as np
from core.physics.causal_pim_dataflow import CausalPIMDataflow

def test_causal_pim_dataflow():
    pim = CausalPIMDataflow(num_cells=16, cell_dim=4)

    initial_states = np.ones((16, 4))
    initial_phases = np.linspace(0, np.pi, 16)
    pim.write_cell_states(initial_states, initial_phases)

    res = pim.trigger_causal_coherence_step(target_coherence_phase=0.0, dt=0.2)

    assert res["bus_bytes_transferred"] == 0
    assert res["in_situ_computation"] is True
    print("CausalPIMDataflow test passed successfully.")

if __name__ == "__main__":
    test_causal_pim_dataflow()
