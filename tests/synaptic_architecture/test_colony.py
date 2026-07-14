import numpy as np
from synaptic_architecture.colony import ResonantColony

def test_colony_init():
    colony = ResonantColony(num_initial_cells=2)
    assert len(colony.cells) == 2
    assert colony.coupling.shape == (2, 2)

def test_colony_pulse():
    colony = ResonantColony(num_initial_cells=2, resolution=16)
    cell_id = colony.cell_ids[0]
    stim = {cell_id: np.array([8, 8, 1.0])}
    colony.pulse_colony(stim)
    assert colony.cells[cell_id].activation[8, 8] > 0
