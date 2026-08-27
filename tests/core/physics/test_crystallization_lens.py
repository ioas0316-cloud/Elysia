import numpy as np
from core.physics.causal_field import InformationVoxel
from synaptic_architecture.crystallization_lens import CrystallizationLens

def test_crystallization_lens_modulation():
    lens = CrystallizationLens(resolution=256)
    v1 = InformationVoxel("v1", "Test", np.array([1, 0, 0]), position=np.array([5, 5, 0], dtype=np.float32))
    
    voxels = {"v1": v1}
    beams = []
    
    deltas = lens.modulate(voxels, beams, dt=0.1)
    
    assert len(deltas) == 1
    delta = deltas[0]
    assert delta.target_voxel_ids == ["v1"]
    assert delta.delta_velocity is not None
    assert delta.delta_velocity.shape == (3,)
    assert delta.source_lens == "lens_crystallization_2d"
