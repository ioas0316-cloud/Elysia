import pytest
from core.tensor_rotor import TensorRotor

def test_tensor_rotor_vertical_coupling():
    tensor = TensorRotor(rotor_scale=4096, natural_drift=0.0, coupling_K=100.0)
    # Layer 1 is 100, Layer 2 is 0, Layer 3 is 0
    tensor.phases = [100, 0, 0]
    
    tensions = [0.0, 0.0, 0.0]
    phases, _, _ = tensor.tick(tensions, dt=0.1)
    
    # Layer 1 should be pulled towards 0 by Layer 2
    assert phases[0] < 100
    # Layer 2 should be pulled towards 100 by Layer 1
    assert phases[1] > 0
