import numpy as np
import pytest
from core.memory.bitmask_rotor_gate import BitmaskRotorGate
from core.memory.spatiotemporal_trajectory_simulator import SpatiotemporalTrajectorySimulator

def test_bypass_trigger_fallback():
    # 수동 폴백 유도
    import core.memory.bitmask_rotor_gate
    core.memory.bitmask_rotor_gate.HAS_NUMBA = False

    gate = BitmaskRotorGate(matrix_dimension=3)
    gate.upload_to_device()

    input_wave = np.array([0xFFFFFFFFFFFFFFFF, 0x00000000000000FF, 0x0F0F0F0F0F0F0F0F], dtype=np.uint64)
    mask_tensor = np.array([0xFFFFFFFFFFFFFFFF, 0x00000000000000F0, 0x0000000000000000], dtype=np.uint64)
    output_ptr = np.zeros(3, dtype=np.uint64)

    gate.bypass_trigger(input_wave, mask_tensor, output_ptr)

    assert output_ptr[0] == 0x0
    assert output_ptr[1] == 0xFF
    assert output_ptr[2] == 0x0F0F0F0F0F0F0F0F

def test_spatiotemporal_bypass_integration():
    import core.memory.bitmask_rotor_gate
    core.memory.bitmask_rotor_gate.HAS_NUMBA = False

    canvas = np.array([0xAAAA, 0xBBBB, 0xCCCC], dtype=np.uint64)
    mask = np.array([0xFFFF, 0x0000, 0xCCCC], dtype=np.uint64)

    out = SpatiotemporalTrajectorySimulator.launch_wye_routing(canvas, mask)

    assert out[0] == 0x0 # AAAA & (~FFFF) = 0
    assert out[1] == 0xBBBB # BBBB & (~0000) = BBBB -> BBBB ^ 0000 = BBBB
    assert out[2] == 0x0 # CCCC & (~CCCC) = 0
