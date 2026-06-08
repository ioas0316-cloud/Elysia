import numpy as np
import pytest
from core.memory.bitmask_rotor_gate import BitmaskRotorGate

def test_pack_unpack_64bit():
    phase = np.uint32(0xDEADBEEF)
    token = np.uint32(0x12345678)
    packed = BitmaskRotorGate.pack_64bit(phase, token)
    u_phase, u_token = BitmaskRotorGate.unpack_64bit(packed)

    assert phase == u_phase, f"Phase mismatch: {phase} != {u_phase}"
    assert token == u_token, f"Token mismatch: {token} != {u_token}"

def test_create_mask():
    phase = np.uint32(0xF0F0F0F0)
    shift = 4
    mask = BitmaskRotorGate.create_mask(phase, shift)

    # Check if lower 32 bits are all 1s
    assert (mask & 0xFFFFFFFF) == 0xFFFFFFFF

    # Extract upper 32 bits of mask
    upper_mask = np.uint32(mask >> 32)
    expected_upper = (phase << shift) | (phase >> (32 - shift))
    assert upper_mask == expected_upper
