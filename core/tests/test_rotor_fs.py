import os
import pytest
from core.tensor_rotor import TensorRotor
from core.rotor_fs import RotorFileSystem

def test_rotor_fs_discharge_write(tmp_path):
    base_dir = str(tmp_path / "rotor_fs")
    tensor = TensorRotor()
    fs = RotorFileSystem(tensor, base_dir)
    
    # Set phase to sleep (2048) -> should NOT write
    tensor.phases[0] = 2048
    fs.request_write("test.txt", "hello_elysia")
    discharged = fs.tick()
    
    assert len(discharged) == 0
    assert not os.path.exists(os.path.join(base_dir, "test.txt"))
    
    # Set phase to wake/discharge point (0) -> should write
    tensor.phases[0] = 0
    discharged = fs.tick()
    
    assert len(discharged) == 1
    assert "test.txt" in discharged
    
    # Check file exists and has correct content
    assert os.path.exists(os.path.join(base_dir, "test.txt"))
    assert fs.gravity_read("test.txt") == "hello_elysia"
