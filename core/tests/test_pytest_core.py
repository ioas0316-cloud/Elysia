import os
import sys
import pytest

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.utils.math_utils import Quaternion, traverse_causal_trajectory
from core.brain.tri_phase_transistor import TriPhaseTransistor
from core.brain.holographic_memory import HologramMemory

def test_quaternion_math():
    # Test Quaternion instantiation and normalization
    q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
    assert q1.w == 1.0
    assert q1.x == 2.0
    assert q1.y == 3.0
    assert q1.z == 4.0
    
    n = q1.norm()
    assert abs(n - (1**2 + 2**2 + 3**2 + 4**2)**0.5) < 1e-9
    
    q1_normalized = q1.normalize()
    assert abs(q1_normalized.norm() - 1.0) < 1e-6
    
    # Test Conjugate
    q1_conj = q1.conjugate()
    assert q1_conj.w == 1.0
    assert q1_conj.x == -2.0
    assert q1_conj.y == -3.0
    assert q1_conj.z == -4.0
    
    # Test Multiplication (Hamilton product)
    qi = Quaternion(0.0, 1.0, 0.0, 0.0)
    qj = Quaternion(0.0, 0.0, 1.0, 0.0)
    
    # i * j = k
    qi_qj = qi * qj
    assert abs(qi_qj.w) < 1e-9
    assert abs(qi_qj.x) < 1e-9
    assert abs(qi_qj.y) < 1e-9
    assert abs(qi_qj.z - 1.0) < 1e-9
    
    # Distance
    dist = Quaternion.distance(qi, qj)
    # qi . qj = 0, distance = 2 * acos(0) = pi
    assert abs(dist - 3.1415926535) < 1e-5
    
    # Slerp
    qslerp = Quaternion.slerp(qi, qj, 0.5)
    # Halfway should have w=0, x=1/sqrt(2), y=1/sqrt(2), z=0
    assert abs(qslerp.norm() - 1.0) < 1e-6
    assert abs(qslerp.x - 0.70710678) < 1e-5
    assert abs(qslerp.y - 0.70710678) < 1e-5

def test_tri_phase_transistor():
    axis = Quaternion(1.0, 0.0, 0.0, 0.0)
    trans = TriPhaseTransistor(axis)
    
    # Process a wave
    cause = Quaternion(0.0, 1.0, 0.0, 0.0)
    result = trans.process_wave(cause)
    
    # result = axis * cause * axis_inv = cause (since axis is identity)
    assert abs(result.w) < 1e-9
    assert abs(result.x - 1.0) < 1e-9
    
    # check tension calculation
    # dot(cause, axis) = dot([0, 1, 0, 0], [1, 0, 0, 0]) = 0
    # current_dissonance = 1 - 0 = 1
    # trapped_tension_magnitude should accumulate
    assert trans.trapped_tension_magnitude > 0.0

def test_hologram_memory_lifecycle(tmp_path):
    memory = HologramMemory(num_layers=2)
    concept = "가방"
    
    # Test registration
    success = memory.register_concept(concept)
    assert success is True
    
    # Save memory to temp path
    filepath = str(tmp_path / "test_memory.json")
    memory.save_to_disk(filepath)
    assert os.path.exists(filepath)
    
    # Load memory from temp path
    new_memory = HologramMemory(num_layers=2)
    loaded = new_memory.load_from_disk(filepath)
    assert loaded is True
    assert concept in new_memory.ui_concept_map
    
    # Check that reconstructed node state matches original
    orig_node = memory.ui_concept_map[concept]
    loaded_node = new_memory.ui_concept_map[concept]
    assert abs(orig_node.state.w - loaded_node.state.w) < 1e-6
    assert abs(orig_node.state.x - loaded_node.state.x) < 1e-6
    assert abs(orig_node.state.y - loaded_node.state.y) < 1e-6
    assert abs(orig_node.state.z - loaded_node.state.z) < 1e-6
