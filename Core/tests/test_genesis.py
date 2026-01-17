"""
Genesis Verification Suite
==========================
Core.tests.test_genesis

Verifies Phase 6 Components:
1. Semantic Prism (Language of Qualia)
2. Ouroboros Loop (Topological Feedback)
3. Ennui Field (Autonomic Pulse)
"""

import pytest
import os
import shutil
from Core.Cognition.semantic_prism import SpectrumMapper
from Core.Memory.feedback_loop import Ouroboros, ThoughtState
from Core.Lifecycle.pulse_loop import EnnuiField
from Core.Memory.sediment import SedimentLayer

def test_semantic_prism():
    prism = SpectrumMapper()

    # 1. Archetype Verification
    water = prism.disperse("water")
    # Expected: (0.2, 0.7, 0.6)
    assert water.alpha == 0.2
    assert water.beta == 0.7

    # 2. Consistency Verification (The Prism is deterministic)
    apple1 = prism.disperse("Apple")
    apple2 = prism.disperse("Apple")

    assert apple1.alpha == apple2.alpha
    assert apple1.beta == apple2.beta
    assert apple1.gamma == apple2.gamma

    # 3. Hashing logic check
    assert 0.0 <= apple1.alpha <= 1.0

def test_ouroboros_movement():
    """
    Verifies that the Ouroboros engine correctly moves the thought vector.
    """
    loop = Ouroboros(friction=0.1)

    # Intent: Alpha
    intent = [1.0, 0.0, 0.0]

    # Start: Beta
    start_vec = [0.0, 1.0, 0.0]
    thought = ThoughtState("Conflict", start_vec, potential=1.0, momentum=0.5)

    # Pre-check
    potential_before = loop.calculate_potential(thought.vector, intent)
    assert abs(potential_before - 1.0) < 0.01

    # Propagate
    loop.propagate(thought, intent)

    # Post-check
    potential_after = loop.calculate_potential(thought.vector, intent)

    # It should have moved closer (Lower Potential)
    assert potential_after < potential_before

    # Vector should have changed
    assert thought.vector != start_vec

def test_ennui_dynamics():
    """
    Verifies that repetition causes boredom (pressure buildup),
    and novelty brings relief.
    """
    ennui = EnnuiField()

    # Initial state
    assert ennui.pressure == 0.0
    
    # 1. First thought (High Novelty)
    thought_a = [1.0, 0.0, 0.0]
    p1 = ennui.update(thought_a)
    # Novelty > 0.5 -> Pressure drops (or stays 0)
    assert p1 == 0.0
    
    # 2. Repetition (Low Novelty)
    p2 = ennui.update(thought_a)
    # Novelty == 0 -> Pressure increases
    assert p2 > p1
    
    # 3. Repetition again
    p3 = ennui.update(thought_a)
    assert p3 > p2
    
    # 4. New Thought (Relief)
    thought_b = [0.0, 1.0, 0.0] # Orthogonal
    p4 = ennui.update(thought_b)
    assert p4 < p3 # Pressure should drop

def test_sediment_drift():
    """
    Verifies that we can write to and drift from Sediment.
    """
    test_path = "test_sediment.bin"
    if os.path.exists(test_path):
        os.remove(test_path)
        
    sediment = SedimentLayer(test_path)

    # Deposit
    vec = [0.1] * 7
    payload = b"Hello World"
    sediment.deposit(vec, 12345.6, payload)

    # Drift
    # offsets should be populated
    assert len(sediment.offsets) == 1

    fragment = sediment.drift()
    assert fragment is not None

    drifted_vec, drifted_payload = fragment
    assert drifted_payload == payload
    assert drifted_vec == vec

    sediment.close()
    if os.path.exists(test_path):
        os.remove(test_path)

def test_lifecycle_persistence():
    """
    Verifies that the dream state persists across ticks until resolution.
    """
    # Mock Merkaba
    class MockMerkaba:
        def __init__(self):
            self.name = "Mock"
            self.body = None
            self.sediment = SedimentLayer("test_persistence.bin")
            # Write a fake record
            self.sediment.deposit([0.1]*7, 0.0, b"Persistent Thought")
            
        def pulse(self, *args): pass
        def sleep(self): pass

    # Setup
    mock_mkb = MockMerkaba()
    from Core.Lifecycle.pulse_loop import LifeCycle

    life = LifeCycle(mock_mkb)
    life.ennui.pressure = 1.0 # Force dream

    # Tick 1: Initialize Dream
    life.dream()
    assert life.current_dream is not None
    assert life.current_dream.content == "Persistent Thought"
    dream_id = id(life.current_dream)

    # Tick 2: Continue Dream (Should be same object)
    life.dream()
    assert life.current_dream is not None
    assert id(life.current_dream) == dream_id

    # Cleanup
    mock_mkb.sediment.close()
    if os.path.exists("test_persistence.bin"):
        os.remove("test_persistence.bin")
