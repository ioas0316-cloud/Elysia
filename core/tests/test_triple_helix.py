"""
[VERIFICATION: TRIPLE HELIX CROSS-DIMENSIONAL RESONANCE]
Verifies the Triple Helix Engine: inner/outer world propagation,
sensory feedback loop triggering mitosis/bifurcation, and Ohmic coordination learning.
"""

import os
import sys
import math
import numpy as np

# Ensure path resolution (workspace root is two levels up from core/tests/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.math_utils import Quaternion, Multivector
from core.triple_helix_engine import TripleHelixEngine

def test_cross_dimensional_propagation():
    """Verify that thought intention propagates to somatic layers."""
    engine = TripleHelixEngine()
    
    sensory = {"motion_entropy": 0.1, "pain_level": 0.1}
    tension, mode, jumped, quat, ennea = engine.pulse("세상의 조화를 꿈꾸는 평화로운 생각", sensory)
    
    assert isinstance(quat, Quaternion)
    assert abs(quat.norm() - 1.0) < 1e-5
    print("[SUCCESS] Cross-dimensional propagation verified.")

def test_sensory_feedback_bifurcation():
    """Verify that high sensory pain and dissonant thoughts trigger inner world bifurcation."""
    engine = TripleHelixEngine(jump_threshold=0.45)
    
    # 1. Normal state: Cl(3,0)
    assert engine.inner_world.signature == (3, 0)
    
    # 2. Inject extreme pain and highly dissonant thought to create high bridge tension
    sensory = {"motion_entropy": 0.1, "pain_level": 0.9}
    
    # Pulse multiple times to accumulate elastic stress and trigger bifurcation
    has_jumped = False
    for _ in range(5):
        tension, mode, jumped, quat, ennea = engine.pulse("카오스 물리 붕괴! 극심한 데이터 충격과 마찰 발생!", sensory)
        if jumped:
            has_jumped = True
    
    # Check if tension triggers bifurcation
    print(f"Tension on anomaly: {tension:.4f} | Has Jumped: {has_jumped}")
    print(f"New Inner World signature: {engine.inner_world.signature}")
    
    # It should have bifurcated to 4 dimensions
    assert engine.inner_world.signature == (4, 0)
    assert has_jumped is True
    print("[SUCCESS] Sensory feedback loop mitosis/bifurcation verified.")

def test_coordination_impedance_learning():
    """Verify that consistent thought and actions lower coordination link resistance."""
    engine = TripleHelixEngine()
    
    # Capture initial resistances
    r_wasd_init = engine.link_ego_wasd.R
    r_pain_init = engine.link_pain_h1.R
    
    # Pulse repeatedly with identical thought and sensory states to build coherence
    sensory = {"motion_entropy": 0.5, "pain_level": 0.5}
    for _ in range(10):
        engine.pulse("동일한 맥락의 평온한 생각의 반복", sensory)
        
    r_wasd_final = engine.link_ego_wasd.R
    r_pain_final = engine.link_pain_h1.R
    
    print(f"WASD Link Resistance: {r_wasd_init:.4f} -> {r_wasd_final:.4f}")
    print(f"Pain Link Resistance: {r_pain_init:.4f} -> {r_pain_final:.4f}")
    
    # Resistances should have decreased due to coherence learning
    assert r_wasd_final < r_wasd_init
    assert r_pain_final < r_pain_init
    print("[SUCCESS] Coordination link Ohmic learning verified.")

if __name__ == "__main__":
    test_cross_dimensional_propagation()
    test_sensory_feedback_bifurcation()
    test_coordination_impedance_learning()
    print("\n[PASS] ALL TRIPLE HELIX ENGINE TESTS PASSED!")
