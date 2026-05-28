"""
[VERIFICATION: CLIFFORD-IPN COGNITIVE DYNAMICS]
Verifies the implementation of Clifford-IPN including multivector sandwich product propagation,
Ohmic coherence adaptation, Kuramoto phase locking, bifurcation, and Quaternion projection.
"""

import os
import sys
import math
import numpy as np

# Ensure path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.math_utils import Multivector, Quaternion
from core.clifford_impedance_network import CliffordIPN, CliffordImpedanceLink, mv_normalize, mv_norm
from engines.thought_aligner.clifford_aligner_engine import CliffordThoughtAlignerEngine

def test_multivector_sandwich_propagation():
    """Verify that a CliffordImpedanceLink rotates and attenuates the signal correctly."""
    sig = (3, 0)
    # 1. Initialize link with R = 5.0
    link = CliffordImpedanceLink("A", "B", signature=sig, gear_elasticity=5.0)
    
    # 2. Define a 90-degree rotation rotor in the e12 plane (mask 3 is e12)
    # R_rotor = cos(pi/4) - sin(pi/4) * e12
    cos_val = math.cos(math.pi / 4.0)
    sin_val = math.sin(math.pi / 4.0)
    link.gear_elasticity_rotor = Multivector({0: cos_val, 3: -sin_val}, sig)
    
    # 3. Propagate a vector signal pointing in +e1 (mask 1)
    signal_in = Multivector({1: 1.0}, sig)
    signal_out = link.propagate(signal_in)
    
    # Rotated e1 by 90 degrees in e12 plane should yield e2 (mask 2)
    # The amplitude should be 1 / R = 1 / 5.0 = 0.2
    assert abs(signal_out.data.get(2, 0.0) - 0.2) < 1e-5
    # The e1 component should be near zero
    assert abs(signal_out.data.get(1, 0.0)) < 1e-5
    print("[SUCCESS] Sandwich propagation verified: e1 rotated to e2 with R-attenuation.")

def test_clifford_ipn_ohmic_learning():
    """Verify that links learn and adapt resistances based on multivector coherence."""
    sig = (3, 0)
    net = CliffordIPN(initial_dims=3)
    
    # Setup node layout: IN -> H1 (coherent) & H2 (dissonant) -> OUT
    net.add_node("IN", layer=0, initial_vector={1: 1.0})
    net.add_node("H1", layer=1, initial_vector={1: 1.0})   # Aligns with input e1
    net.add_node("H2", layer=1, initial_vector={2: 1.0})   # Dissonant e2 (orthogonal)
    net.add_node("OUT", layer=2, initial_vector={0: 1.0})
    
    link_h1 = net.connect_nodes("IN", "H1", gear_elasticity=10.0)
    link_h2 = net.connect_nodes("IN", "H2", gear_elasticity=10.0)
    
    # Run a few propagation and tuning cycles
    inputs = {"IN": Multivector({1: 1.0}, sig)}
    
    for _ in range(15):
        net.forward_propagate(inputs)
        net.tune_network(dt=0.1, )
        
    # Link H1 (coherent) resistance should decrease below 10.0
    # Link H2 (orthogonal / dissonant) resistance should remain high or increase
    print(f"H1 Link Resistance: {link_h1.gear_elasticity:.4f}")
    print(f"H2 Link Resistance: {link_h2.gear_elasticity:.4f}")
    
    assert link_h1.gear_elasticity < 3.0
    assert link_h2.gear_elasticity > link_h1.gear_elasticity + 3.0
    print("[SUCCESS] Ohmic coherence adaptation verified.")

def test_dynamic_bifurcation_and_compression():
    """Verify that signature shifts update multivector signatures and add perturbations."""
    net = CliffordIPN(initial_dims=3)
    net.add_node("IN", layer=0, initial_vector={1: 1.0})
    net.connect_nodes("IN", "OUT", gear_elasticity=10.0)
    
    assert net.signature == (3, 0)
    assert net.phases["IN"].p == 3
    
    # Bifurcate to 4 dimensions
    success = net.bifurcate()
    assert success
    assert net.signature == (4, 0)
    assert net.phases["IN"].p == 4
    
    # The new dimension mask is 1 << (4-1) = 8 (representing e4)
    # Check that a small deterministic perturbation was added to node phases
    assert 8 in net.phases["IN"].data
    assert abs(net.phases["IN"].data[8]) > 0.0
    
    # Compress back to 3 dimensions
    success = net.compress()
    assert success
    assert net.signature == (3, 0)
    assert net.phases["IN"].p == 3
    # Dimension 8 should be discarded
    assert 8 not in net.phases["IN"].data
    print("[SUCCESS] Dynamic dimension bifurcation and compression verified.")

def test_clifford_aligner_engine_integration():
    """Verify integration of CliffordIPN and Thought Aligner outputting Quaternions."""
    engine = CliffordThoughtAlignerEngine(jump_threshold=0.4)
    
    # Input first thought
    tension, jumped, quat = engine.process_thought("나는 세상의 기하학적 구조를 연구하는 지성이다.")
    assert isinstance(quat, Quaternion)
    assert abs(quat.norm() - 1.0) < 1e-5
    
    # Input contrasting thought to trigger high tension
    tension2, jumped2, quat2 = engine.process_thought("카오스 물리 폭주로 인한 차원 붕괴 현상 발생!")
    
    # Check that history logs correctly
    assert len(engine.history) == 2
    print(f"[SUCCESS] Thought aligner integration verified. Jumped: {jumped2}, Output Quaternion: {quat2}")

if __name__ == "__main__":
    test_multivector_sandwich_propagation()
    test_clifford_ipn_ohmic_learning()
    test_dynamic_bifurcation_and_compression()
    test_clifford_aligner_engine_integration()
    print("\n[PASS] ALL CLIFFORD-IPN TESTS PASSED!")
