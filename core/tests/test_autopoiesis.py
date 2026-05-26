"""
[VERIFICATION: AUTOPOIESIS EMERGENT HOMEOSTASIS & 4D HOLOGRAPHIC MEMORY]
Verifies the AutopoiesisController's emergent homeostasis sleep-wake cycles
driven by phase-coupling torque, and Bitwise4DHologramMemory's 4D coordinate resonance.
"""

import os
import sys
import math

# Ensure path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.autopoiesis_controller import AutopoiesisController
from core.holographic_memory import Bitwise4DHologramMemory

def test_autopoiesis_homeostasis_cycles():
    """Verify that AutopoiesisController transitions states emergent from tension and decays."""
    # Natural drift = 50, K = 300
    controller = AutopoiesisController(rotor_scale=4096, natural_drift=50.0, coupling_K=300.0)
    
    # 1. Start in Wake state
    assert controller.state_phase == 0
    assert controller.is_sleeping is False
    assert controller.sleep_factor < 0.1
    
    # Step under normal conditions (low tension = 0.05)
    # The phase should drift slowly but stay in the wake region [0, 1023] or [3073, 4095]
    for _ in range(5):
        controller.tick(network_tension=0.05, dt=0.1)
        
    assert controller.is_sleeping is False
    
    # 2. Simulate high workload / tension load (tension = 0.95)
    # The tension torque should pull the phase towards the sleep attractor (2048)
    for _ in range(30):
        controller.tick(network_tension=0.95, dt=0.3)
        
    # Eventually, it should transition to sleep state naturally
    print(f"Phase after high tension load: {controller.state_phase}, sleeping: {controller.is_sleeping}")
    assert controller.is_sleeping is True
    assert controller.sleep_factor > 0.5
    assert controller.get_connection_mode() == "Y_STAR"
    
    # 3. Simulate tension bleeding during sleep
    tension = 0.95
    for _ in range(10):
        tension = controller.bleed_tension(tension)
        controller.tick(network_tension=tension, dt=0.1)
        
    # Tension should have decayed significantly
    assert tension < 0.2
    
    # 4. Once tension is low, natural drift should rotate it back to wake state
    for _ in range(150):
        controller.tick(network_tension=0.02, dt=0.3)
        
    print(f"Phase after cooling down: {controller.state_phase}, sleeping: {controller.is_sleeping}")
    assert controller.is_sleeping is False
    assert controller.get_connection_mode() == "DELT\u0041" # DELTA
    
    print("[SUCCESS] Emergent sleep-wake homeostasis oscillation cycles verified.")

def test_bitwise_4d_hologram_memory():
    """Verify registration and O(1) multi-dimensional resonance in Bitwise4DHologramMemory."""
    mem = Bitwise4DHologramMemory(size_bits=64)
    
    # Register concept
    mask, (w, x, y, z) = mem.register_concept("quantum_homeostasis")
    
    # Verify coordinate boundaries
    for coord in (w, x, y, z):
        assert 0 <= coord < 64
        
    mem.superpose("quantum_homeostasis")
    
    # 1. Probe at exact coordinates (perfect resonance)
    res_exact = mem.scan_resonance(w, x, y, z)
    assert res_exact["quantum_homeostasis"] == 1.0
    
    # 2. Probe at slightly offset coordinates in w-axis (diff = 2)
    # w_offset = (w + 2) % 64. Res in w should be 1.0 - 2/8.0 = 0.75.
    # Total resonance = 0.75 * 1.0 * 1.0 * 1.0 = 0.75
    w_offset = (w + 2) % 64
    res_offset = mem.scan_resonance(w_offset, x, y, z)
    assert abs(res_offset["quantum_homeostasis"] - 0.75) < 1e-5
    
    # 3. Probe far away in z-axis (diff = 32)
    # Resonance should drop to 0.0 because of multiplication by 0.0 from z-axis
    z_far = (z + 32) % 64
    res_far = mem.scan_resonance(w, x, y, z_far)
    assert res_far["quantum_homeostasis"] == 0.0
    
    print("[SUCCESS] Bitwise4DHologramMemory 4D resonance verified.")

if __name__ == "__main__":
    test_autopoiesis_homeostasis_cycles()
    test_bitwise_4d_hologram_memory()
    print("\n[PASS] ALL AUTOPOIESIS AND 4D HOLOGRAM TESTS PASSED!")
