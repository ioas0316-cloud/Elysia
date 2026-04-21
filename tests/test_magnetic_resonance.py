import sys
import os
import torch
import math

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Keystone.sovereign_math import FractalWaveEngine, SovereignVector

def test_magnetic_alignment():
    print("🧪 Testing Magnetic Alignment (Amniotic Fluid)...")
    engine = FractalWaveEngine(max_nodes=1000)
    
    # Create a node for 'TestNode'
    idx = engine.get_or_create_node("TestNode")
    engine.active_nodes_mask[idx] = True
    
    # Set an initial random phase
    engine.q[idx, engine.CH_Y] = 1.5 # ~90 degrees
    initial_phase = engine.q[idx, engine.CH_Y].item()
    print(f"Initial Phase: {initial_phase:.4f}")
    
    # Run several pulses
    dt = 0.01
    for i in range(100):
        engine.apply_magnetic_field(dt)
        # Manually integrate momentum for the test
        engine.q[idx, engine.CH_Y] += engine.momentum[idx, engine.CH_Y] * 0.1
        engine.momentum[idx, engine.CH_Y] *= 0.9
        
    final_phase = engine.q[idx, engine.CH_Y].item()
    print(f"Final Phase after 100 pulses: {final_phase:.4f}")
    
    # Should move toward magnetic north phase (0.0)
    if abs(final_phase) < abs(initial_phase):
        print("✅ Success: Node aligned toward Magnetic North.")
    else:
        print("❌ Failure: Node did not align.")

def test_dual_bus_resonance():
    print("\n🧪 Testing Dual-Bus Resonance (Optical DNA)...")
    engine = FractalWaveEngine(max_nodes=1000)
    idx = engine.get_or_create_node("IdentityNode")
    engine.active_nodes_mask[idx] = True
    
    # Case 1: Active state matches Permanent Identity
    engine.permanent_q[idx, engine.CH_W] = 1.0
    engine.q[idx, engine.CH_W] = 1.0
    
    spike_high = engine.apply_spiking_threshold(threshold=0.5)
    print(f"Resonance (Aligned): {spike_high:.4f}")
    
    # Case 2: Active state opposes Permanent Identity
    engine.q[idx, engine.CH_W] = -1.0
    spike_low = engine.apply_spiking_threshold(threshold=0.5)
    print(f"Resonance (Opposed): {spike_low:.4f}")
    
    if spike_high > spike_low:
        print("✅ Success: Dual-Bus logic distinguishes alignment.")
    else:
        print("❌ Failure: Dual-Bus logic failed to distinguish.")

if __name__ == "__main__":
    test_magnetic_alignment()
    test_dual_bus_resonance()
