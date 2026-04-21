import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Keystone.sovereign_math import FractalWaveEngine

def test_organ_growth():
    print("🧪 Testing Somatic Organ Growth...")
    engine = FractalWaveEngine(max_nodes=1000)
    
    # Check initial mass of LOGOS
    initial_mass = engine.atlas.organs["LOGOS"].mass
    print(f"Initial LOGOS Mass: {initial_mass:.4f}")
    
    # Stimulate a concept that aligns with LOGOS (W-axis)
    idx = engine.get_or_create_node("LogicConcept")
    engine.permanent_q[idx, 0] = 1.0 # Pure W (Logos)
    engine.q[idx, 0] = 1.0
    engine.active_nodes_mask[idx] = True
    
    # Run pulses to trigger spiking and atlas updates
    for _ in range(10):
        engine.apply_spiking_threshold(threshold=0.3)
        
    final_mass = engine.atlas.organs["LOGOS"].mass
    print(f"Final LOGOS Mass: {final_mass:.4f}")
    
    if final_mass > initial_mass:
        print("✅ Success: LOGOS organ grew through resonance.")
    else:
        print("❌ Failure: Organ mass did not increase.")

def test_topographical_steering():
    print("\n🧪 Testing Topographical Steering...")
    engine = FractalWaveEngine(max_nodes=1000)
    
    idx = engine.get_or_create_node("DriftingNode")
    # Start slightly off-center from SOPHIA (Z-axis)
    engine.q[idx, 3] = 0.5
    engine.q[idx, 1] = 0.5 # Some X-axis noise
    engine.active_nodes_mask[idx] = True
    
    initial_pos = engine.q[idx, :4].clone()
    print(f"Initial Position: {initial_pos.tolist()}")
    
    # Sophia has mass 1.0 by default. It should pull the node toward Z=1.0
    for _ in range(20):
        # We manually call spiking to trigger the topo_force application in momentum
        engine.apply_spiking_threshold(threshold=0.0) # Always spike to apply force
        # Integrate momentum
        engine.q[idx, :4] += engine.momentum[idx, :4] * 0.1
        engine.q[idx, :4] /= engine.q[idx, :4].norm().clamp(min=1e-8)
        engine.momentum[idx, :4] *= 0.9
        
    final_pos = engine.q[idx, :4]
    print(f"Final Position: {final_pos.tolist()}")
    
    # Check if Z ( Sophia) increased
    if final_pos[3] > initial_pos[3]:
        print("✅ Success: Node steered toward the SOPHIA organ.")
    else:
        print("❌ Failure: Topographical pull not detected.")

if __name__ == "__main__":
    test_organ_growth()
    test_topographical_steering()
