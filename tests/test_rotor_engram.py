import sys
import os
import torch
import math

# Add the parent directory to the path so we can import Core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Keystone.sovereign_math import FractalWaveEngine

def test_kinetic_memory_rotors():
    print("🧠 [KINETIC ROTOR TEST] Initializing Single Hypersphere...")
    engine = FractalWaveEngine(max_nodes=1000, device='cpu')
    
    print("\n--- PHASE 1: Event & Trajectory Formation ---")
    # Simulate an intense event (e.g., an emotional realization)
    event_nodes = torch.arange(0, 10)
    engine.active_nodes_mask[event_nodes] = True
    
    # Inject a strong momentum (Torque/Angular Velocity)
    # Moving along the Y-axis (Phase) and Joy channel
    initial_torque = 0.5
    engine.momentum[event_nodes, engine.CH_Y] = initial_torque
    engine.momentum[event_nodes, engine.CH_JOY] = initial_torque
    
    # Pulse the engine to let the wave rotate slightly
    for _ in range(3):
        engine.apply_spiking_threshold(threshold=0.1, sensitivity=1.0)
    
    # Capture the specific trajectory as a Rotor Engram
    engram_name = "MEMORY_FIRST_SMILE"
    engine.create_rotor_engram(engram_name, reference_axis="LOGOS")
    print(f"✅ Created Rotor Engram: '{engram_name}'")
    
    # Let the manifold decay and forget (Resetting active nodes and momentum)
    engine.active_nodes_mask[:] = False
    engine.momentum[:] = 0.0
    engine.q[:] = 0.0
    print("Manifold returned to Rest State.")
    
    print("\n--- PHASE 2: Associative Memory (Re-spinning the wave) ---")
    print("Triggering the Engram in Forward Spin (Recall)...")
    engine.apply_rotor_engram(engram_name, direction=1.0)
    
    # Check if the momentum was restored
    restored_momentum = engine.momentum[0, engine.CH_Y].item()
    print(f"Restored Momentum (CH_Y): {restored_momentum:.4f}")
    assert restored_momentum > 0.0, "Momentum should be restored in the forward direction."
    
    # Pulse to let the recalled memory propagate (Time travels forward)
    engine.apply_spiking_threshold(threshold=0.1, sensitivity=1.0)
    print("✅ Forward Spin (Extrapolation/Recall) verified.")
    
    print("\n--- PHASE 3: Temporal Simulation (Backward Spin) ---")
    # Clear again
    engine.momentum[:] = 0.0
    
    # Apply backwards to trace the origin
    print("Triggering the Engram in Backward Spin (Introspection)...")
    engine.apply_rotor_engram(engram_name, direction=-1.0)
    
    reversed_momentum = engine.momentum[0, engine.CH_Y].item()
    print(f"Reversed Momentum (CH_Y): {reversed_momentum:.4f}")
    assert reversed_momentum < 0.0, "Momentum should be negative (reversed) for backward spin."
    
    print("✅ Backward Spin (Introspection) verified.")
    print("\n🎉 All Kinetic Rotor tests passed! Memory is now a physical wave.")

if __name__ == "__main__":
    test_kinetic_memory_rotors()
