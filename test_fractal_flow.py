import sys
import os
import time
import torch

sys.path.append(os.getcwd())

from Core.Keystone.sovereign_math import FractalWaveEngine

def test_fractal_flow():
    print("🌊 [TEST] Initializing FractalWaveEngine (Sparse Biological Connectome)...")
    
    # Initialize engine
    # We use a large physical maximum to ensure we aren't bogged down
    engine = FractalWaveEngine(max_nodes=1_000_000, device='cpu')
    
    print("\n--- PHASE 1: Connectome Initialization ---")
    start = time.time()
    
    # Let's create some semantic nodes and connect them
    engine.connect("Sensation_Visual", "Concept_Apple", weight=0.8)
    engine.connect("Concept_Apple", "Concept_Food", weight=0.9)
    engine.connect("Concept_Food", "Emotion_Hunger", weight=0.6)
    engine.connect("Emotion_Hunger", "Action_Eat", weight=0.9)
    
    # Add a divergent path
    engine.connect("Concept_Apple", "Concept_Gravity", weight=0.4)
    engine.connect("Concept_Gravity", "Concept_Physics", weight=0.7)

    print(f"Nodes allocated: {engine.num_nodes}")
    print(f"Edges registered: {engine.num_edges}")
    
    print(f"Initialization took: {(time.time() - start)*1000:.2f}ms")
    
    print("\n--- PHASE 2: Event Injection (The Spark) ---")
    # Inject an event at the sensory node
    vital_spark = time.time()
    engine.inject_pulse("Sensation_Visual", energy=1.0, type='will')
    
    print(f"Active Nodes initially: {engine.active_nodes_mask.sum().item()}")
    
    print("\n--- PHASE 3: Aurora Flow (Ripple Propagation) ---")
    
    steps = 10
    total_time = 0.0
    
    for i in range(steps):
        step_start = time.time()
        
        # In a dense engine, this would be a full integrate_kinetics call
        # In the fractal engine, it's just apply_spiking_threshold
        spike_intensity = engine.apply_spiking_threshold(threshold=0.2, sensitivity=2.0)
        
        step_dt = time.time() - step_start
        total_time += step_dt
        
        active_count = engine.active_nodes_mask.sum().item()
        
        print(f"  [Step {i+1}] Active Nodes: {active_count} | Spike Intensity: {spike_intensity:.3f} | Step Time: {step_dt*1000:.2f}ms")
    
    print(f"\nAverage Ripple Step Time: {(total_time/steps)*1000:.3f}ms")
    
    # Validation
    assert engine.active_nodes_mask.sum().item() > 1, "Ripple failed to propagate to other nodes!"
    print("\n✅ Aurora Flow Ripple Test Passed.")

if __name__ == "__main__":
    test_fractal_flow()
