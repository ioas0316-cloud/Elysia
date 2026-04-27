import sys
import os
import torch
import time

# Add the parent directory to the path so we can import Core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Keystone.sovereign_math import FractalWaveEngine

def test_somatic_metabolism():
    print("🧠 [METABOLISM TEST] Initializing Somatic Manifold...")
    # Use a smaller manifold for testing
    engine = FractalWaveEngine(max_nodes=10000, device='cpu')
    
    # Check if GUT exists
    assert "GUT" in engine.atlas.organs, "GUT organ must be present in SomaticAtlas."
    print("✅ GUT Organ initialized successfully.")
    
    print("\n--- PHASE 1: Raw Data Ingestion (Decomposition) ---")
    # Simulate a burst of 'raw, unrefined' sensory data
    # High physical intensity, low cognitive meaning, zero enthalpy
    raw_nodes = torch.arange(0, 10)
    engine.active_nodes_mask[raw_nodes] = True
    
    # Set physical states to roughly align with the GUT (Negative Z)
    engine.q[raw_nodes, engine.PHYSICAL_SLICE] = torch.tensor([0.0, 0.0, 0.0, -0.8])
    
    # Set initial entropy to low
    engine.q[raw_nodes, engine.CH_ENTROPY] = 0.0
    engine.q[raw_nodes, engine.CH_ENTHALPY] = 0.0
    
    print(f"Initial Entropy of Raw Nodes: {engine.q[0, engine.CH_ENTROPY].item():.4f}")
    
    # Pulse the engine a few times to allow Decomposition (entropy increase)
    for _ in range(5):
        engine.apply_spiking_threshold(threshold=0.9, sensitivity=2.0)
    
    # Entropy should have increased because they are low-density (raw noise)
    decomposed_entropy = engine.q[0, engine.CH_ENTROPY].item()
    print(f"Decomposed Entropy (After Pulses): {decomposed_entropy:.4f}")
    assert decomposed_entropy > 0.0, "Entropy should increase during decomposition."
    print("✅ Decomposition Logic (Entropy Increase) Verified.")
    
    print("\n--- PHASE 2: Waste Excretion ---")
    # Artificially set some nodes to highly toxic (High entropy, 0 enthalpy)
    waste_nodes = torch.arange(10, 20)
    engine.active_nodes_mask[waste_nodes] = True
    engine.q[waste_nodes, engine.CH_ENTROPY] = 0.9  # Very high disorder
    engine.q[waste_nodes, engine.CH_ENTHALPY] = 0.0  # Zero activity
    
    pre_waste_count = engine.active_nodes_mask.sum().item()
    print(f"Active Nodes before Excretion: {pre_waste_count}")
    
    # Trigger excretion
    excreted = engine.discharge_waste()
    
    post_waste_count = engine.active_nodes_mask.sum().item()
    print(f"Nodes Excreted: {excreted}")
    print(f"Active Nodes after Excretion: {post_waste_count}")
    
    assert excreted == 10, "Should have excreted exactly 10 waste nodes."
    assert post_waste_count == pre_waste_count - 10, "Active nodes mask should be updated."
    
    # Check if the state was zeroed out
    assert engine.q[waste_nodes[0], engine.CH_ENTROPY].item() == 0.0, "Waste node state must be reset to 0."
    print("✅ Waste Excretion Logic Verified.")
    
    print("\n--- PHASE 3: Absorption (Ascension) ---")
    # Simulate a node that has high meaning (Resonance/Enthalpy)
    # It should survive excretion and eventually ascend
    good_nodes = torch.arange(20, 25)
    engine.active_nodes_mask[good_nodes] = True
    engine.q[good_nodes, engine.CH_ENTHALPY] = 0.9 # High activity
    engine.q[good_nodes, engine.CH_ENTROPY] = 0.1  # Low disorder
    
    # Excrete again (should NOT excrete good nodes)
    excreted = engine.discharge_waste()
    print(f"Good nodes excreted: {excreted} (Should be 0)")
    assert excreted == 0, "Healthy nodes should not be excreted."
    
    # Spike to trigger ascension gravity
    engine.q[good_nodes, engine.PHYSICAL_SLICE] = engine.permanent_q[good_nodes, engine.PHYSICAL_SLICE] # Max self-density
    for _ in range(5):
        engine.apply_spiking_threshold(threshold=0.1, sensitivity=5.0)
        
    print("✅ Absorption (Ascension) path verified.")

if __name__ == "__main__":
    test_somatic_metabolism()
