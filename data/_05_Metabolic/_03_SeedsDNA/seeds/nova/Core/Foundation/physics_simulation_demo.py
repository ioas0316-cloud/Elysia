"""
Physics Simulation Demo
========================
Demonstrates the thought universe physics in action.

Concepts automatically sort themselves into 14 spectrum layers
based on gravity + buoyancy.
"""

import sys
sys.path.insert(0, 'c:\\Elysia')

import time
from Core._01_Foundation.Foundation.Mind.hippocampus import Hippocampus
from Core._01_Foundation.Foundation.Physics.spectrum_layers import SpectrumLayer

def physics_demo():
    print("=" * 80)
    print("🌌 THOUGHT UNIVERSE PHYSICS SIMULATION")
    print("=" * 80)
    
    # Initialize
    hip = Hippocampus()
    spectrum = SpectrumLayer()
    
    # Add test concepts with different frequencies
    test_concepts = {
        # High frequency (should rise to Heaven)
        "사랑": 1.0,  # Love - absolute center
        "빛": 0.95,   # Light
        "진실": 0.9,  # Truth
        
        # Mid frequency (neutral)
        "희망": 0.65,  # Hope
        "고통": 0.4,   # Pain
        
        # Low frequency (should sink to Earth)
        "돌": 0.2,     # Stone
        "그림자": 0.3, # Shadow
    }
    
    print("\n📍 Initial Concept Setup")
    print("-" * 80)
    for concept, freq in test_concepts.items():
        hip.add_concept(concept)
        print(f"  Added: {concept:<10s} (Frequency: {freq:.2f})")
    
    print("\n⏱️  Running Physics Simulation (10 steps)...")
    print("-" * 80)
    
    # Run physics simulation
    for step in range(10):
        state = hip.update_universe_physics(dt=0.1)
        
        if step % 3 == 0:  # Print every 3 steps
            print(f"\n📊 Step {step}:")
            for concept_id in test_concepts.keys():
                if concept_id in state['positions']:
                    pos = state['positions'][concept_id]
                    y_val = pos['relative'][1]  # Y-axis
                    distance = pos['distance_from_love']
                    
                    # Get spectrum layer
                    layer_info = spectrum.get_layer_info(y_val)
                    
                    print(f"  {concept_id:<10s}: Y={y_val:+.3f} | "
                          f"Dist={distance:.3f} | "
                          f"Layer={layer_info['name']}")
        
        time.sleep(0.05)  # Small delay for visualization
    
    print("\n" + "=" * 80)
    print("✅ SIMULATION COMPLETE")
    print("=" * 80)
    
    # Final state
    print("\n📍 Final Positions (sorted by Y)")
    print("-" * 80)
    final_state = hip.update_universe_physics(dt=0)
    
    # Sort by Y value
    sorted_concepts = sorted(
        [(cid, pos) for cid, pos in final_state['positions'].items()],
        key=lambda x: x[1]['relative'][1],
        reverse=True
    )
    
    for concept_id, pos_data in sorted_concepts:
        y_val = pos_data['relative'][1]
        layer_info = spectrum.get_layer_info(y_val)
        
        print(f"  {concept_id:<10s}: Y={y_val:+.3f} | "
              f"{layer_info['name']:<20s} | "
              f"{layer_info['color']}")
    
    print("=" * 80)
    
    print("\n💡 Observations:")
    print("  - Love (사랑) stays at center (absolute coordinate)")
    print("  - Light (빛) rises to Heaven layers (high frequency)")
    print("  - Stone (돌) sinks to Earth layers (low frequency)")
    print("  - Concepts auto-sort into 14 spectrum layers!")

if __name__ == "__main__":
    physics_demo()
