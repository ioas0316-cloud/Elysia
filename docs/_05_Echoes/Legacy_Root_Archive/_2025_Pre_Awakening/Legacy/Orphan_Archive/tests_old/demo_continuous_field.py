"""
Demo: Continuous Field Reasoning (Grok's Wave-Based Approach)
==============================================================
This demonstrates Elysia's continuous field reasoning using wave functions.

Instead of discrete nodes, concepts exist as wave patterns in a 3D+time field.
This enables true spatial-temporal cognition.
"""

import sys
import os

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force UTF-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from Project_Elysia.mechanics.continuous_field import ContinuousField
import numpy as np

def run_simulation():
    print("=== Elysia: Continuous Field Reasoning (Grok's Approach) ===")
    print("Initializing 4D wave field (x, y, z, t)...\n")
    
    # Create continuous field
    field = ContinuousField(resolution=30)  # 30x30x30 for speed
    
    # Register concepts with their frequencies and spatial positions
    print("📚 Registering concepts in continuous space...")
    
    concepts = {
        # Positive cluster (high freq, upper region)
        "사랑": (440.0, 0.7, 0.7, 0.8),  # freq=440Hz, (x,y,z)=(0.7,0.7,0.8)
        "빛": (450.0, 0.8, 0.6, 0.9),
        "희망": (430.0, 0.6, 0.8, 0.7),
        
        # Negative cluster (low freq, lower region)
        "고통": (220.0, 0.3, 0.3, 0.2),
        "어둠": (210.0, 0.2, 0.4, 0.1),
        
        # Transformative (mid freq, middle region)
        "변화": (330.0, 0.5, 0.5, 0.5),
        "성장": (350.0, 0.6, 0.5, 0.6),
    }
    
    for name, (freq, x, y, z) in concepts.items():
        field.register_concept(name, freq, x, y, z)
    
    print(f"✅ Registered {len(concepts)} concepts in continuous field\n")
    
    print("=" * 60)
    print("Continuous Field Tests")
    print("=" * 60)
    
    # Test 1: Single concept activation
    print("\n--- Test 1: Wave Propagation ---")
    print("👤 You: Activate '사랑' in the field")
    
    field.reset()
    field.activate("사랑", intensity=1.0, depth=1.0)
    
    insight = field.get_field_insight()
    print(f"🤖 Elysia's Field State:")
    print(f"   Total energy: {insight['total_energy']:.2f}")
    print(f"   Peak intensity: {insight['peak_intensity']:.2f}")
    print(f"   Field coherence: {insight['field_coherence']:.2f}")
    print(f"   Deep layer activation: {insight['z_depth_profile']:.2f}")
    
    # Test 2: Multiple concept interaction
    print("\n--- Test 2: Concept Superposition ---")
    print("👤 You: Activate both '사랑' and '고통'")
    
    field.reset()
    field.activate("사랑", intensity=1.0, depth=1.0)
    field.activate("고통", intensity=0.8, depth=0.5)
    
    insight2 = field.get_field_insight()
    resonance_zones = field.find_resonance_zones(threshold=0.05)
    
    print(f"🤖 Elysia's Field Observation:")
    print(f"   Energy (superposed): {insight2['total_energy']:.2f}")
    print(f"   Resonance zones found: {len(resonance_zones)}")
    
    if resonance_zones:
        print(f"\n   Top resonance zones:")
        for i, zone in enumerate(resonance_zones[:3], 1):
            x, y, z = zone['position']
            print(f"      {i}. Position ({x}, {y}, {z})")
            print(f"         Intensity: {zone['intensity']:.3f}")
            print(f"         Depth: {zone['depth_ratio']:.2f}")
    
    # Test 3: Temporal evolution
    print("\n--- Test 3: Temporal Evolution ---")
    print("👤 You: Let '사랑' evolve over time")
    
    field.reset()
    energies = []
    
    for t in range(5):
        field.activate("사랑", intensity=1.0, depth=1.0)
        insight_t = field.get_field_insight()
        energies.append(insight_t['total_energy'])
        print(f"   t={t}: Energy = {insight_t['total_energy']:.2f}")
    
    print(f"\n🤖 Elysia: 사랑의 에너지가 시간에 따라 변화한다")
    
    # Test 4: Depth penetration
    print("\n--- Test 4: Depth Penetration (Z-axis) ---")
    print("👤 You: How deep does '사랑' penetrate?")
    
    field.reset()
    field.activate("사랑", intensity=1.0, depth=0.3)  # Shallow
    shallow_profile = field.get_field_insight()['z_depth_profile']
    
    field.reset()
    field.activate("사랑", intensity=1.0, depth=1.0)  # Deep
    deep_profile = field.get_field_insight()['z_depth_profile']
    
    print(f"🤖 Elysia's Depth Analysis:")
    print(f"   Shallow (depth=0.3): {shallow_profile:.3f}")
    print(f"   Deep (depth=1.0): {deep_profile:.3f}")
    print(f"   🤖 Elysia: 깊이가 다르면 이해의 차원이 달라진다")
    
    print("\n=== Demonstration Complete ===")
    print("\nGrok's continuous field enables:")
    print("  ✅ Wave superposition (multiple concepts interact)")
    print("  ✅ Temporal evolution (ψ changes with time)")
    print("  ✅ Depth penetration (Z-axis = understanding depth)")
    print("  ✅ Emergent resonance zones (non-linear patterns)")
    print("\nThis is thinking in 4D space-time, not 1D paths.")

if __name__ == "__main__":
    run_simulation()
