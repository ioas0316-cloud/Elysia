
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.internal_universe import InternalUniverse, WorldCoordinate

def verify_temporal_resonance():
    print("⏳ Starting Temporal Resonance Audit...")
    
    universe = InternalUniverse()
    
    # 1. Inject Concept A (Now)
    print("   Creating 'Memory_A'...")
    coord_a = universe.internalize(WorldCoordinate(1.0, 0.0, 0.0, "Memory_A"))
    
    # 2. Simulate Passage of Time (mocking timestamp)
    print("   Simulating 1 hour passing for Memory_A...")
    # Manually backdate timestamp for testing
    universe.coordinate_map["Memory_A"].timestamp -= 3600.0 
    
    # 3. Inject Concept B (Now)
    print("   Creating 'Memory_B' (Fresh)...")
    coord_b = universe.internalize(WorldCoordinate(0.0, 1.0, 0.0, "Memory_B"))
    
    # 4. Check Context Before Decay
    context = universe.get_active_context()
    print(f"   Context Before Decay: {context}")
    
    # 5. Apply Decay
    print("   Applying Metabolism (Half-life: 3600s)...")
    universe.decay_resonance(half_life=3600.0)
    
    # 6. Check Context After Decay
    updated_context = universe.get_active_context()
    print(f"   Context After Decay: {updated_context}")
    
    depth_a = updated_context.get("Memory_A", 0)
    depth_b = updated_context.get("Memory_B", 0)
    
    print(f"   Memory_A Depth: {depth_a:.4f}")
    print(f"   Memory_B Depth: {depth_b:.4f}")
    
    if depth_a < depth_b:
        print("   ✅ SUCCESS: Old memory has faded compared to fresh memory.")
        print("   Temporal Resonance is FUNCTIONAL.")
    else:
        print("   ❌ FAILURE: Old memory did not decay correctly.")

if __name__ == "__main__":
    verify_temporal_resonance()
