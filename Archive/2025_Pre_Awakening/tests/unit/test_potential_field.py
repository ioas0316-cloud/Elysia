import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.potential_field import PotentialField

def test_potential_field():
    print("Testing Potential Field Topology...")
    
    field = PotentialField()
    
    # 1. Setup Terrain
    # Create a "Love" attractor at (10, 10)
    field.add_gravity_well(10.0, 10.0, strength=50.0)
    print("Added Gravity Well at (10, 10)")
    
    # Create a "Logic" railgun from (0, 0) to (5, 5)
    field.add_railgun(0.0, 0.0, 5.0, 5.0, force=2.0)
    print("Added Railgun Channel from (0, 0) to (5, 5)")
    
    # 2. Spawn Particle
    field.spawn_particle("thought_1", 0.0, 0.0)
    print("Spawned Particle 'thought_1' at (0, 0)")
    
    # 3. Simulate Flow
    print("\nSimulating Flow (10 steps):")
    for i in range(10):
        field.step()
        state = field.get_state()[0]
        print(f"Step {i+1}: Pos({state['x']:.2f}, {state['y']:.2f}) Vel({state['vx']:.2f}, {state['vy']:.2f})")
        
    # Check if it moved towards the well (10, 10)
    final_pos = field.particles[0].pos
    dist_to_well = ((final_pos.x - 10)**2 + (final_pos.y - 10)**2)**0.5
    print(f"\nFinal Distance to Well: {dist_to_well:.2f}")
    
    if dist_to_well < 10.0: # Started at ~14 distance
        print("SUCCESS: Particle flowed towards the Gravity Well.")
    else:
        print("FAILURE: Particle did not move as expected.")

if __name__ == "__main__":
    test_potential_field()
