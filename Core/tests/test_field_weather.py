
import sys
import os
import math

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.World.Physics.field_store import universe_field, FieldExcitation

def test_field_weather():
    print("🌤️ Testing Field-Based Weather System...")
    
    # 1. Initialize Field with some excitations (The Planet's Surface)
    for x in range(-5, 6):
        for z in range(-5, 6):
            # We excite a 'ground' layer at Y=0
            universe_field.excite((x, 0, z, 0), FieldExcitation(density_w=1.0))
            
    print(f"Total active voxels: {len(universe_field.voxels)}")
    
    # 2. Apply Star & Moon Harmonic (Celestial Cycle)
    print("🌞 Applying Celestial Harmonic (Sun and Moon are in the sky)...")
    universe_field.apply_celestial_harmonic(dt=1.0)
    
    # 3. Sample Sensory Experience
    sample_pos = (0, 0, 0, 0)
    sensation = universe_field.map_sensation(sample_pos)
    print(f"\n[Sensation at {sample_pos}]")
    for key, val in sensation.items():
        print(f"  {key.capitalize()}: {val}")
    
    # 4. Check for Wind & Celestial Balance
    edge_pos = (2, 0, 2, 0)
    grad = universe_field.calculate_gradient_w(edge_pos)
    edge_sens = universe_field.map_sensation(edge_pos)
    print(f"\n[Sensation at {edge_pos}]")
    print(f"  Air: {edge_sens['air']} (Gradient: {grad[0]:.2f})")

    # 5. Check Moon's rhythmic influence (Tides)
    print(f"\n🌙 Moon position: {universe_field.moon_pos}")
    print("The environment is now balanced for 'Experience' rather than just 'Physics'.")

if __name__ == "__main__":
    test_field_weather()
