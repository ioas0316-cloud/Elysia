"""
Test: Phase 38 Visual Cortex
============================

Verifies that the VisualCortex can perceive OmniField states and produce
meaningful Visual DNA descriptions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.1_Body.L1_Foundation.Foundation.hyper_sphere_core import HyperSphereCore
from Core.1_Body.L4_Causality.World.Nature.visual_cortex import VisualCortex, VisualDNA

def test_visual_cortex():
    print("🧪 [Test] Phase 38: The Visual Cortex (Seeing the OmniField)")
    
    # 1. Create a world with varied terrain
    core = HyperSphereCore()
    field = core.field
    
    # Field config: size=100, resolution=2.0 -> World coords range: -100 to +100
    # Grid center (50,50) = World (0,0)
    
    # Seed a Mountain at grid (60, 50) = World (20, 0)
    field.grid[58:63, 48:53, 18] = 15.0  # High elevation, wider area
    
    # Seed Heat across the mountain
    field.grid[58:63, 48:53, 25] = 35.0
    
    # Seed Moisture (Lake) at grid (40, 50) = World (-20, 0)
    field.grid[38:43, 48:53, 28] = 25.0  # High moisture
    
    # Seed Resources (Forest) at grid (50, 60) = World (0, 20)
    field.grid[48:53, 58:63, 20] = 10.0  # High resources
    
    # Tick the world to initialize rotors
    for _ in range(5):
        core.tick(dt=0.5)
    
    print(f"\n1. [WORLD] Created with:")
    print(f"   Mountain at World (20, 0)")
    print(f"   Lake at World (-20, 0)")
    print(f"   Forest at World (0, 20)")
    
    # 2. Create Visual Cortex
    sun_rotor = core.rotors.get("Reality.Sun")
    cortex = VisualCortex(field_ref=field, sun_rotor_ref=sun_rotor)
    
    # 3. Perceive the Mountain at World (20, 0)
    dna_mountain = cortex.perceive(20.0, 0.0, radius=5)
    print(f"\n2. [MOUNTAIN] Visual DNA:")
    print(f"   Elevation: {dna_mountain.avg_elevation:.2f}")
    print(f"   Heat: {dna_mountain.avg_heat:.2f}")
    print(f"   Color Temp: {dna_mountain.color_temperature:.2f}")
    print(f"   Brightness: {dna_mountain.brightness:.2f}")
    print(f"   Description: {dna_mountain.describe()}")
    
    # 4. Perceive the Lake area at World (-20, 0)
    dna_lake = cortex.perceive(-20.0, 0.0, radius=5)
    print(f"\n3. [LAKE] Visual DNA:")
    print(f"   Moisture: {dna_lake.avg_moisture:.2f}")
    print(f"   Atmosphere: {dna_lake.atmosphere_density:.2f}")
    print(f"   Has Water: {dna_lake.has_water}")
    print(f"   Description: {dna_lake.describe()}")
    
    # 5. Perceive the Forest area at World (0, 20)
    dna_forest = cortex.perceive(0.0, 20.0, radius=5)
    print(f"\n4. [FOREST] Visual DNA:")
    print(f"   Has Vegetation: {dna_forest.has_vegetation}")
    print(f"   Description: {dna_forest.describe()}")
    
    # 6. Verification
    print("\n5. [VERIFICATION]")
    
    success = True
    
    if dna_mountain.avg_elevation > 0:
        print("   ✅ Mountain detected (elevation > 0)")
    else:
        print("   ❌ Mountain NOT detected")
        success = False
        
    if dna_lake.has_water:
        print("   ✅ Lake detected (has_water = True)")
    else:
        print("   ❌ Lake NOT detected")
        success = False
    
    if dna_mountain.brightness >= 0.0 and dna_mountain.brightness <= 1.0:
        print(f"   ✅ Sun brightness valid: {dna_mountain.brightness:.2f}")
    else:
        print("   ❌ Sun brightness invalid")
        success = False
    
    if success:
        print("\n✅ Phase 38 Verification Successful: Elysia can SEE her world!")
    else:
        print("\n❌ Phase 38 Verification Failed.")

if __name__ == "__main__":
    test_visual_cortex()
