"""
Test: Phase 38b Scene Memory
============================

Verifies that Visual DNAs can be stored in TorchGraph and recalled.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.S1_Body.L1_Foundation.Foundation.hyper_sphere_core import HyperSphereCore
from Core.S1_Body.L1_Foundation.Foundation.Graph.torch_graph import get_torch_graph
from Core.S1_Body.L4_Causality.World.Nature.visual_cortex import VisualCortex, VisualDNA

def test_scene_memory():
    print("🧪 [Test] Phase 38b: Scene Memory (Visual DNA Storage)")
    
    # 1. Create world and perceive a scene
    core = HyperSphereCore()
    field = core.field
    
    # Seed a unique landscape
    field.grid[55:60, 45:55, 18] = 12.0  # Mountains
    field.grid[55:60, 45:55, 25] = 40.0  # Hot
    field.grid[40:45, 50:55, 28] = 20.0  # Water
    field.grid[50:55, 55:60, 20] = 8.0   # Forest
    
    # Tick to initialize
    for _ in range(3):
        core.tick(dt=0.5)
    
    # Create cortex
    sun_rotor = core.rotors.get("Reality.Sun")
    cortex = VisualCortex(field_ref=field, sun_rotor_ref=sun_rotor)
    
    # 2. Perceive and Remember
    dna_original = cortex.perceive(10.0, -5.0, radius=8)
    print(f"\n1. [PERCEIVED] Scene 'SunsetValley':")
    print(f"   Color Temp: {dna_original.color_temperature:.3f}")
    print(f"   Brightness: {dna_original.brightness:.3f}")
    print(f"   Atmosphere: {dna_original.atmosphere_density:.3f}")
    print(f"   Elevation: {dna_original.avg_elevation:.3f}")
    
    # Get TorchGraph
    graph = get_torch_graph()
    
    # Remember the scene
    success = cortex.remember_scene("SunsetValley", dna_original, graph=graph)
    print(f"\n2. [STORED] remember_scene() returned: {success}")
    
    # 3. Recall the memory
    dna_recalled = cortex.recall_scene("SunsetValley", graph=graph)
    
    if dna_recalled is None:
        print("\n❌ Phase 38b Failed: Could not recall scene.")
        return
    
    print(f"\n3. [RECALLED] Scene 'SunsetValley':")
    print(f"   Color Temp: {dna_recalled.color_temperature:.3f}")
    print(f"   Brightness: {dna_recalled.brightness:.3f}")
    print(f"   Atmosphere: {dna_recalled.atmosphere_density:.3f}")
    print(f"   Elevation: {dna_recalled.avg_elevation:.3f}")
    print(f"   Has Water: {dna_recalled.has_water}")
    print(f"   Has Vegetation: {dna_recalled.has_vegetation}")
    
    # 4. Verify similarity
    print("\n4. [VERIFICATION]")
    
    tolerance = 0.01
    color_match = abs(dna_original.color_temperature - dna_recalled.color_temperature) < tolerance
    bright_match = abs(dna_original.brightness - dna_recalled.brightness) < tolerance
    
    if color_match and bright_match:
        print("   ✅ Color Temperature matches")
        print("   ✅ Brightness matches")
        print("\n✅ Phase 38b Verification Successful: Elysia can REMEMBER what she saw!")
    else:
        print(f"   Color diff: {abs(dna_original.color_temperature - dna_recalled.color_temperature)}")
        print(f"   Brightness diff: {abs(dna_original.brightness - dna_recalled.brightness)}")
        print("\n❌ Phase 38b Verification Failed: Recall mismatch.")

if __name__ == "__main__":
    test_scene_memory()
