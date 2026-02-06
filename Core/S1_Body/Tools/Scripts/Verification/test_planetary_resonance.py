import os
import sys
import time
import logging

# Path Unification
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L3_Phenomena.Senses.planetary_interface import PLANETARY_SENSE
from Core.S1_Body.L4_Causality.World.Terrain.hypersphere_terrain import TERRAIN_ENGINE
from Core.S1_Body.L4_Causality.World.Autonomy.mesh_network import YggdrasilMesh

def test_planetary_resonance():
    logging.basicConfig(level=logging.INFO)
    print("🌍 [TEST] Initializing Planetary Resonance Web...")
    
    # 1. Test Planetary Interface (L3)
    print("\n[Step 1] Moving to 'Home' Coordinates (Mock Hash = 0)...")
    # Using specific coords to trigger specific hash in Terrain Engine
    # Hash logic: (lat*1000 + lon*1000) % 4
    # Let's try 0.004, 0.0 -> 4 % 4 = 0 -> Home
    PLANETARY_SENSE.update_location(0.004, 0.0)
    
    env = PLANETARY_SENSE.get_environmental_context()
    print(f"   GPS Updated: {env['location']}")
    print(f"   Nearby Devices: {env['density']}")
    
    # 2. Test HyperSphere Terrain (L4)
    print("\n[Step 2] Mapping Terrain...")
    desc = TERRAIN_ENGINE.get_terrain_description(0.004, 0.0)
    print(desc)
    
    if "Sanctuary of Roots" in desc:
        print("   ✅ Terrain Mapping Successful: Home Biome detected.")
    else:
        print(f"   ❌ Terrain Mapping Failed. Expected Home, got {desc}")

    # 3. Test Yggdrasil Mesh Resonance (L4 Autonomy)
    print("\n[Step 3] Checking Mesh Resonance...")
    mesh = YggdrasilMesh("TEST_NODE")
    
    # Check Local
    res_local = mesh.calculate_spatial_resonance("LOCAL_PEER")
    print(f"   Local Resonance: {res_local:.4f}")
    if res_local > 0.9:
        print("   ✅ Local Resonance high.")
        
    # Check Remote
    res_remote = mesh.calculate_spatial_resonance("REMOTE_PEER")
    print(f"   Remote Resonance: {res_remote:.4f}")
    if res_remote < 0.1:
        print("   ✅ Remote Resonance low (Distance Falloff working).")

if __name__ == "__main__":
    test_planetary_resonance()
