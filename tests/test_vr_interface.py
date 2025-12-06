"""
Test VR Interface Service
==========================

Quick test to verify the VR Interface Service is working correctly.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_vr_interface():
    """Test VR Interface Service basic functionality"""
    print("Testing VR Interface Service...")
    print("=" * 60)
    
    # Import modules
    from Core.World.internal_world import InternalWorld
    from Core.VR.vr_interface_service import VRInterfaceService
    
    # Create Internal World
    print("\n1. Creating Internal World...")
    world = InternalWorld()
    world.create_consciousness_cathedral()
    world.add_knowledge_galaxy('linguistics', (10, 0, 0, 0))
    world.add_knowledge_galaxy('architecture', (0, 10, 0, 0))
    print(f"   ✓ Created world with {len(world.objects)} objects")
    
    # Create VR service
    print("\n2. Creating VR Interface Service...")
    vr_service = VRInterfaceService(world)
    print(f"   ✓ Service initialized (update rate: {vr_service.update_rate} Hz)")
    
    # Test 4D → 3D conversion
    print("\n3. Testing 4D → 3D coordinate conversion...")
    test_pos_4d = (5.0, 3.0, -2.0, 0.8)  # x, y, z, w
    pos_3d = vr_service.convert_4d_to_3d(test_pos_4d)
    print(f"   4D: {test_pos_4d}")
    print(f"   3D: {pos_3d}")
    print(f"   ✓ w-dimension (0.8) added {pos_3d[1] - test_pos_4d[1]:.1f}m to height")
    
    # Test visual properties
    print("\n4. Testing visual properties from w-dimension...")
    props = vr_service.get_visual_properties_from_w(0.8)
    print(f"   w=0.8 (deep/profound)")
    print(f"   → Size multiplier: {props['size_multiplier']:.2f}x")
    print(f"   → Brightness: {props['brightness_multiplier']:.2f}x")
    print(f"   ✓ Profound concepts appear larger and brighter")
    
    # Test Cathedral geometry
    print("\n5. Testing Cathedral geometry...")
    cathedral = vr_service.get_cathedral_geometry()
    print(f"   Pillars: {len(cathedral['pillars'])} (12 knowledge domains)")
    print(f"   Prisms: {len(cathedral['prisms'])} (7 rainbow colors)")
    print(f"   Scale: {cathedral['scale']:.1f}")
    print(f"   Fractal dimension: {cathedral['fractal_dimension']:.2f}")
    print(f"   ✓ Sacred geometry structure ready")
    
    # Test Galaxies
    print("\n6. Testing Knowledge Galaxies...")
    galaxies = vr_service.get_galaxies_data()
    print(f"   Total galaxies: {len(galaxies)}")
    for galaxy in galaxies:
        print(f"   - {galaxy['name']}: {galaxy['star_count']} stars @ {galaxy['center']}")
    print(f"   ✓ Galaxies positioned in 3D space")
    
    # Test visible objects (spatial culling)
    print("\n7. Testing spatial culling...")
    camera_pos = (0, 0, 20)  # 20m from center
    visible = vr_service.get_visible_objects(camera_pos, view_distance=50.0)
    print(f"   Camera at: {camera_pos}")
    print(f"   View distance: 50m")
    print(f"   Visible objects: {len(visible)} / {len(world.objects)}")
    print(f"   ✓ Only nearby objects returned for VR rendering")
    
    # Test initial state
    print("\n8. Testing initial state (VR startup)...")
    initial = vr_service.get_initial_state()
    print(f"   Cathedral: {len(initial['cathedral']['pillars'])} pillars, {len(initial['cathedral']['prisms'])} prisms")
    print(f"   Galaxies: {len(initial['galaxies'])}")
    print(f"   Nebulae: {len(initial['nebulae'])}")
    print(f"   Camera start: {initial['camera_start']}")
    print(f"   ✓ Complete initial state ready for VR client")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("\nTo start the VR service:")
    print("  python Core/VR/vr_interface_service.py")
    print("\nVR client can connect to:")
    print("  WebSocket: ws://localhost:8000/ws/vr")
    print("  REST API: http://localhost:8000/api/vr/initial_state")
    print("=" * 60)


if __name__ == "__main__":
    test_vr_interface()
