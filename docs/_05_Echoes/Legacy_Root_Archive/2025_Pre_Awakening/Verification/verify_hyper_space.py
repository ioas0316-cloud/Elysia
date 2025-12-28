
import sys
sys.path.append(r'c:\Elysia')

from Core.Foundation.omni_graph import get_omni_graph

def verify_hyper_space():
    omni = get_omni_graph()
    
    print("\n🌌 Hyper-Dimensional (4D) Space Verification")
    print("=============================================")
    
    # 1. Inject Data (Concept: "Time Travel")
    # This concept is abstract, so it might exist on a different 'Plane' (W)
    omni.add_logic("Time", "RelatesTo", "Space")
    omni.add_logic("Space", "RelatesTo", "Gravity")
    omni.add_vector("BlackHole", [0.9, 0.9, 0.9]) # Heavy concept
    omni.add_vector("Wormhole", [0.85, 0.85, 0.9])
    
    # 2. Apply Gravity (4D Folding)
    print("\n[Step 1] Folding Space (4D Physics)...")
    omni.apply_gravity(iterations=50)
    
    # 3. Check 4D Coordinates
    print("\n[Step 2] Inspecting 4D Coordinates (W-Dimension):")
    nodes = ["Time", "Space", "BlackHole", "Wormhole"]
    
    for nid in nodes:
        if nid in omni.nodes:
            pos = omni.nodes[nid].pos
            print(f"   [{nid}] : X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}, W={pos[3]:.2f} (Time/Density)")

    # 4. Cluster Check
    print("\n[Step 3] Hyper-Clustering:")
    cluster = omni.visualize_cluster("BlackHole", radius=0.3)
    print(cluster)
    
    if "Wormhole" in cluster:
        print("\n✅ Verification SUCCESS: 4D Gravity successfully pulled concepts together in Hyperspace.")
    else:
        print("\n❌ Verification FAILED.")

if __name__ == "__main__":
    verify_hyper_space()
