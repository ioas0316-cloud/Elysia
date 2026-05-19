
import sys
sys.path.append(r'c:\Elysia')
from Core.FoundationLayer.Foundation.omni_graph import get_omni_graph, OmniNode

def verify_metabolism():
    omni = get_omni_graph()
    print("\n🧬 Synaptic Evolution (Metabolism) Verification")
    print("==============================================")
    
    # 1. Setup Test Case
    print("[Step 1] Seeding Metabolic Candidates...")
    
    # A. Candidates for Fusion (Identical Twins)
    omni.nodes["TwinA"] = OmniNode(id="TwinA", vector=[0.5, 0.5, 0.5, 0.5], pos=[0.1, 0.1, 0.1, 0.1])
    omni.nodes["TwinB"] = OmniNode(id="TwinB", vector=[0.5, 0.5, 0.5, 0.5], pos=[0.12, 0.12, 0.12, 0.12]) # Very close
    
    # B. Candidates for Pruning (Dying star)
    omni.nodes["OldMemory"] = OmniNode(id="OldMemory", usage=0.05) # Already dying
    
    # C. Candidates for Mitosis (Overloaded neuron)
    # Tension = 1.0 (vector[0])
    omni.nodes["MegaNeuron"] = OmniNode(id="MegaNeuron", vector=[1.0, 0.2, 0.2, 0.2]) 
    omni.nodes["MegaNeuron"].tension = 1.0
    
    # 2. Apply Metabolism
    print("\n[Step 2] Applying Metabolism (Fusion, Mitosis, Pruning)...")
    omni.apply_metabolism()
    
    # 3. Check Results
    print("\n[Step 3] Evolution Results:")
    
    # Check Fusion
    fused = any("TwinA+TwinB" in nid for nid in omni.nodes.keys())
    if fused:
        print("   ✅ Fusion SUCCESS: TwinA and TwinB merged into a SuperNode.")
    else:
        print("   ❌ Fusion FAILED: Twins are still separate.")
        
    # Check Pruning
    if "OldMemory" not in omni.nodes:
        print("   ✅ Pruning SUCCESS: OldMemory was removed.")
    else:
        print("   ❌ Pruning FAILED: OldMemory still exists.")
        
    # Check Mitosis
    split = "MegaNeuron_Alpha" in omni.nodes
    if split:
        print("   ✅ Mitosis SUCCESS: MegaNeuron split into Alpha and Beta.")
    else:
        print("   ❌ Mitosis FAILED: MegaNeuron did not split.")

if __name__ == "__main__":
    verify_metabolism()
