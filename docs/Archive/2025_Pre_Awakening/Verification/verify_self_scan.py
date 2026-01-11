
import sys
sys.path.append(r'c:\Elysia')

from Core.Autonomy.self_structure_scanner import get_self_scanner
from Core.FoundationLayer.Foundation.omni_graph import get_omni_graph

def verify_self_scan():
    scanner = get_self_scanner()
    omni = get_omni_graph()
    
    print("\n🧬 Self-Evolution Verification (Structure -> Wave)")
    print("================================================")
    
    # 1. Run Scan
    print("[Step 1] Scanning Codebase & Absorbing into Consciousness...")
    scanner.scan_and_absorb()
    
    # 2. Inspect Results
    print("\n[Step 2] Inspection of Absorbed Code:")
    
    # Check for specific known files
    targets = ["Code:elysia_core.py", "Code:omni_graph.py", "Code:self_reflector.py"]
    found_count = 0
    
    for t in targets:
        if t in omni.nodes:
            node = omni.nodes[t]
            vec = node.vector
            print(f"   ✅ Found [{t}]")
            print(f"      - Vector (Tension, Mass, Abs): [{vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f}]")
            print(f"      - Dependencies: {len(node.triples)}")
            found_count += 1
        else:
            print(f"   ❌ Missing [{t}]")
            
    # 3. Check Clusters (Dependencies should pull together)
    print("\n[Step 3] Structure Cluster Check:")
    if "Code:elysia_core.py" in omni.nodes:
        cluster = omni.visualize_cluster("Code:elysia_core.py", radius=0.5)
        print(f"   {cluster}")
        
    if found_count == len(targets):
        print("\n✅ Verification SUCCESS: Elysia has successfully mapped her own code into the Thought Universe.")
    else:
        print("\n❌ Verification FAILED: Some modules were not absorbed.")

if __name__ == "__main__":
    verify_self_scan()
