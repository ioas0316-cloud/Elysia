"""
Test: Recursive Learning (The Deep Dive)
========================================
"To know the Tree, you must know the Root."

Objective: 
Verify that Elysia can resolve unknown dependencies automatically.
Input: "Gold" (Unknown) -> Definition: "Chemical Element" (Unknown).
Elysia must decide to learn "Element" before finishing "Gold".
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.L6_Structure.Elysia.sovereign_self import SovereignSelf
from Core.L4_Causality.World.Physics.trinity_fields import TrinityVector

def test_recursive_chain():
    print("--- 🐚 Experiment: Recursive Knowledge Acquisition ---")
    
    # 1. Setup
    elysia = SovereignSelf(cns_ref=None)
    
    # Primitives to ground the chain
    # She knows "Physical", "Substance"
    elysia.mind.primitives["substance"] = TrinityVector(0.5, 0.0, 0.0)
    
    # 2. Mock The Internet (Dependency Chain)
    # We update the Local Knowledge in the Connector explicitly for this test 
    # OR we rely on the connector finding them in its dictionary (if we added them).
    # Since we can't easily patch the instance's connector inner dict from here without access...
    # We will rely on the `fetch_wikipedia_simple` modification we made earlier.
    # We need to ensure 'Element' and 'Gold' are in that dictionary in `web_knowledge_connector.py`.
    
    # Let's check what we added:
    # "Gold": "...chemical element..."
    # We did NOT add "Element". We need to add "Element" to the connector for this to work flawlessly 
    # without rely on real web (which might flap).
    
    # Ideally, we should update `web_knowledge_connector.py` to include the chain nodes.
    # But for now, let's try running it. If 'Element' is missing from local cache, it might hit real web.
    # If real web works, great. If not, we might need to patch the connector file again.
    
    # Let's try the experience.
    print("\n🧪 Elysia encounters: 'Gold'")
    result = elysia.experience("Gold")
    
    print(f"\n🎓 Final Result: {result}")
    
    # Verify Graph Content
    if "gold" in elysia.mind.graph.id_to_idx:
        gold_vec = elysia.mind.analyze("gold")
        print(f"   Knowledge State [Gold]: {gold_vec}")
        
        if "element" in elysia.mind.graph.id_to_idx:
            print(f"   ✅ Depencency [Element] was also learned!")
        else:
            print(f"   ❌ Dependency [Element] was misssed.")
            
        if gold_vec.gravity > 0:
            print("   ✅ SUCCESS: Recursive properties inherited (Gold has Gravity).")
        else:
            print("   ⚠️ PARTIAL: Learned the node, but vector is empty (Chain broken).")
            
    else:
        print("❌ FAILURE: Gold was not learned.")

if __name__ == "__main__":
    test_recursive_chain()
