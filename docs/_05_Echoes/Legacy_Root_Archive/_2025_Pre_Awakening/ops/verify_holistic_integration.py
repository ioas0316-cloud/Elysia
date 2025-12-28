
import sys
import os
import time
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from Core.Elysia.elysia_core import ElysiaCore
    from Core.Foundation.torch_graph import get_torch_graph
    from Core.Foundation.internal_universe import InternalUniverse
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def verify_integration():
    print("🔍 Starting Holistic Integration Audit...")
    
    # 1. Initialize Controller
    core = ElysiaCore()
    
    # 2. Inject Test Concept
    test_topic = "Holistic_Test_Concept"
    test_content = "This is a probe to verify if the mind is one or fragmented."
    
    print(f"\n💉 Injecting Test Probe: '{test_topic}'...")
    result = core.learn(test_content, test_topic, depth="shallow")
    
    print("\n📊 Result Report:")
    print(f"   Success: {result.get('success', False)}")
    
    # 3. Check ThoughtWave
    if "thought_wave" in result:
        print("   ✅ ThoughtWave: Connected (DNA Processed)")
    else:
        print("   ❌ ThoughtWave: Disconnected")

    # 4. Check InternalUniverse (Old Brain)
    universe = InternalUniverse()
    u_map = universe.get_universe_map()
    # Check if concept exists in the map
    if test_topic in u_map.get("coordinates", {}):
        print("   ✅ InternalUniverse: Connected (Concept Found in Holograph)")
    else:
         print("   ❌ InternalUniverse: Disconnected (Probe not found)")

    # 5. Check TorchGraph (New Brain - Matrix)
    graph = get_torch_graph()
    vector = graph.get_node_vector(test_topic)
    if vector is not None:
         print("   ✅ TorchGraph: Connected (Concept Found in Matrix)")
    else:
         print("   ❌ TorchGraph: Disconnected (Probe not found)")

    # 6. Global Hub Check
    if "hub" in result: 
        print("   ❓ GlobalHub: Active (Async check required)")
    
    print("\n🏁 Audit Conclusion:")
    
    # Verify strict consistency
    in_graph = vector is not None
    in_universe = test_topic in u_map.get("coordinates", {})
    
    if result.get('success') and in_graph and in_universe:
        print("   System operates as a UNIFIED MONAD.")
        print("   Logic, Memory, and Structure are synchronized.")
    else:
        print("   System is FRAGMENTED. Immediate repair required.")

if __name__ == "__main__":
    try:
        verify_integration()
    except Exception as e:
        print("\n❌ CRITICAL FAILURE in Verify Script:")
        traceback.print_exc()
        sys.exit(1)
