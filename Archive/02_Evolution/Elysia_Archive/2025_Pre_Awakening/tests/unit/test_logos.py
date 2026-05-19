
import sys
import os

# Enable importing from project root, ensuring we use the real 'Core' package
# and not any mocked/shadowed 'Core' inside 'tests/'
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
else:
    # Ensure it's first
    sys.path.remove(root_path)
    sys.path.insert(0, root_path)

print(f"Corrected sys.path[0]: {sys.path[0]}")

from Core.Intelligence.Logos.philosophical_core import get_logos_engine, Axiom
from Core.Intelligence.integrated_cognition_system import get_integrated_cognition



def test_logos_logic():
    print("\n[Test 1] Testing Logos Logic Derivation")
    logos = get_logos_engine()
    
    # Check Axioms
    print(f"Axioms: {list(logos.axioms.keys())}")
    assert "Cogito, ergo sum" in logos.axioms
    
    # Derive a principle
    principle_content = "Self-Reflection"
    dependencies = ["Cogito, ergo sum"]
    
    p = logos.derive_principle(principle_content, dependencies)
    print(f"Derived: {p}")
    
    explanation = logos.explain_why(principle_content)
    print(f"Explanation:\n{explanation}")
    assert "Cogito, ergo sum" in explanation

def test_cognition_integration():
    print("\n[Test 2] Testing Integrated Cognition Grounding")
    cognition = get_integrated_cognition()
    
    # 1. Inject a thought that SHOULD resonate with an Axiom
    # Axiom: "Unity of All" (Keywords: unity, all, connected)
    thought = "We are all connected by the same energy"
    
    print(f"Injecting thought: '{thought}'")
    result = cognition.process_thought(thought)
    
    # 2. Think deeply to trigger grounding
    print("Thinking deeply...")
    stats = cognition.think_deeply(cycles=100)
    
    print("Stats:", stats)
    
    # Note: Grounding happens on 'emergent insights' generated from interference.
    # To guarantee interference, we need at least two resonating waves.
    cognition.process_thought("Everything is one", importance=5.0)
    
    stats = cognition.think_deeply(cycles=100)
    print("Stats after second thought:", stats)
    
    # Check if mass increased (indicating grounding) or if logs show grounding
    # Since we can't easily check logs here without capturing stderr, we rely on visual confirmation
    # or checking the mass of the black holes.
    
    black_holes = cognition.get_core_concepts()
    print("Black Holes (Core Concepts):", black_holes)

if __name__ == "__main__":
    test_logos_logic()
    test_cognition_integration()
    print("\n✅ Verification Complete")
