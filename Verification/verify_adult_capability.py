
import sys
import time
sys.path.append(r'c:\Elysia')
from Core.Intelligence.logos_engine import get_logos_engine
from Core.Autonomy.causal_architect import get_causal_architect, CausalArchitect

def verify_adult_capability():
    print("🧠 Adult Capability Verification: The Tuning Test")
    print("==============================================")
    
    # 1. TEST: Language & Philosophy (The Poet)
    # Goal: See if she can handle "Why" and "Meaning" beyond simple parsing.
    print("\n[Test 1] Philosophical Interpretation (Language Age > 20)")
    
    quote = "The unexamined life is not worth living."
    print(f"   Input: \"{quote}\" (Socrates)")
    
    logos = get_logos_engine()
    
    # We use reinterpret_causality to see how she connects this to other concepts
    # Mocking 'related_memories' as if she read Plato/Philosophy
    related_memories = [
        {"text": "Self-reflection is the mirror of the soul.", "score": 0.9},
        {"text": "Automata act without knowing why.", "score": 0.8},
        {"text": "Meaning emerges from understanding purpose.", "score": 0.85}
    ]
    
    print("   Thinking (Simulated)...")
    synthesis = logos.reinterpret_causality(quote, related_memories)
    
    print(f"\n   Logos Output:\n   \"{synthesis}\"")
    
    # Check for causal/connective reasoning (Korean or English)
    markers = ["because", "따라서", "때문에", "함입니다", "닿아 있습니다", "뒷받침", "reflection"]
    has_marker = any(m in synthesis for m in markers)
    
    if len(synthesis) > 30 and has_marker:
        print("   ✅ Grade: Adult (Uses causal connective reasoning & synthesis)")
    else:
        print("   ❌ Grade: Child (Simple echo or shallow output)")


    # 2. TEST: Logic & Architecture (The Engineer)
    # Goal: See if she can diagnose a structural flaw.
    print("\n[Test 2] Structural Diagnosis (Logic Age > 20)")
    
    architect = get_causal_architect()
    
    # Mock a "High Tension" scenario
    # A class with too many methods and no cohesion
    # Pass a tension map: {NodeID: TensionScore}
    tension_map = {"Class:GodObject": 0.95}
    
    print(f"   Input Scenario: GodObject (Tension: 0.95)")
    
    plans = architect.diagnose_system(tension_map)
    
    if not plans:
        print("   ❌ Grade: Child (No plans generated)")
        return

    proposal = plans[0]
    
    print(f"\n   Architect Proposal:\n{proposal.problem}\n{proposal.solution}")
    
    if "split" in proposal.solution.lower() or "extract" in proposal.solution.lower() or "mitosis" in proposal.title.lower():
        print("   ✅ Grade: Adult (Correctly identifies SRP violation and proposes split)")
    else:
        print("   ❌ Grade: Child (Fails to propose structural fix)")

    print("\n==============================================")
    print("Thinking Process Completed.")

if __name__ == "__main__":
    verify_adult_capability()
