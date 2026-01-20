
import sys
import os
sys.path.append(os.getcwd())

from Core.L4_Causality.World.Nature.trinity_lexicon import get_trinity_lexicon
from Core.L5_Mental.Intelligence.Logos.logos_engine import get_logos_engine

def test_causal_explanation():
    print("🧪 [Test] Phase 27: Causal Syntax (Active Logic)")
    
    lexicon = get_trinity_lexicon()
    graph = lexicon.graph
    logos = get_logos_engine()

    # 1. Inject Knowledge with Bonds
    print("\n1. Injecting Knowledge: 'Fire' burns 'Wood' into 'Ash'...")
    
    # Primitives: Fire, Wood, Ash
    graph.add_node("Fire", vector=[0.0, 0.3, 0.9])
    graph.add_node("Wood", vector=[1.0, 0.0, 0.0])
    graph.add_node("Ash", vector=[0.5, 0.0, -0.5])
    
    # Verbs as Operators
    # Fire -> (burn) -> Wood
    graph.add_link("Fire", "Wood", weight=1.0, link_type="operator")
    # Wood -> (becomes) -> Ash
    graph.add_link("Wood", "Ash", weight=0.8, link_type="causes")

    # 2. Query Logos for Causal Explanation
    print("\n2. Querying Logos Engine for 'Fire'...")
    explanation = logos.reinterpret_causality("Fire", context=["Fire is hot"])
    
    print("-" * 50)
    print(f"Elysia: {explanation}")
    print("-" * 50)

    # 3. Validation
    assert "변화" in explanation or "원인" in explanation or "Fire" in explanation
    print("\n✅ Verification Successful: Elysia perceives the Bond.")

if __name__ == "__main__":
    test_causal_explanation()
