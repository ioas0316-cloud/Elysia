# -*- coding: utf-8 -*-
"""
Galactic Story Test
===================

Tests the Galactic Expansion of Cosmic Syntax.
Verifies Nebula (Paragraph) generation from multiple Star Systems (Sentences).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.L1_Foundation.Foundation.resonance_grammar import CosmicSyntaxEngine

def main():
    print("\n" + "="*70)
    print("  Galactic Story Test")
    print("="*70 + "\n")
    
    engine = CosmicSyntaxEngine()
    
    # Define Thoughts (Star Systems)
    thought1 = ["Love", "Bonds", "Creates"]
    thought2 = ["Trust", "Cooperation", "Enables"]
    
    # 1. Test Nebula Weaving (English)
    print("1. Testing Nebula Weaving (English)...")
    nebula_text = engine.weave_nebula([thought1, thought2], medium="and")
    print(f"Thought 1: {thought1}")
    print(f"Thought 2: {thought2}")
    print(f"Nebula: {nebula_text}")
    
    expected = "Love creates Bonds and Trust enables Cooperation."
    if expected in nebula_text:
        print("  SUCCESS: Nebula woven correctly.")
    else:
        print(f"  FAILED: Expected '{expected}', got '{nebula_text}'")

    # 2. Test Nebula Weaving (Korean)
    print("\n2. Testing Nebula Weaving (Korean)...")
    engine.set_korean_mode(True)
    nebula_text_kr = engine.weave_nebula([thought1, thought2], medium="and")
    print(f"Nebula (KR): {nebula_text_kr}")
    
    # Expected:     ...          ...       .
    # Note: 'Trust' -> 'Trust' (if not in lexicon), 'Cooperation' -> 'Cooperation'
    # 'Enables' -> '       ' (mapped in language_projector)
    
    if "    " in nebula_text_kr and "   " in nebula_text_kr and "       " in nebula_text_kr:
        print("  SUCCESS: Korean Nebula woven correctly.")
    else:
        print("  FAILED: Korean Nebula incorrect.")

    print("\n" + "="*70)
    print("  Galactic Story Test Complete")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
