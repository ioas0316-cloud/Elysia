"""
Test: Lexicon Logic (Korean)
============================
Verifies that the Lexicon can find '파도' (Wave) in a Korean sentence.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.World.Nature.trinity_lexicon import TrinityLexicon

def test_logic():
    print("--- Testing Lexicon Logic ---")
    lexicon = TrinityLexicon("dummy.json")
    
    # 1. The Text (From Wikipedia Log)
    text = "쓰나미는 일본어로 항구(tsu)와 파도(nami)가 합쳐진 단어."
    print(f"Input: {text}")
    
    # 2. Analyze
    # We call analyze() but force the text input.
    # Note: analyze() splits by default. We need to test _analyze_primitives_only directly 
    # OR update analyze to use substring matching too?
    # Wait, analyze() uses .split()!
    # _analyze_primitives_only uses substring!
    # When we use learn_from_hyper_sphere, we use _analyze_primitives_only.
    # So we should test that.
    
    vector = lexicon._analyze_primitives_only(text)
    
    print(f"Result: G{vector.gravity} F{vector.flow} A{vector.ascension}")
    
    if vector.gravity > 0 or vector.flow > 0:
        print("✅ Analysis Successful: Found the concept.")
    else:
        print("❌ Analysis Failed: Zero Vector.")
        
        # Debug: Check logic manually
        print("Debug: Checking primitives...")
        for key in lexicon.lexicon:
             if key in text:
                 print(f"   MATCH: '{key}' found in text!")

if __name__ == "__main__":
    test_logic()
