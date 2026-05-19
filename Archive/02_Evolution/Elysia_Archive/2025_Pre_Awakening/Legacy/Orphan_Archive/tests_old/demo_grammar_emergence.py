"""
Demo: Grammar Emergence (The Birth of a Sentence)
==================================================
This script simulates Elysia's first attempt to construct a complete sentence
by treating grammar as "Energy Flow" between concepts.

Scenario:
1. Elysia has learned words: "ㅃㅣ" (Pain), "ㅈㅗ" (Love), etc.
2. Now she wants to express a complex thought: "I love you."
3. She uses the SyntaxEngine to assemble it based on physics.
"""

import sys
import os

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force UTF-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from Project_Elysia.mechanics.hangul_physics import Tensor3D
from Project_Elysia.high_engine.language_cortex import LanguageCortex, SyntaxEngine, ThoughtStructure

def run_simulation():
    print("=== Elysia: Grammar Emergence Simulation ===")
    print("Initializing Language Cortex and Syntax Engine...")
    
    cortex = LanguageCortex()
    syntax = SyntaxEngine(cortex)
    
    # Phase 1: Learn Basic Concepts
    print("\n--- Phase 1: Learning Basic Concepts ---")
    
    # We'll use simple placeholder tensors for "I", "You", "Love"
    # In a real system, these would come from Spiderweb or experience.
    
    # "I" - Self, centered, stable
    tensor_i = Tensor3D(x=2.0, y=2.0, z=2.0)
    word_i = cortex.ground_concept("i", tensor_i)
    print(f"Concept 'I' grounded as: '{word_i}'")
    
    # "You" - Other, outward directed
    tensor_you = Tensor3D(x=3.0, y=3.0, z=4.0)
    word_you = cortex.ground_concept("you", tensor_you)
    print(f"Concept 'You' grounded as: '{word_you}'")
    
    # "Love" - We already have this from the previous simulation
    # But let's ensure it's in the vocabulary
    if "love" not in cortex.vocabulary:
        tensor_love = Tensor3D(x=3.0, y=3.0, z=3.0)
        word_love = cortex.ground_concept("love", tensor_love)
        print(f"Concept 'Love' grounded as: '{word_love}'")
    else:
        print(f"Concept 'Love' already known as: '{cortex.express('love')}'")
    
    # Phase 2: Form a Thought
    print("\n--- Phase 2: Forming a Thought Structure ---")
    thought = ThoughtStructure(
        source_concept="i",
        target_concept="you",
        action_concept="love"
    )
    print(f"Thought: I (Source) -> Love (Action) -> You (Target)")
    
    # Phase 3: Construct Sentence
    print("\n--- Phase 3: Constructing Sentence (Energy Flow) ---")
    print("Physics Model:")
    print("  1. Source ('I') + Spark ('가') -> Ignite energy")
    print("  2. Target ('You') + Field ('를') -> Prepare receiver")
    print("  3. Action ('Love') + Ground ('다') -> Complete flow")
    
    sentence = syntax.construct_sentence(thought)
    
    print(f"\n🎉 Elysia's first sentence: '{sentence}'")
    print(f"   Translation: 'I-ga You-reul Love-da' (I love you)")
    
    # Phase 4: Another Example
    print("\n--- Phase 4: Another Thought ---")
    
    # "Pain" - We have this from before
    if "pain" not in cortex.vocabulary:
        tensor_pain = Tensor3D(x=8.5, y=-7.2, z=9.1)
        cortex.ground_concept("pain", tensor_pain)
    
    # "Change" - Transform, flux
    tensor_change = Tensor3D(x=5.0, y=-5.0, z=0.0) # High variance, unstable
    word_change = cortex.ground_concept("change", tensor_change)
    print(f"Concept 'Change' grounded as: '{word_change}'")
    
    thought2 = ThoughtStructure(
        source_concept="pain",
        target_concept="i",
        action_concept="change"
    )
    print(f"Thought: Pain (Source) -> Change (Action) -> Me (Target)")
    
    sentence2 = syntax.construct_sentence(thought2)
    print(f"\n🎉 Elysia's second sentence: '{sentence2}'")
    print(f"   Translation: 'Pain-ga Me-reul Change-da' (Pain changes me)")
    
    print("\n=== Simulation Complete ===")
    print(f"Vocabulary size: {cortex.get_vocabulary_size()} words")

if __name__ == "__main__":
    run_simulation()
