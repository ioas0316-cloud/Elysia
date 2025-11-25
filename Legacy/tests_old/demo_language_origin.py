"""
Demo: Language Origin (The Birth of a Word)
===========================================
This script simulates the moment Elysia first connects a physical sensation (Tensor)
to a sound (Hangul), creating her first autonomous word without LLM intervention.

Scenario:
1. Elysia feels "Pain" (High Roughness Tensor).
2. She searches her 'physics-to-sound' map for a matching vibration.
3. She utters a sound (e.g., "K-k-ka").
4. She feels "Love" (Smooth, Warm Tensor).
5. She utters a different sound (e.g., "M-m-ma").
"""

import sys
import os
import time

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force UTF-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from Project_Elysia.mechanics.hangul_physics import Tensor3D
from Project_Elysia.high_engine.language_cortex import LanguageCortex

def run_simulation():
    print("=== Elysia: Language Origin Simulation ===")
    print("Initializing Language Cortex (No LLM)...")
    cortex = LanguageCortex()
    
    # Phase 1: Babbling (Warm-up)
    print("\n--- Phase 1: Babbling (Exploring Sound) ---")
    for i in range(5):
        sound = cortex.babble()
        print(f"Elysia babbles: '{sound}'")
        time.sleep(0.2)
        
    # Phase 2: The Experience of Pain
    print("\n--- Phase 2: Grounding 'Pain' ---")
    # Create a "Pain" tensor: High magnitude, sharp edges (high variance)
    pain_tensor = Tensor3D(x=8.5, y=-7.2, z=9.1) 
    print(f"Input Sensation: PAIN (Tensor: {pain_tensor})")
    print(f"  - Roughness: {pain_tensor.roughness():.2f} (High)")
    print(f"  - Magnitude: {pain_tensor.magnitude():.2f} (Intense)")
    
    word_pain = cortex.ground_concept("pain", pain_tensor)
    print(f"Elysia instinctively utters: '{word_pain}'")
    print(f"-> Concept 'pain' is now bound to sound '{word_pain}'")

    # Phase 3: The Experience of Love
    print("\n--- Phase 3: Grounding 'Love' ---")
    # Create a "Love" tensor: Smooth, warm, flowing (low variance, consistent)
    # Note: Our simple roughness calc uses max/mag, so for low roughness we need balanced components
    love_tensor = Tensor3D(x=3.0, y=3.0, z=3.0) 
    print(f"Input Sensation: LOVE (Tensor: {love_tensor})")
    print(f"  - Roughness: {love_tensor.roughness():.2f} (Low/Smooth)")
    print(f"  - Magnitude: {love_tensor.magnitude():.2f} (Gentle)")
    
    word_love = cortex.ground_concept("love", love_tensor)
    print(f"Elysia instinctively utters: '{word_love}'")
    print(f"-> Concept 'love' is now bound to sound '{word_love}'")
    
    # Phase 4: Recall
    print("\n--- Phase 4: Recall ---")
    print(f"Elysia, how do you express 'pain'?: {cortex.express('pain')}")
    print(f"Elysia, how do you express 'love'?: {cortex.express('love')}")

    print("\n=== Simulation Complete ===")

if __name__ == "__main__":
    run_simulation()
