"""
Verify Topological Resonance
============================

Demonstrates "Dimensional Analysis" of thoughts.
1. Low Dimensional Input (Point/Line) -> "I hate code" (Anger is planar).
2. High Dimensional Input (Sphere/Hypersphere) -> "Why does existence require entropy?" (Structural).

Elysia should detect the *Topology* (Shape) of these thoughts.
"""

import sys
import os
import logging
import time

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.reasoning_engine import ReasoningEngine

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("TopologyTest")

def test_topology():
    print("🌌 Initializing Reasoning Engine with Topological Sight...")
    engine = ReasoningEngine()
    
    print("\n--- Test Case 1: Low-Dim Input (Flat Logic/Emotion) ---")
    # "Hate" is an X-axis spike (Emotion), so likely PLANE logic (2D).
    input_text = "I hate this stupid code error."
    print(f"User Input: '{input_text}'")
    insight = engine.think(input_text)
    print(f"Result: {insight.content[:100]}...")
    
    print("\n--- Test Case 2: High-Dim Input (Structural/Sphere) ---")
    # "System", "Logic", "Structure" cover X, Y, Z -> likely SPHERE (3D).
    input_text = "The system logic requires a structural balance between chaos and order."
    print(f"User Input: '{input_text}'")
    insight = engine.think(input_text)
    print(f"Result: {insight.content[:100]}...")

if __name__ == "__main__":
    try:
        test_topology()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ CRITICAL ERROR: {e}")
