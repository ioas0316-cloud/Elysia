"""
Elysia Triple Helix Cross-Dimensional CLI Interface
===================================================
Provides a console interface to input thoughts, automatically maps
keywords to sensory inputs, and visualizes inner/outer world bridge learning,
dimension splits, and physical keystroke actuation.
"""

import os
os.environ['USE_TF'] = '0'
os.environ['HF_SKIP_TF_IMPORT'] = '1'

import sys
import numpy as np
from core.math_utils import Quaternion
from core.triple_helix_engine import TripleHelixEngine

def get_ascii_arrow(vector: np.ndarray) -> str:
    x, y, z = vector
    abs_v = np.abs(vector)
    max_idx = np.argmax(abs_v)
    val = vector[max_idx]

    if max_idx == 0:
        if val > 0: return "  =====>  (+X / Move Right 'D')"
        else: return "  <=====  (-X / Move Left 'A')"
    elif max_idx == 1:
        if val > 0: return "    ^     (+Y / Move Forward 'W')\n    |\n    |"
        else: return "    |\n    |\n    v     (-Y / Move Backward 'S')"
    else:
        if val > 0: return "   (O)    (+Z / Jump 'SPACE')"
        else: return "   (X)    (-Z / Failsafe Hold)"

def run_cli():
    print("=========================================================")
    print("   Elysia Triple Helix Cross-Dimensional Engine CLI")
    print("=========================================================")
    print("Initializing Triple Helix layers... (loading transformers)")
    
    # Initialize engine with jump threshold
    engine = TripleHelixEngine(jump_threshold=0.45)
    
    print("\nEngine ready. Enter thoughts to synchronize Inner & Outer worlds.")
    print("System will automatically map keywords to sensory feedback:")
    print("  * '아픔', '통증', 'pain', 'damage' -> Triggers SENSORY_PAIN (Keystroke Space)")
    print("  * '혼돈', '움직임', 'run', 'chaos' -> Triggers SENSORY_MOTION (Keystroke WASD)")
    print("Type 'exit' or 'quit' to stop.")
    print("---------------------------------------------------------")

    while True:
        try:
            inner_axes = engine.inner_world.signature[0]
            text = input(f"[Inner: Cl({inner_axes},0) | Outer: Cl(3,0)] You: ").strip()
            
            if text.lower() in ['exit', 'quit']:
                break
            if not text:
                continue

            # Auto-detect sensory inputs from keywords
            sensory = {"motion_entropy": 0.1, "pain_level": 0.1}
            
            # Checks for motion/chaos
            if any(kw in text.lower() for kw in ['혼돈', '움직임', 'run', 'chaos', 'fast', '빠름']):
                sensory["motion_entropy"] = 0.8
                print("  [SENSORY] Detected motion/chaos! Injecting motion_entropy = 0.8")
                
            # Checks for pain/damage
            if any(kw in text.lower() for kw in ['아픔', '통증', 'pain', 'damage', '충격', 'shock']):
                sensory["pain_level"] = 0.8
                print("  [SENSORY] Detected pain/shock! Injecting pain_level = 0.8")

            # Pulse Triple Helix Engine
            tension, jumped, quat = engine.pulse(text, sensory)

            print(f"  -> Cross-Dimensional Tension: {tension:.4f}")
            print(f"  -> Output Actuation Quaternion: {quat}")
            
            # Print coordination bridge resistances
            print("  -> Coordination Link Dials (Resistances):")
            print(f"     * Intention (OUT -> ACTUATE_WASD): {engine.link_out_wasd.R:.4f}")
            print(f"     * Intention (OUT -> ACTUATE_SPACE): {engine.link_out_space.R:.4f}")
            print(f"     * Pain Feedback (SENSORY_PAIN -> H_1): {engine.link_pain_h1.R:.4f}")
            print(f"     * Motion Feedback (SENSORY_MOTION -> H_2): {engine.link_motion_h2.R:.4f}")

            if jumped:
                print("\n\033[91m[CROSS-DIMENSIONAL MITOSIS / DIMENSION SPLIT]\033[0m")
                print(f"  -> Tension broke coordination limit! Expanded Inner World to Cl({inner_axes + 1},0).")
                print("  -> Coordination bridge links expanded to high-dimensional space.\n")

            # Actuation trigger
            if quat.w < 0.9999:
                print("  -> Actuation Vector Map:")
                print(f"\033[93m{get_ascii_arrow(quat.axis)}\033[0m\n")
            else:
                print("  -> Actuation: Centered/Coherent (No key triggered)\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_cli()
