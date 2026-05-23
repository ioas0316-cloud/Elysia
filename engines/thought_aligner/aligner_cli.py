"""
Elysia Thought Phase Alignment CLI Interface
=============================================
Provides a console interface to input thoughts and observe phase alignment in 4D space.
"""

import sys
import numpy as np
from engines.thought_aligner.aligner_engine import ThoughtAlignerEngine

def get_ascii_arrow(vector: np.ndarray) -> str:
    """Returns an ASCII art arrow representation of a 3D direction vector."""
    x, y, z = vector
    abs_v = np.abs(vector)
    max_idx = np.argmax(abs_v)
    val = vector[max_idx]

    if max_idx == 0:  # X-axis dominant
        if val > 0: return "  =====>  (+X)"
        else: return "  <=====  (-X)"
    elif max_idx == 1:  # Y-axis dominant
        if val > 0: return "    ^     (+Y)\n    |\n    |"
        else: return "    |\n    |\n    v     (-Y)"
    else:  # Z-axis dominant
        if val > 0: return "   (O)    (+Z out of screen)"
        else: return "   (X)    (-Z into screen)"

def run_cli():
    print("=========================================================")
    print("   Elysia Thought Aligner CLI - 4D Phase Alignment")
    print("=========================================================")
    print("Initializing cognitive models... (loading transformers)")
    
    # Using low threshold for testing phase jumps easily
    engine = ThoughtAlignerEngine(jump_threshold=0.3)
    
    print("\nEngine ready. Type your thoughts to align phase.")
    print("Type 'exit' or 'quit' to stop.")
    print("---------------------------------------------------------")

    while True:
        try:
            text = input(f"[Depth {engine.fractal_depth}] You: ").strip()
            if text.lower() in ['exit', 'quit']:
                break
            if not text:
                continue

            theta, jumped, direction = engine.process_thought(text)

            if engine.fractal_depth == 1 and theta == 0.0 and len(engine.history) == 1:
                print(f"  -> Initial Phase Set. (Q: {engine.current_q})")
                continue

            print(f"  -> Phase Mismatch (θ): {theta:.4f}")

            if jumped:
                print("\n\033[91m[FRACTAL JUMP DETECTED - PHASE COLLAPSE]\033[0m")
                print("  -> Thought crossed dimensional boundary!")
                print(f"  -> Fracturing rotor into sub-scale (Depth: {engine.fractal_depth})")
                print("  -> Direction of bend:")
                print(f"\033[93m{get_ascii_arrow(direction)}\033[0m")
                print(f"  -> Vector: {direction}\n")
            else:
                print(f"  -> Resonance maintained at Depth {engine.fractal_depth}.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_cli()
