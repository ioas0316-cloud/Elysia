import sys
import numpy as np
from rotor_engine import FractalRotorEngine

def get_ascii_arrow(vector):
    """
    Returns an ASCII art arrow representation of a 3D direction vector.
    """
    x, y, z = vector

    # Simple mapping of dominant axis
    abs_v = np.abs(vector)
    max_idx = np.argmax(abs_v)

    val = vector[max_idx]

    if max_idx == 0: # X-axis
        if val > 0: return "  =====>  (+X)"
        else: return "  <=====  (-X)"
    elif max_idx == 1: # Y-axis
        if val > 0: return "    ^     (+Y)\n    |\n    |"
        else: return "    |\n    |\n    v     (-Y)"
    else: # Z-axis (in/out of screen)
        if val > 0: return "   (O)    (+Z out of screen)"
        else: return "   (X)    (-Z into screen)"

def main():
    print("=========================================================")
    print("   Fractal Rotor CLI - 4D Phase Alignment Engine")
    print("=========================================================")
    print("Initializing cognitive tensors... (loading models)")

    # We use a lower threshold to easily trigger jumps in testing,
    # but in a real scenario, this would be tuned.
    engine = FractalRotorEngine(jump_threshold=0.3)

    print("Engine ready. Enter your thoughts to align phase.")
    print("Type 'exit' or 'quit' to stop.")
    print("---------------------------------------------------------")

    while True:
        try:
            text = input(f"[Depth {engine.fractal_depth}] You: ")
            if text.lower() in ['exit', 'quit']:
                break
            if not text.strip():
                continue

            theta, jumped, direction = engine.process_thought(text)

            if engine.fractal_depth == 1 and theta == 0.0 and len(engine.history) == 1:
                print(f"  -> Initial Phase Set. (Q: {engine.current_q})")
                continue

            print(f"  -> Phase Mismatch (θ): {theta:.4f}")

            if jumped:
                print("\n\033[91m[FRACTAL JUMP DETECTED - PHASE COLLAPSE]\033[0m")
                print(f"  -> Thought crossed dimensional boundary!")
                print(f"  -> Fracturing rotor into sub-scale (Depth: {engine.fractal_depth})")
                print(f"  -> Direction of bend:")
                print(f"\033[93m{get_ascii_arrow(direction)}\033[0m")
                print(f"  -> Vector: {direction}\n")
            else:
                print(f"  -> Resonance maintained at Depth {engine.fractal_depth}.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
