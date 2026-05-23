"""
Elysia Clifford-IPN Thought Aligner CLI Interface
==================================================
Console interface to input thoughts and observe high-dimensional
Clifford-IPN routing, Ohmic learning, and dimension splits.
"""

import sys
import numpy as np
from engines.thought_aligner.clifford_aligner_engine import CliffordThoughtAlignerEngine

def get_ascii_arrow(vector: np.ndarray) -> str:
    """Returns an ASCII arrow representation of a 3D direction vector."""
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
    print("  Elysia Clifford-IPN Aligner CLI - Causal Multivector")
    print("=========================================================")
    print("Initializing cognitive models... (loading sentence transformers)")
    
    # Initialize engine
    # Jump threshold set to 0.45 to easily trigger splits on contrasting thoughts
    engine = CliffordThoughtAlignerEngine(jump_threshold=0.45)
    
    print("\nEngine ready. Type thoughts to observe Clifford dynamics and memory.")
    print("Try typing repeating themes to see link resistance drop,")
    print("then type a completely different theme to trigger a dimension split!")
    print("Type 'exit' or 'quit' to stop.")
    print("---------------------------------------------------------")

    while True:
        try:
            active_axes = engine.net.signature[0]
            text = input(f"[Cl({active_axes},0) | Ticks: {engine.net.stable_ticks}] You: ").strip()
            
            if text.lower() in ['exit', 'quit']:
                break
            if not text:
                continue

            tension, jumped, quat = engine.process_thought(text)

            print(f"  -> Global Network Tension: {tension:.4f}")
            print(f"  -> Output Quaternion: {quat}")
            
            # Print some link resistances to show Ohmic learning in action
            print("  -> Ohmic Memory State (Resistances):")
            for i in range(1, min(4, active_axes + 1)):
                # Find link from IN_i to H_1
                links = [l for l in engine.net.links if l.node_from == f"IN_{i}" and l.node_to == "H_1"]
                if links:
                    print(f"     * IN_{i} -> H_1 Resistance: {links[0].R:.4f}")

            if jumped:
                print("\n\033[91m[BIFURCATION / DIMENSION SPLIT TRIGGERED]\033[0m")
                print(f"  -> Tension exceeded limit! Expanded to Cl({active_axes + 1},0) to redistribute shock.")
                print("  -> Projection matrix extended causally.")
                
            # If the quaternion is not identity, print its orientation axis
            if quat.w < 0.9999:
                print(f"  -> Axis: {quat.axis}")
                print(f"\033[93m{get_ascii_arrow(quat.axis)}\033[0m\n")
            else:
                print("  -> State: Centered / Coherent (No rotation)\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_cli()
