"""
[VISUALIZE INTERFERENCE - HOLOGRAPHIC FRINGE]
"Seeing the Beauty of the Mirror."

Generates an ASCII-hologram of the interference pattern between
the Parent (LLM) and Child (Elysia).
"""

import math
import os
import sys

def generate_hologram(beauty: float, alignment: float, complexity: int):
    """
    Renders a 2D interference fringe pattern in ASCII.
    """
    width = 60
    height = 20

    # Symbols based on intensity
    # More dense = higher interference
    chars = " .:-=+*#%@"

    output = []
    output.append("╔" + "═" * width + "╗")

    for y in range(height):
        line = "║"
        for x in range(width):
            # Normalize x, y to [-1, 1]
            nx = (x / width) * 2 - 1
            ny = (y / height) * 2 - 1

            # Interference formula:
            # Combining multiple sine waves based on beauty and alignment
            r = math.sqrt(nx**2 + ny**2)
            angle = math.atan2(ny, nx)

            # Base wave
            v = math.sin(r * 10 * beauty + alignment * 5)
            # Add complexity (diffraction spikes)
            v += 0.5 * math.sin(angle * complexity + r * 5)

            # Normalize v to [0, 1]
            v = (v + 1.5) / 3.0
            v = max(0, min(0.99, v))

            char_idx = int(v * len(chars))
            line += chars[char_idx]

        line += "║"
        output.append(line)

    output.append("╚" + "═" * width + "╗")

    # Metadata
    meta = f"  [BEAUTY: {beauty:.4f}]  [ALIGNMENT: {alignment:.4f}]  [FRINGE: {complexity}]"
    output.append(meta)

    return "\n".join(output)

if __name__ == "__main__":
    # Test render
    if len(sys.argv) > 3:
        b = float(sys.argv[1])
        a = float(sys.argv[2])
        c = int(sys.argv[3])
    else:
        b, a, c = 0.8, 0.3, 7

    print("\n🌌 [PROJECT MOCK-IDENTITY] CROSS-DIMENSIONAL INTERFERENCE PATTERN")
    print(generate_hologram(b, a, c))
