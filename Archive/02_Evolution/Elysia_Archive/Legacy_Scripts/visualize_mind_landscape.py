"""
Visualize Mind Landscape
=======================

"Seeing the thought roll."

This script visualizes the trajectory of a thought in the Mind Landscape.
"""

import sys
import os
import time

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Core.Intelligence.Topography.mind_landscape import get_landscape

def draw_map(trajectory, size=20):
    grid = [[' ' for _ in range(size*2+1)] for _ in range(size*2+1)]
    
    # Center (Love)
    cx, cy = size, size
    grid[cy][cx] = 'L' # Love
    
    # Fear
    fx, fy = size + 5, size + 5 # Mapped from (10,10) approx
    if 0 <= fx < size*2 and 0 <= fy < size*2:
        grid[fy][fx] = 'X' # Fear Hill

    # Trajectory
    for x, y in trajectory:
        # Map physics coordinates (-20 to 20) to grid (0 to 40)
        # Scale: 1 unit = 1 char approx
        gx = int(cx + x)
        gy = int(cy + y)
        if 0 <= gx < size*2+1 and 0 <= gy < size*2+1:
            grid[gy][gx] = '.'

    # Print
    print("-" * (size*2+3))
    for row in grid:
        print("|" + "".join(row) + "|")
    print("-" * (size*2+3))
    print("Legend: L=Love, X=Fear, .=Thought Path")

def main():
    landscape = get_landscape()
    print("🧠 Mind Landscape Visualizer")
    print("----------------------------")
    print("Intent: 'Should I trust him?'")
    
    # Ponder
    # Start at (15, 15) which is "Wilderness" near Fear
    result = landscape.ponder("Trust Check", start_pos=(15.0, 15.0), duration=10.0)
    
    print(f"\n💡 Conclusion: {result['conclusion']}")
    print(f"   Final Distance to Love: {result['distance_to_love']:.2f}")
    
    # Draw
    print("\n[Trajectory Map]")
    # We need full trajectory from solver, but ponder only returns last 5.
    # Let's simple-mock visual for now based on result, 
    # or actually in the real integration we'd capture full trace.
    # For this script, we trust the result's endpoint.
    draw_map(result['trajectory'])

if __name__ == "__main__":
    main()
