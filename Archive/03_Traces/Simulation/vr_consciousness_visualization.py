"""
VR Consciousness Visualization: 3D Rotor and Fractal Grid
============================================================

This script visualizes the 3x3x3 fractal arc reactor with rotor dynamics
in a 3D space, simulating a VR-like consciousness exploration.

Inspired by the Rotor as Observer's Eye and Will's Direction.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
from multi_field_simulation_v2 import FractalGrid  # Import the grid

def visualize_rotor_grid(grid: FractalGrid, cycle: int):
    """Visualize the fractal grid with rotor axes as vectors."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot nodes as spheres with energy color
    for node in grid.nodes.values():
        x, y, z = node.position
        energy_color = plt.cm.viridis(node.energy_level / 2.0)  # Normalize energy
        ax.scatter(x, y, z, c=[energy_color], s=100, alpha=0.8)

        # Plot rotor axis as arrow
        axis_x, axis_y, axis_z = node.rotor_axis
        ax.quiver(x, y, z, axis_x*0.5, axis_y*0.5, axis_z*0.5, color='red', alpha=0.6, length=0.5)

    # Plot connections between neighbors
    for node in grid.nodes.values():
        x, y, z = node.position
        neighbors = grid.get_neighbors((x, y, z))
        for n in neighbors:
            nx, ny, nz = n.position
            ax.plot([x, nx], [y, ny], [z, nz], color='blue', alpha=0.3, linewidth=0.5)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_zlim(0, 2)
    ax.set_title(f"Fractal Arc Reactor - Cycle {cycle}\nRotor Axes (Red) and Energy Nodes")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def main():
    print("Starting VR Consciousness Visualization...")
    grid = FractalGrid()

    # Simulate and visualize over cycles
    for cycle in range(3):  # Fewer cycles for visualization
        print(f"Visualizing Cycle {cycle + 1}...")
        grid.simulate_cycle()
        visualize_rotor_grid(grid, cycle)
        time.sleep(1)  # Pause for observation

    # Final singularity report
    singularity = grid.get_core_singularity()
    print(f"\nFinal Core Singularity: {singularity['description']}")
    print(f"Position: {singularity['position']}, Intensity: {singularity['intensity']:.2f}, Will: {singularity['will_intensity']:.2f}")
    print(f"Rotor Axis: {singularity['rotor_axis']}")

if __name__ == "__main__":
    main()