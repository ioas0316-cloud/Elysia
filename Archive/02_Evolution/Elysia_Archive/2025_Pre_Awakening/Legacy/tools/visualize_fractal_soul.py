import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.core.self_fractal import SelfFractalCell

def visualize_soul_growth():
    # Setup output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'gallery', 'soul_fractal')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created gallery directory: {output_dir}")

    # Initialize the Soul (Seed)
    elysia_soul = SelfFractalCell()
    print("Initialized SelfFractalCell (The Seed).")

    # Visualization settings
    plt.figure(figsize=(10, 10), facecolor='black')

    # Capture the initial state (Layer 0)
    save_snapshot(elysia_soul, 0, output_dir)

    # Grow the soul (Layers 1-12)
    max_layers = 12
    print(f"Starting growth for {max_layers} layers...")

    for layer in range(1, max_layers + 1):
        complexity = elysia_soul.autonomous_grow()
        print(f"  - Layer {layer}: Complexity = {complexity}")
        save_snapshot(elysia_soul, layer, output_dir)

    print(f"\nVisualization complete. {max_layers + 1} images saved to {output_dir}")

def save_snapshot(cell, layer, output_dir):
    plt.clf()  # Clear current figure

    # Plot the grid
    # vmin=0, vmax=1.0 ensures consistent color scaling
    # cmap='plasma' gives the requested purple/blue/yellow energy feel
    plt.imshow(cell.grid, cmap='plasma', vmin=0, vmax=1.0, origin='upper')

    # Styling
    plt.axis('off')  # Hide axis for pure visual
    plt.title(f"Elysia's Soul - Layer {layer}", color='white', fontsize=16, pad=20)

    # Save
    filename = f"soul_layer_{layer:02d}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, facecolor='black', bbox_inches='tight', dpi=100)

if __name__ == "__main__":
    visualize_soul_growth()
