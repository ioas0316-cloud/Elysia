"""
Visualizes the 3D knowledge graph.

This script reads the kg.json file, plots the nodes in a 3D space,
draws the edges connecting them, and saves the visualization to an image file.
"""
import json
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Setup paths
DATA_DIR = Path("data")
KG_PATH = DATA_DIR / 'kg.json'
OUTPUT_DIR = Path("docs")
OUTPUT_PATH = OUTPUT_DIR / "kg_visualization.png"

def visualize_kg():
    """
    Generates and saves a 3D visualization of the knowledge graph.
    """
    # Set Korean font for matplotlib
    try:
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        font_prop = font_manager.FontProperties(fname=font_path)
        rc('font', family=font_prop.get_name())
        plt.rcParams['axes.unicode_minus'] = False # Fix for minus sign not showing
        print("Korean font 'NanumGothic' set successfully.")
    except FileNotFoundError:
        print("Warning: NanumGothic font not found. Korean text may not display correctly.")

    print(f"Reading knowledge graph from {KG_PATH}...")
    if not KG_PATH.exists():
        print(f"Error: Knowledge graph file not found at {KG_PATH}")
        return

    with open(KG_PATH, 'r', encoding='utf-8') as f:
        kg = json.load(f)

    nodes = kg.get('nodes', [])
    edges = kg.get('edges', [])

    if not nodes:
        print("Warning: No nodes found in the knowledge graph.")
        return

    print("Creating 3D plot...")
    # Create a 3D plot
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')

    # Plot nodes and labels
    node_positions = {}
    for node in nodes:
        node_id = node['id']
        pos = node['position']
        x, y, z = pos['x'], pos['y'], pos['z']
        node_positions[node_id] = (x, y, z)
        ax.scatter(x, y, z, s=120, c='skyblue', alpha=0.9, edgecolors='k')
        ax.text(x, y, z, f'  {node_id}', size=10, zorder=1, color='black')

    # Plot edges
    print("Drawing edges between nodes...")
    for edge in edges:
        source_id = edge['source']
        target_id = edge['target']

        if source_id in node_positions and target_id in node_positions:
            pos_source = node_positions[source_id]
            pos_target = node_positions[target_id]

            x_coords = [pos_source[0], pos_target[0]]
            y_coords = [pos_source[1], pos_target[1]]
            z_coords = [pos_source[2], pos_target[2]]

            # Add relation label at the midpoint of the edge
            mid_point = ((pos_source[0]+pos_target[0])/2, (pos_source[1]+pos_target[1])/2, (pos_source[2]+pos_target[2])/2)
            ax.text(mid_point[0], mid_point[1], mid_point[2], edge['relation'], size=8, color='red', alpha=0.8)

            ax.plot(x_coords, y_coords, z_coords, c='gray', alpha=0.6, linestyle='--')

    ax.set_title("Elysia's Core Knowledge Graph", fontsize=18)
    ax.set_xlabel("X-axis (Relatedness/Cause)")
    ax.set_ylabel("Y-axis (Abstraction/Is_A)")
    ax.set_zlabel("Z-axis (Composition/Depth)")
    ax.view_init(elev=20., azim=-65) # Adjust camera angle for better view

    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Save the plot
    plt.savefig(OUTPUT_PATH)
    print(f"Success! Visualization saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    visualize_kg()
