import json
import sys
import os

# Add project root to the Python path to allow absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from tools.canvas_tool import Canvas
from tools.kg_manager import KGManager
from Project_Sophia.wave_mechanics import WaveMechanics


def visualize_kg(start_node_id: str = None):
    """
    Visualizes the KG. If a start_node_id is provided, it shows
    the activation wave. Otherwise, it shows the entire structure.
    """
    kg_manager = KGManager()
    kg = kg_manager.kg

    if not kg['nodes']:
        print("Knowledge graph is empty. Nothing to visualize.")
        return

    activated_nodes = {}
    if start_node_id:
        wave_mechanics = WaveMechanics(kg_manager)
        activated_nodes = wave_mechanics.spread_activation(start_node_id)
        if not activated_nodes:
            print(f"Could not find start node '{start_node_id}' or no activation spread.")
            return
    else: # If no start node, activate all nodes to show structure
        activated_nodes = {node['id']: 1.0 for node in kg['nodes']}


    canvas = Canvas(width=512, height=512, bg_color=(20, 20, 35))

    min_coord, max_coord = [float('inf')] * 3, [float('-inf')] * 3
    for node in kg['nodes']:
        pos = node['position']
        for i, axis in enumerate(['x', 'y', 'z']):
            min_coord[i], max_coord[i] = min(min_coord[i], pos[axis]), max(max_coord[i], pos[axis])

    center = [(max_coord[i] + min_coord[i]) / 2 for i in range(3)]
    scale = 20

    # Draw edges
    for edge in kg['edges']:
        source_node, target_node = kg_manager.get_node(edge['source']), kg_manager.get_node(edge['target'])
        if source_node and target_node:
            start_pos, end_pos = source_node['position'], target_node['position']
            x1, y1 = canvas._project(*[(start_pos[axis] - center[i]) * scale for i, axis in enumerate(['x','y','z'])])
            x2, y2 = canvas._project(*[(end_pos[axis] - center[i]) * scale for i, axis in enumerate(['x','y','z'])])
            canvas.draw.line([(x1, y1), (x2, y2)], fill=(80, 80, 120), width=1)

    # Draw nodes
    for node in kg['nodes']:
        pos = node['position']
        x, y, z = [(pos[axis] - center[i]) * scale for i, axis in enumerate(['x','y','z'])]

        energy = activated_nodes.get(node['id'], 0.0)

        # Color interpolation
        base_color = (40, 40, 60)
        active_color = (180, 180, 255) if not start_node_id else (255, 255, 100)
        color = tuple(int(b + (a - b) * energy) for a, b in zip(active_color, base_color))

        canvas.add_voxel(x, y, z, color)

    output_filename = f"wave_visualization_{start_node_id}.png" if start_node_id else "kg_full_structure.png"
    output_path = os.path.join("data", output_filename)
    canvas.render(output_path, voxel_size=8)
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize_kg(sys.argv[1])
    else:
        print("No start node provided. Visualizing the full knowledge graph structure.")
        visualize_kg()
