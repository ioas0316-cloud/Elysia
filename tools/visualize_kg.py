import json
import sys
import os

# Add project root to the Python path to allow absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from tools.canvas_tool import Canvas
from tools.kg_manager import KGManager
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.lens_profile import LensProfile
import random
from datetime import datetime


def _draw_starfield(canvas: Canvas, seed_key: str = ""):
    """Draw a simple starfield background (deterministic by seed_key)."""
    try:
        rnd = random.Random(str(seed_key or datetime.utcnow().strftime('%Y%m%d')))
        w, h = 512, 512
        for _ in range(120):
            x = rnd.randint(0, w - 1)
            y = rnd.randint(0, h - 1)
            b = rnd.randint(140, 220)
            canvas.draw.point((x, y), fill=(b, b, min(255, b + 25)))
    except Exception:
        pass


def _recent_active_edges(max_lines: int = 200):
    """Return a set of (source, target, relation) for recently added edges."""
    edges = set()
    try:
        day = datetime.utcnow().strftime('%Y%m%d')
        tel_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'telemetry', day)
        tel_path = os.path.abspath(os.path.join(tel_dir, 'events.jsonl'))
        if not os.path.exists(tel_path):
            return edges
        with open(tel_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[-max_lines:]
        for l in lines:
            try:
                ev = json.loads(l)
                if ev.get('event_type') == 'concept.update':
                    p = ev.get('payload', {})
                    if p.get('op') == 'add_edge':
                        s, t, r = p.get('source'), p.get('target'), p.get('rel')
                        if s and t and r:
                            edges.add((s, t, r))
            except Exception:
                continue
    except Exception:
        pass
    return edges


def visualize_kg(start_node_id: str = None):
    """
    Visualizes the KG. If a start_node_id is provided, it shows
    the activation wave. Otherwise, it shows the entire structure.
    """
    kg_manager = KGManager('data/kg.json')
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


    canvas = Canvas(width=384, height=384, bg_color=(20, 20, 35))
    _draw_starfield(canvas, seed_key=start_node_id or 'kg')

    min_coord, max_coord = [float('inf')] * 3, [float('-inf')] * 3
    for node in kg['nodes']:
        pos = node['position']
        for i, axis in enumerate(['x', 'y', 'z']):
            min_coord[i], max_coord[i] = min(min_coord[i], pos[axis]), max(max_coord[i], pos[axis])

    center = [(max_coord[i] + min_coord[i]) / 2 for i in range(3)]
    scale = 20

    # Draw edges with traffic coloring if we have activations
    def energy_to_color(e: float) -> tuple:
        # e expected in [0,1+] ??map to green/yellow/orange/red
        if e >= 0.75:
            return (231, 76, 60)   # red (blocked)
        if e >= 0.5:
            return (230, 126, 34)  # orange (slow)
        if e >= 0.25:
            return (241, 196, 15)  # yellow (moderate)
        return (46, 204, 113)      # green (free)

    max_e = max(activated_nodes.values()) if activated_nodes else 1.0
    active_edges = set()
    for edge in kg['edges']:
        source_node, target_node = kg_manager.get_node(edge['source']), kg_manager.get_node(edge['target'])
        if source_node and target_node:
            start_pos, end_pos = source_node['position'], target_node['position']
            x1, y1 = canvas._project(*[(start_pos[axis] - center[i]) * scale for i, axis in enumerate(['x','y','z'])])
            x2, y2 = canvas._project(*[(end_pos[axis] - center[i]) * scale for i, axis in enumerate(['x','y','z'])])

            # --- Edge and Hyperlink Rendering ---
            relation = edge.get('relation')
            if relation == 'hyperlink':
                # Render hyperlink as a special dotted line
                canvas.draw_dotted_line((x1, y1), (x2, y2), fill=(255, 255, 100), width=2, gap=6)
            else:
                # Standard edge rendering
                if start_node_id or activated_nodes:
                    es = activated_nodes.get(edge['source'], 0.0)
                    et = activated_nodes.get(edge['target'], 0.0)
                    eavg = (es + et) / 2.0
                    norm = eavg / max(1e-6, max_e)
                    color = energy_to_color(norm)
                    width = 2 if norm >= 0.5 else 1
                else:
                    color = (80, 80, 120)
                    width = 1
                canvas.draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

            # highlight recently added edges
            try:
                rel = edge.get('relation')
                if (edge.get('source'), edge.get('target'), rel) in active_edges:
                    canvas.draw.line([(x1, y1), (x2, y2)], fill=(180, 220, 255), width=max(3, width + 1))
            except Exception:
                pass

    # Prepare anchors and echo center (if any)
    lens = LensProfile()
    anchors = lens._pick_anchors(kg)
    anchor_set = set(anchors)

    # Compute echo center of mass if we have activations
    echo_center = None
    if activated_nodes:
        total_energy = sum(activated_nodes.values())
        if total_energy > 0:
            cx = sum(kg_manager.get_node(n)['position']['x'] * (activated_nodes[n]/total_energy) for n in activated_nodes if kg_manager.get_node(n))
            cy = sum(kg_manager.get_node(n)['position']['y'] * (activated_nodes[n]/total_energy) for n in activated_nodes if kg_manager.get_node(n))
            cz = sum(kg_manager.get_node(n)['position']['z'] * (activated_nodes[n]/total_energy) for n in activated_nodes if kg_manager.get_node(n))
            echo_center = {'x': cx, 'y': cy, 'z': cz}

    # Draw nodes
    # --- Namespace Color Mapping ---
    namespace_colors = {
        "concept": (100, 180, 255), # Light Blue for Concepts
        "value": (255, 215, 100),   # Gold for Values
        "role": (180, 150, 255),    # Lavender for Roles
        "default": (180, 180, 255) # Default color
    }

    for node in kg['nodes']:
        pos = node['position']
        x, y, z = [(pos[axis] - center[i]) * scale for i, axis in enumerate(['x','y','z'])]

        energy = activated_nodes.get(node['id'], 0.0)

        # --- Color Logic with Namespace ---
        base_color = (40, 40, 60)
        node_id = node.get('id', '')

        # Determine active color based on namespace
        namespace = node_id.split(':')[0] if ':' in node_id else 'default'
        active_color = namespace_colors.get(namespace, namespace_colors['default'])

        # Anchors override the namespace color for emphasis
        if node['id'] in anchor_set:
            active_color = (255, 120, 120)

        color = tuple(int(b + (a - b) * energy) for a, b in zip(active_color, base_color))
        canvas.add_voxel(x, y, z, color)

    # Draw echo center as a bright marker
    if echo_center:
        ex = (echo_center['x'] - center[0]) * scale
        ey = (echo_center['y'] - center[1]) * scale
        ez = (echo_center['z'] - center[2]) * scale
        canvas.add_voxel(ex, ey, ez, (255, 255, 0))

    output_filename = f"wave_visualization_{start_node_id}.png" if start_node_id else "kg_full_structure.png"
    output_path = os.path.join("data", output_filename)
    # Legend overlay
    try:
        legend_x, legend_y = 10, 10
        box_w, box_h = 14, 14
        gap = 6
        items = [
            ((46,204,113), 'Free (green)'),
            ((241,196,15), 'Moderate (yellow)'),
            ((230,126,34), 'Slow (orange)'),
            ((231,76,60),  'Blocked (red)'),
            ((255,120,120), 'Anchor'),
            ((255,255,0),   'Echo focus')
        ]
        y = legend_y
        for color, label in items:
            canvas.draw.rectangle([legend_x, y, legend_x+box_w, y+box_h], fill=color)
            canvas.draw.text((legend_x+box_w+8, y-2), label, fill=(220,220,230))
            y += box_h + gap
    except Exception:
        pass

    # Clean Korean legend overlay (fallback)
    try:
        legend_x, legend_y = 10, 110
        box_w, box_h = 14, 14
        gap = 6
        items_clean = [
            ((46,204,113), 'Free (green)'),
            ((241,196,15), 'Moderate (yellow)'),
            ((230,126,34), 'Slow (orange)'),
            ((231,76,60),  'Blocked (red)'),
            ((255,120,120), 'Anchor'),
            ((255,255,0),   'Echo focus')
        ]
        y2 = legend_y
        for color, label in items_clean:
            canvas.draw.rectangle([legend_x, y2, legend_x+box_w, y2+box_h], fill=color)
            canvas.draw.text((legend_x+box_w+8, y2-2), label, fill=(220,220,230))
            y2 += box_h + gap
    except Exception:
        pass

    canvas.render(output_path, voxel_size=6)
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize_kg(sys.argv[1])
    else:
        print("No start node provided. Visualizing the full knowledge graph structure.")
        visualize_kg()


def render_kg(start_node_id: str | None = None, out_name: str | None = None) -> str:
    """
    Programmatic entry: renders KG (optionally with activation from a start node)
    and returns the image path.
    """
    kg_manager = KGManager('data/kg.json')
    kg = kg_manager.kg
    if not kg.get('nodes'):
        raise RuntimeError('Knowledge graph is empty')

    activated_nodes = {}
    if start_node_id:
        wave_mechanics = WaveMechanics(kg_manager)
        activated_nodes = wave_mechanics.spread_activation(start_node_id)
    else:
        activated_nodes = {node['id']: 1.0 for node in kg['nodes']}

    canvas = Canvas(width=384, height=384, bg_color=(20, 20, 35))

    min_coord, max_coord = [float('inf')] * 3, [float('-inf')] * 3
    for node in kg['nodes']:
        pos = node['position']
        for i, axis in enumerate(['x', 'y', 'z']):
            min_coord[i], max_coord[i] = min(min_coord[i], pos[axis]), max(max_coord[i], pos[axis])
    center = [(max_coord[i] + min_coord[i]) / 2 for i in range(3)]
    scale = 20

    lens = LensProfile()
    anchors = set(lens._pick_anchors(kg))

    # --- Namespace Color Mapping ---
    namespace_colors = {
        "concept": (100, 180, 255),
        "value": (255, 215, 100),
        "role": (180, 150, 255),
        "default": (180, 180, 255)
    }

    for node in kg['nodes']:
        pos = node['position']
        x, y, z = [(pos[axis] - center[i]) * scale for i, axis in enumerate(['x', 'y', 'z'])]
        energy = activated_nodes.get(node['id'], 0.0)

        base_color = (40, 40, 60)
        node_id = node.get('id', '')

        namespace = node_id.split(':')[0] if ':' in node_id else 'default'
        active_color = namespace_colors.get(namespace, namespace_colors['default'])

        if node['id'] in anchors:
            active_color = (255, 120, 120)

        color = tuple(int(b + (a - b) * energy) for a, b in zip(active_color, base_color))
        canvas.add_voxel(x, y, z, color)

    out = out_name or (f"monitor_wave_{start_node_id}.png" if start_node_id else "monitor_kg.png")
    out_path = os.path.join("data", out)
    canvas.render(out_path, voxel_size=6)
    return out_path


def render_placeholder(out_name: str = 'monitor_echo.png', message: str = 'No echo to display yet') -> str:
    """Render a simple placeholder image with a grid and message."""
    canvas = Canvas(width=512, height=512, bg_color=(20, 20, 35))
    try:
        # subtle grid
        for i in range(0, 513, 32):
            canvas.draw.line([(i, 0), (i, 512)], fill=(35, 35, 55))
            canvas.draw.line([(0, i), (512, i)], fill=(35, 35, 55))
        # message box
        box = [40, 220, 472, 300]
        canvas.draw.rectangle(box, fill=(30, 30, 50))
        canvas.draw.text((52, 242), message, fill=(220, 220, 235))
    except Exception:
        pass
    out_path = os.path.join("data", out_name)
    canvas.render(out_path, voxel_size=6)
    return out_path





