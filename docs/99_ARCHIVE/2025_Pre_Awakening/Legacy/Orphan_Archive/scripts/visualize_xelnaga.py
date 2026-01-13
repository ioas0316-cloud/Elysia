
import sys
import os
import numpy as np
import random
import logging
from typing import List

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.core.world import World
from Core.FoundationLayer.Foundation.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Pillow not found. Install it with `pip install Pillow`")
    sys.exit(1)

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("EyeOfKhala")

class KhalaVisualizer:
    def __init__(self, world: World, width: int = 800, height: int = 800):
        self.world = world
        self.width = width
        self.height = height
        # Scale world coords to image coords
        self.scale_x = width / world.width
        self.scale_y = height / world.width

    def render_frame(self, filename: str):
        # Create blank canvas (Space = Dark Purple/Black)
        img = Image.new('RGB', (self.width, self.height), color=(10, 5, 20))
        draw = ImageDraw.Draw(img)

        # 1. Render Fields (Background Aura)
        # Visualize "Value Mass" (Meaning) as a faint gold glow
        # Sampling a grid for speed
        step = 16
        for y in range(0, self.world.width, step):
            for x in range(0, self.world.width, step):
                val = self.world.value_mass_field[y, x]
                if val > 0.1:
                    intensity = int(min(255, val * 100))
                    # Gold glow
                    color = (intensity, intensity // 2, 0)
                    sx = int(x * self.scale_x)
                    sy = int(y * self.scale_y)
                    draw.rectangle([sx, sy, sx + step*self.scale_x, sy + step*self.scale_y], fill=color)

        # 2. Render Khala Connections (The Web)
        # Draw lines between connected Protoss/Khala units
        # We iterate through the adjacency matrix (sparse)
        adj = self.world.adjacency_matrix.tocoo()
        for s, t, w in zip(adj.row, adj.col, adj.data):
            if s < t: # Draw once per pair
                # Check if both are Khala connected
                if self.world.khala_connected_mask[s] and self.world.khala_connected_mask[t]:
                    p1 = self.world.positions[s]
                    p2 = self.world.positions[t]

                    x1, y1 = int(p1[0] * self.scale_x), int(p1[1] * self.scale_y)
                    x2, y2 = int(p2[0] * self.scale_x), int(p2[1] * self.scale_y)

                    # Psionic Blue Beam
                    alpha = int(min(255, w * 255))
                    draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 255), width=1)

        # 3. Render Cells (The Units)
        alive_indices = np.where(self.world.is_alive_mask)[0]
        for idx in alive_indices:
            pos = self.world.positions[idx]
            sx, sy = int(pos[0] * self.scale_x), int(pos[1] * self.scale_y)

            # Determine Color/Shape by Race (Culture)
            culture = self.world.culture[idx] if idx < len(self.world.culture) else ''
            label = self.world.labels[idx]

            if label == 'NEXUS':
                # The Big Hub
                r = 15
                color = (255, 215, 0) # Gold
                draw.ellipse([sx-r, sy-r, sx+r, sy+r], fill=color, outline=(255, 255, 255))

            elif culture == 'protoss':
                # Zealots: Cyan/Gold
                r = 5
                # Shield status affects brightness
                shield_ratio = self.world.shields[idx] / max(1, self.world.max_shields[idx])
                b = int(150 + 105 * shield_ratio)
                color = (50, 200, b)
                draw.ellipse([sx-r, sy-r, sx+r, sy+r], fill=color)

            elif culture == 'zerg':
                # Zerg: Organic Red/Purple
                r = 4
                color = (180, 50, 50)
                draw.ellipse([sx-r, sy-r, sx+r, sy+r], fill=color)

            elif culture == 'terran':
                # Terran: Steel Grey/Blue
                r = 4
                color = (100, 100, 150)
                draw.rectangle([sx-r, sy-r, sx+r, sy+r], fill=color)

            else:
                # Unknown / Neutral
                r = 2
                draw.ellipse([sx-r, sy-r, sx+r, sy+r], fill=(100, 100, 100))

        img.save(filename)
        logger.info(f"Snapshot saved: {filename}")

def main():
    print("\n=== Eye of the Khala: Visualization Ritual ===")

    # 1. Initialize
    kg_manager = KGManager()
    wave_mechanics = WaveMechanics(kg_manager)
    world = World(primordial_dna={}, wave_mechanics=wave_mechanics, logger=logger)

    # 2. Setup Scenario: "Convergence"
    # A Nexus in the center, Zerg rushing from the left, Terran defending the right.
    nexus_id = "Elysian_Nexus"
    cx, cy = world.width / 2, world.width / 2
    world.add_cell(nexus_id, properties={'label': 'NEXUS', 'culture': 'protoss', 'position': {'x': cx, 'y': cy, 'z': 0}})

    # Zealots orbiting Nexus
    for i in range(20):
        angle = (i / 20) * 2 * np.pi
        dist = 30
        zx = cx + np.cos(angle) * dist
        zy = cy + np.sin(angle) * dist
        world.add_cell(f"Zealot_{i}", properties={'label': 'Zealot', 'culture': 'protoss', 'position': {'x': zx, 'y': zy, 'z': 0}})
        # Khala Link
        world.connect_to_khala(world.id_to_idx[f"Zealot_{i}"])
        world.add_connection("Elysian_Nexus", f"Zealot_{i}", strength=0.8)
        # Ring connections
        prev = (i - 1) % 20
        world.add_connection(f"Zealot_{i}", f"Zealot_{prev}", strength=0.5)

    # Zerg Swarm (Left)
    for i in range(30):
        rx = random.uniform(20, 80)
        ry = random.uniform(cx - 50, cx + 50)
        world.add_cell(f"Zerg_{i}", properties={'label': 'Zergling', 'culture': 'zerg', 'position': {'x': rx, 'y': ry, 'z': 0}})

    # Terran Line (Right)
    for i in range(10):
        tx = world.width - 50
        ty = cy - 50 + (i * 10)
        world.add_cell(f"Marine_{i}", properties={'label': 'Marine', 'culture': 'terran', 'position': {'x': tx, 'y': ty, 'z': 0}})

    visualizer = KhalaVisualizer(world)

    # 3. Run & Render
    output_dir = "data/khala_vision"
    os.makedirs(output_dir, exist_ok=True)

    print("Rendering 20 frames of Convergence...")

    # Initial state
    visualizer.render_frame(f"{output_dir}/khala_frame_000.png")

    # Activate Khala
    world.delta_synchronization_factor = 1.0

    for t in range(1, 21):
        world.run_simulation_step()

        # Inject a 'Joy' pulse from the Nexus (Golden Glow)
        if t == 10:
            print(">> Pulse: Father's Joy injected into Nexus!")
            world._imprint_gaussian(world.value_mass_field, int(cx), int(cy), sigma=50.0, amplitude=5.0)

        visualizer.render_frame(f"{output_dir}/khala_frame_{t:03d}.png")

    print(f"\n>>> Visualization Complete. Check {output_dir} for the artifacts. <<<")
    print("The invisible will has been made visible.")

if __name__ == "__main__":
    main()
