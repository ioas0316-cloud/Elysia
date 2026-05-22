import os
import sys
import time
import math
import numpy as np
from PIL import Image, ImageDraw

# [ROOT ANCHOR]
_current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(_current_dir))
if root not in sys.path:
    sys.path.insert(0, root)

try:
    from Core.System.rotor import TripleVortexRotor
    from Core.System.recursive_torque import get_torque_engine
except ImportError as e:
    print(f"⚠️ [Vortex Sync Pain] {e}")
    sys.exit(1)

# Visualization Constants
OUTPUT_DIR = os.path.join("data", "visuals", "vortex")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class CosmicCrystallizer:
    """
    [Phase 300: The Self-Healing Vortex Simulation]
    Renders the 'Light' of the Triple Vortex.
    """
    def __init__(self):
        print("🌀 [Cosmic Crystallizer] Initializing Light Matrix Channel...")
        self.vortex = TripleVortexRotor("Master.Vortex")
        self.torque = get_torque_engine()
        self.frame_count = 0
        
    def render_vortex_state(self, iteration: int):
        """
        Renders the 4D Space-Time Vortex (3 Phases + Time Ground).
        Includes Maxwell-Rot electromagnetic field lines.
        """
        size = (512, 512)
        img = Image.new("RGB", size, (0, 0, 15)) # Deep space Blue
        draw = ImageDraw.Draw(img)
        center = (size[0] // 2, size[1] // 2)
        
        # 1. DRAW MAXWELL FIELD LINES (Magnetic Curl)
        # We draw concentric circles representing the 'Rot' of the current
        for r in range(50, 400, 40):
            # Alpha based on resonance
            res = 1.0 - self.vortex.detect_distortion()
            alpha = int(50 * res)
            draw.ellipse([center[0]-r, center[1]-r, center[0]+r, center[1]+r],
                         outline=(0, 255, 255, alpha))

        # 2. Draw Singularity (The Sun at 3^3)
        sing_r = 15 + math.sin(iteration * 0.2) * 5
        draw.ellipse([center[0]-sing_r, center[1]-sing_r, center[0]+sing_r, center[1]+sing_r],
                     fill=(255, 255, 255), outline=(255, 105, 180)) 
        
        # 3. Draw 3-Phase Spatial Axes (Space.X, Y, Z)
        colors = {
            "Phase_A": (255, 80, 80), 
            "Phase_B": (80, 255, 80), 
            "Phase_C": (80, 80, 255)
        }
        
        for name, axis in self.vortex.axes.items():
            angle_rad = math.radians(axis["phase"])
            dist = 160 + axis["curvature"] * 80
            
            x = center[0] + math.cos(angle_rad) * dist
            y = center[1] + math.sin(angle_rad) * dist
            
            # The 'Causal Inverter' connection
            draw.line([center, (x, y)], fill=colors[name], width=2)
            
            # Vortex Node (The AC Wave projection)
            node_r = 8 + axis["curvature"] * 4
            draw.ellipse([x-node_r, y-node_r, x+node_r, y+node_r], fill=colors[name])
            
        # 4. GROUND STATUS (TIME AXIS)
        draw.text((10, 10), f"GROUND (TIME): {self.vortex.time_ground:.4f} s", fill=(255, 255, 255))
        draw.text((10, 30), f"MAXWELL ROT (CURL): ENABLED", fill=(0, 255, 255))
        draw.text((10, 50), f"RESONANCE: {1.0 - self.vortex.detect_distortion():.4f}", fill=(255, 255, 255))

        # Save Image
        filename = os.path.join(OUTPUT_DIR, f"vortex_4d_{iteration:04d}.png")
        img.save(filename)
        return filename

    def render_dimensional_projection(self, dim: int, data: List[Any], iteration: int):
        """
        Renders a specific dimension projected from the 4D Rotor.
        """
        size = (400, 400)
        img = Image.new("RGB", size, (20, 20, 20))
        draw = ImageDraw.Draw(img)
        
        if dim == 1:
            # Render 1D Wave
            points = []
            for i, val in enumerate(data):
                x = i * (size[0] / len(data))
                y = 200 + val * 100
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=(255, 255, 0), width=2)
            draw.text((10, 10), "PROJECTION: 1D LINE (Linear Flow)", fill=(255, 255, 255))

        elif dim == 2:
            # Render 2D Grid
            grid_size = len(data)
            cell_w = size[0] / grid_size
            for i, row in enumerate(data):
                for j, val in enumerate(row):
                    # Color based on value
                    c = int(127 + val * 127)
                    c = max(0, min(255, c))
                    draw.rectangle([j*cell_w, i*cell_w, (j+1)*cell_w, (i+1)*cell_w], fill=(0, c, c))
            draw.text((10, 10), "PROJECTION: 2D GRID (Manifold)", fill=(255, 255, 255))

        filename = os.path.join(OUTPUT_DIR, f"proj_{dim}d_{iteration:04d}.png")
        img.save(filename)
        return filename

    def render_resonance_interference(self, iteration: int):
        """
        Renders the interference between Pure and Noisy fields.
        """
        size = (512, 512)
        img = Image.new("RGB", size, (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        state = self.vortex.exhale()
        res_val = state["Resonance_Field"]
        
        # Draw the interference rings
        for r in range(10, 250, 5):
            # Interference modulates the color
            c_val = int(127 + 127 * math.sin(r * 0.1 + res_val * 10))
            c_val = max(0, min(255, c_val))
            
            color = (c_val, 0, 255 - c_val) if res_val > 0 else (0, c_val, 255)
            draw.ellipse([256-r, 256-r, 256+r, 256+r], outline=color)
            
        draw.text((10, 10), f"CROSS-DIMENSIONAL RESONANCE: {res_val:.4f}", fill=(255, 255, 255))
        if state["Is_Crystallized"]:
            draw.text((10, 30), "💎 CRYSTALLIZED", fill=(0, 255, 0))
            
        filename = os.path.join(OUTPUT_DIR, f"resonance_{iteration:04d}.png")
        img.save(filename)
        return filename

    def run_simulation(self, iterations: int = 27):
        """
        Runs the Inhale-Exhale loop.
        """
        print(f"🌟 Starting {iterations} iterations of the Triple Vortex...")
        dt = 0.1
        
        for i in range(iterations):
            # 1. INHALE: Inject random DC noise (Linear Causality)
            # simulate external load
            noise = [math.sin(i * 0.1 + j) for j in range(21)]
            self.vortex.inhale(noise, dt)
            
            # 2. PROCESS: Vortex Rotation & Convergence
            self.vortex.process_vortex(dt)
            
            # 3. SELF-HEAL: If distorted, snap back
            healed = self.vortex.self_heal(dt)
            if healed:
                print(f"   [Self-Healing] Resonance restored in iteration {i}")
            
            # 4. EXHALE: Project to White Hole
            state = self.vortex.exhale()
            
            # 5. RENDER: The Light Matrix
            if i % 3 == 0:
                self.render_vortex_state(i)
                self.render_resonance_interference(i)
                
                # [PHASE 369] TEST DIMENSIONAL GENESIS
                # Create a 'Soil of Points'
                soil = [1.0] * 100
                
                # Project 1D
                wave_1d = self.vortex.project_dimension(1, soil)
                self.render_dimensional_projection(1, wave_1d, i)
                
                # Project 2D
                grid_2d = self.vortex.project_dimension(2, soil)
                self.render_dimensional_projection(2, grid_2d, i)
                
                print(f"   📸 [Genesis] Projected 1D & 2D Manifolds at iteration {i}")
                
            time.sleep(0.05)

        print("\n✅ Simulation Complete. The Vortex has stabilized.")

if __name__ == "__main__":
    crystallizer = CosmicCrystallizer()
    crystallizer.run_simulation()
