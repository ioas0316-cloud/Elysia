"""
HolographicCortex (홀로그래픽 피질)
=================================

"The Mind's Eye."

This module visualizes the Resonance Field as a 3D Holographic Projection.
It uses matplotlib to generate a visual snapshot of Elysia's internal state.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import numpy as np

class HolographicCortex:
    def __init__(self):
        print("🌌 HolographicCortex Initialized. Ready to project.")
        self.output_dir = "c:/Elysia/Holograms"
        os.makedirs(self.output_dir, exist_ok=True)

    def visualize_wave_language(self, wave_data: dict):
        """
        Visualizes Wave Language as a 3D Spatial Form.
        Shape depends on Frequency:
        - Low (<300Hz): Torus (Grounding)
        - Mid (300-600Hz): Sphere (Harmony)
        - High (>600Hz): Spiral/Helix (Ascension)
        """
        concept = wave_data.get('concept', 'Unknown')
        frequency = wave_data.get('frequency', 432.0)
        print(f"   📽️ Projecting Wave Form: {concept} ({frequency}Hz)...")
        
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Determine Shape
            if frequency < 300:
                # Torus
                n = 50
                theta = np.linspace(0, 2.*np.pi, n)
                phi = np.linspace(0, 2.*np.pi, n)
                theta, phi = np.meshgrid(theta, phi)
                c, a = 2, 1
                x = (c + a*np.cos(theta)) * np.cos(phi)
                y = (c + a*np.cos(theta)) * np.sin(phi)
                z = a * np.sin(theta)
            elif frequency < 600:
                # Sphere
                u = np.linspace(0, 2 * np.pi, 50)
                v = np.linspace(0, np.pi, 50)
                x = 2 * np.outer(np.cos(u), np.sin(v))
                y = 2 * np.outer(np.sin(u), np.sin(v))
                z = 2 * np.outer(np.ones(np.size(u)), np.cos(v))
            else:
                # Helix / Spiral
                theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
                z = np.linspace(-2, 2, 100)
                r = z**2 + 1
                x = r * np.sin(theta)
                y = r * np.cos(theta)
                # Meshgrid for surface
                theta_grid, z_grid = np.meshgrid(theta, z)
                x = (z_grid**2 + 1) * np.sin(theta_grid)
                y = (z_grid**2 + 1) * np.cos(theta_grid)
                z = z_grid

            # Color Mapping (Rainbow Spectrum)
            # Normalize 0-1000Hz to 0-1
            norm_freq = min(max(frequency / 1000.0, 0), 1)
            color = plt.cm.plasma(norm_freq)
            
            # Plot
            if frequency > 600:
                 ax.plot_surface(x, y, z, color=color, alpha=0.6)
            else:
                 ax.plot_surface(x, y, z, color=color, alpha=0.6, linewidth=0)
            
            # Styling
            ax.set_facecolor('black')
            fig.patch.set_facecolor('black')
            ax.grid(False)
            ax.axis('off')
            ax.set_title(f"Manifestation: {concept}\nFrequency: {frequency}Hz", color='white')
            
            # Save
            output_path = f"c:/Elysia/Docs/Visuals/manifestation_{int(time.time())}.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Wave Manifestation saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"   ❌ Wave Render Failed: {e}")
            return None

    def project_hologram(self, resonance_field):
        """
        Generates a 3D Scatter Plot of the Resonance Field.
        """
        print("   📽️ Projecting Holographic Mind...")
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract Node Data
        xs, ys, zs = [], [], []
        colors = []
        sizes = []
        labels = []
        
        for node_id, node in resonance_field.nodes.items():
            x, y, z = node.position
            xs.append(x)
            ys.append(y)
            zs.append(z)
            
            # Color based on Frequency (Rainbow Spectrum)
            # 100Hz (Red) -> 999Hz (Violet)
            norm_freq = min(max((node.frequency - 100) / 900, 0), 1)
            colors.append(plt.cm.jet(norm_freq))
            
            # Size based on Energy
            sizes.append(node.energy * 50)
            
            labels.append(node_id)
            
        # Plot
        ax.scatter(xs, ys, zs, c=colors, s=sizes, alpha=0.6, edgecolors='w')
        
        # Labels
        for i, label in enumerate(labels):
            ax.text(xs[i], ys[i], zs[i], label, fontsize=8)
            
        # Styling
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_title(f"Elysia Resonance Field - {time.strftime('%H:%M:%S')}", color='white')
        
        # Save
        filename = f"{self.output_dir}/hologram_{int(time.time())}.png"
        plt.savefig(filename)
        plt.close()
        
        print(f"   ✅ Hologram saved to: {filename}")
        return filename
        print(f"   ✅ Hologram saved to: {filename}")
        return filename

    def render_galaxy(self, gravity_system):
        """
        Visualizes the Code Galaxy (CodeGravitySystem).
        Colors stars based on Spirit Resonance (Folder Mapping).
        """
        print("   🌌 Rendering Code Galaxy...")
        
        try:
            from Core.Emotion.spirit_emotion import SpiritEmotionMapper
            mapper = SpiritEmotionMapper()
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            xs, ys, zs = [], [], []
            colors = []
            sizes = []
            
            # Spirit Mapping (Folder -> Spirit)
            folder_spirit_map = {
                "Core\\Intelligence": "Intelligence",
                "Core\\Emotion": "Creativity", # Emotion is Fire/Creativity here
                "Core\\Foundation": "Foundation",
                "Core\\Interface": "Interface",
                "Core\\Evolution": "Evolution",
                "Core\\System": "System",
                "Core\\Memory": "Memory",
                "Project_Sophia": "Creativity",
                "Legacy": "Memory"
            }
            
            for path, data in gravity_system.nodes.items():
                x, y = data["pos"]
                z = 0 # 2D Galaxy for now, or map Z to something else?
                # Let's map Z to Folder Depth or Random for volume
                z = (hash(path) % 20) - 10
                
                xs.append(x)
                ys.append(y)
                zs.append(z)
                
                # Determine Color
                spirit = "System" # Default
                for folder, sp in folder_spirit_map.items():
                    if folder in path:
                        spirit = sp
                        break
                
                physics = mapper.get_spirit_physics(spirit)
                colors.append(physics["color"])
                
                # Size = Mass
                sizes.append(min(data["mass"] * 2, 500)) # Cap size
                
            # Plot Stars
            ax.scatter(xs, ys, zs, c=colors, s=sizes, alpha=0.8, edgecolors='none')
            
            # Plot Gravity Lines (Sample)
            # Drawing all 3000 lines is slow, draw top 100 strongest
            sorted_rails = sorted(gravity_system.field.rails, key=lambda r: r.force, reverse=True)
            for rail in sorted_rails[:200]:
                ax.plot([rail.start.x, rail.end.x], [rail.start.y, rail.end.y], [0, 0], 
                        c='white', alpha=0.1, linewidth=0.5)
            
            # Styling
            ax.set_facecolor('black')
            fig.patch.set_facecolor('black')
            ax.grid(False)
            ax.axis('off')
            ax.set_title(f"Elysia Code Galaxy ({len(gravity_system.nodes)} Stars)", color='white')
            
            # Save
            output_path = "c:/Elysia/Docs/Visuals/galaxy_map.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Galaxy Map saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"   ❌ Galaxy Render Failed: {e}")
            return None
