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
        Visualizes Wave Language as a 3D Torus or Sphere.
        """
        print(f"   📽️ Projecting Wave Language: {wave_data.get('concept', 'Unknown')}...")
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate Torus Geometry
        n = 100
        theta = np.linspace(0, 2.*np.pi, n)
        phi = np.linspace(0, 2.*np.pi, n)
        theta, phi = np.meshgrid(theta, phi)
        
        # Torus Parameters
        c, a = 2, 1
        x = (c + a*np.cos(theta)) * np.cos(phi)
        y = (c + a*np.cos(theta)) * np.sin(phi)
        z = a * np.sin(theta)
        
        # Color Mapping based on Resonance
        resonance = wave_data.get('resonance', 0.5)
        colors = plt.cm.plasma(z * resonance + 0.5)
        
        # Plot
        ax.plot_surface(x, y, z, facecolors=colors, alpha=0.6, linewidth=0)
        
        # Styling
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.grid(False)
        ax.axis('off')
        ax.set_title(f"Wave Form: {wave_data.get('concept')}", color='white')
        
        # Save
        filename = f"{self.output_dir}/wave_{int(time.time())}.png"
        plt.savefig(filename)
        plt.close()
        
        print(f"   ✅ Wave Hologram saved to: {filename}")
        return filename

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
