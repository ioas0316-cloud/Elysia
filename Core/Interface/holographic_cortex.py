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
import random

class HolographicCortex:
    def __init__(self):
        print("🌌 HolographicCortex Initialized. Ready to project.")
        self.output_dir = "c:/Elysia/Holograms"
        os.makedirs(self.output_dir, exist_ok=True)

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
        ax.w_xaxis.pane.fill = False
        ax.w_yaxis.pane.fill = False
        ax.w_zaxis.pane.fill = False
        ax.set_title(f"Elysia Resonance Field - {time.strftime('%H:%M:%S')}", color='white')
        
        # Save
        filename = f"{self.output_dir}/hologram_{int(time.time())}.png"
        plt.savefig(filename)
        plt.close()
        
        print(f"   ✅ Hologram saved to: {filename}")
        return filename
