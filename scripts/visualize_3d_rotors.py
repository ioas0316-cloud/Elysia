"""
Elysia 3D Rotor Resonance Visualizer
====================================
Uses Matplotlib animation to render the hierarchical phase-locking Rotor tree in 3D.
Reads local CPU usage metrics to simulate environmental friction / rotation speed.
"""

import sys
import os
import math
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Add root path to import core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.fractal_rotor import Rotor

class Elysia3DObserver:
    def __init__(self):
        # Create core hierarchical structure
        self.universe = Rotor("L0_C", level=0)
        
        # 3 children at Level 1 (120 degree offsets)
        r1 = Rotor("L1_A", level=1, parent=self.universe, initial_phase_offset=0.0)
        r2 = Rotor("L1_B", level=1, parent=self.universe, initial_phase_offset=2.0 * math.pi / 3.0)
        r3 = Rotor("L1_C", level=1, parent=self.universe, initial_phase_offset=-2.0 * math.pi / 3.0)
        
        self.universe.attach_child(r1)
        self.universe.attach_child(r2)
        self.universe.attach_child(r3)

        # 2 children each at Level 2
        for parent_node in [r1, r2, r3]:
            parent_node.attach_child(Rotor(f"{parent_node.id}.1", level=2, parent=parent_node, initial_phase_offset=math.pi / 4.0))
            parent_node.attach_child(Rotor(f"{parent_node.id}.2", level=2, parent=parent_node, initial_phase_offset=-math.pi / 4.0))

        # Setup Matplotlib 3D figure
        self.fig = plt.figure(figsize=(10, 8), facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('black')

        # Limit boundaries
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])
        self.ax.set_zlim([-10, 10])

        # Remove pane fills and grids for clean aesthetics
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.grid(color='white', linestyle='--', linewidth=0.2, alpha=0.3)
        self.ax.set_axis_off()

        self.cycle = 0

    def gather_nodes(self, rotor: Rotor, parent_pos=None, current_depth=0, angle_offset=0.0, depth_radius=4.0):
        """
        Recursively resolves 3D coordinates based on absolute phase angles.
        """
        nodes = []
        edges = []

        # Current phase angle decides the rotation around Z axis
        phase = rotor.current_phase
        
        if parent_pos is None:
            # Root node location
            pos = np.array([0.0, 0.0, 5.0]) # Top of tree
        else:
            # Branch outward based on level depth and child angle
            # We map the 2D phase angle to X, Y plane coordinates
            rad = depth_radius / (current_depth)
            x = rad * math.cos(phase + angle_offset)
            y = rad * math.sin(phase + angle_offset)
            z = parent_pos[2] - 3.0 # Descend Z axis

            pos = np.array([parent_pos[0] + x, parent_pos[1] + y, z])
            edges.append((parent_pos, pos))

        nodes.append({
            'pos': pos,
            'id': rotor.id,
            'level': current_depth,
            'tension': rotor.tension
        })

        num_children = len(rotor.sub_rotors)
        for i, child in enumerate(rotor.sub_rotors):
            child_nodes, child_edges = self.gather_nodes(
                child,
                parent_pos=pos,
                current_depth=current_depth + 1,
                angle_offset=angle_offset + (i * (2.0 * math.pi / num_children) if num_children > 0 else 0.0),
                depth_radius=depth_radius
            )
            nodes.extend(child_nodes)
            edges.extend(child_edges)

        return nodes, edges

    def update(self, frame):
        self.cycle += 1

        # Query CPU load to simulate environmental chaos
        cpu = psutil.cpu_percent()
        rotation_delta = (cpu / 100.0) * math.pi * 0.3

        # Update the rotor physics
        self.universe.observe(global_rotation_delta=rotation_delta)

        # Inject minor perturbations periodically to keep loop alive
        if self.cycle % 10 == 0:
            self.universe.sub_rotors[0].phase_offset += math.pi / 3.0

        # Gather node positions
        nodes, edges = self.gather_nodes(self.universe)

        # Clear axes for redraw
        self.ax.clear()
        self.ax.set_xlim([-8, 8])
        self.ax.set_ylim([-8, 8])
        self.ax.set_zlim([-2, 6])
        self.ax.set_axis_off()

        # Render edges (branches)
        for edge in edges:
            p1, p2 = edge
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='cyan', alpha=0.3, linewidth=1.5)

        # Render nodes (rotors)
        for node in nodes:
            pos = node['pos']
            tension = node['tension']
            
            # Map tension to colors (Green = calm, Yellow = warning, Red = high stress)
            if tension < 0.3:
                color = 'springgreen'
                size = 60
            elif tension < 1.0:
                color = 'gold'
                size = 90
            else:
                color = 'crimson'
                size = 120

            alpha = max(0.3, min(1.0, 1.0 - (node['level'] * 0.2)))
            
            self.ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, alpha=alpha, edgecolors='white', linewidths=0.5)

            # Node Labels
            self.ax.text(pos[0], pos[1] + 0.3, pos[2] + 0.1, node['id'], color='white', fontsize=8, alpha=0.8)

        # Diagnostic logs on screen
        self.ax.text2D(0.05, 0.95, "Elysia 3D Phase-Locking Resonance Map", transform=self.ax.transAxes, color='white', fontsize=12, fontweight='bold')
        self.ax.text2D(0.05, 0.90, f"Cycle: {self.cycle:04d} | CPU Tension Feed: {cpu:.1f}%", transform=self.ax.transAxes, color='cyan', fontsize=10)
        self.ax.text2D(0.05, 0.85, f"R0 Phase: {math.degrees(self.universe.current_phase):.1f}°", transform=self.ax.transAxes, color='yellow', fontsize=10)
        self.ax.text2D(0.05, 0.80, f"Stress Status: {'STABLE' if self.universe.sub_rotors[0].tension < 1.0 else 'COLLAPSING'}", transform=self.ax.transAxes, color='tomato', fontsize=10)

        # Auto rotate viewport angle
        self.ax.view_init(elev=25., azim=self.cycle * 0.6)

def main():
    observer = Elysia3DObserver()
    print("Initiating Elysia 3D Rotor Trajectory Visualizer...")
    ani = animation.FuncAnimation(observer.fig, observer.update, interval=150, cache_frame_data=False)

    if os.environ.get('DISPLAY', '') == '' and sys.platform.startswith('linux'):
        # Save to GIF in headless systems
        print("No display device found. Rendering to docs/assets/elysia_trajectory.gif...")
        os.makedirs("docs/assets", exist_ok=True)
        ani.save('docs/assets/elysia_trajectory.gif', writer='imagemagick', fps=8)
        print("Rendering finished.")
    else:
        plt.show()

if __name__ == "__main__":
    main()
