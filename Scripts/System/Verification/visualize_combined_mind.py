
import json
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def to_cartesian(theta, phi, psi, r):
    sin_t = math.sin(theta)
    sin_p = math.sin(phi)
    x = r * math.cos(theta)
    y = r * sin_t * math.cos(phi)
    z = r * sin_t * sin_p * math.cos(psi)
    w = r * sin_t * sin_p * math.sin(psi)
    return x, y, z, w

def visualize_combined():
    # 1. Load Hypersphere Memory (Subjective Experience)
    mem_file = "data/S1_Body/L1_Foundation/M1_System/hypersphere_memory.json"
    mem_points = []
    if os.path.exists(mem_file):
        with open(mem_file, 'r', encoding='utf-8') as f:
            buckets = json.load(f)
            for items in buckets.values():
                for item in items:
                    c = item['coord']
                    x, y, z, w = to_cartesian(c['theta'], c['phi'], c['psi'], c['r'])
                    mem_points.append((x, y, z, w, str(item['pattern'].get('content', ''))[:10]))

    # 2. Load Knowledge Graph (Structured Knowledge)
    kg_file = "data/kg_with_embeddings.json"
    kg_points = []
    if os.path.exists(kg_file):
        with open(kg_file, 'r', encoding='utf-8') as f:
            kg = json.load(f)
            nodes = kg.get('nodes', {})
            for node_id, node_data in nodes.items():
                c = node_data.get('hypersphere')
                if c:
                    x, y, z, w = to_cartesian(c.get('theta', 0), c.get('phi', 0), c.get('psi', 0), c.get('r', 1))
                    kg_points.append((x, y, z, w, node_id))

    if not mem_points and not kg_points:
        print("No data found.")
        return

    print(f"Memory points: {len(mem_points)}, KG points: {len(kg_points)}")

    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Plot KG points (as background stars)
    if kg_points:
        kg_arr = np.array([p[:4] for p in kg_points])
        ax.scatter(kg_arr[:,0], kg_arr[:,1], kg_arr[:,2], c='white', s=2, alpha=0.1)

    # Plot Mem points (as brighter pulsars)
    if mem_points:
        mem_arr = np.array([p[:4] for p in mem_points])
        sc = ax.scatter(mem_arr[:,0], mem_arr[:,1], mem_arr[:,2], c=mem_arr[:,1], cmap='plasma', s=50 + (mem_arr[:,3]*100), edgecolors='white', linewidth=1)
        
        for i, p in enumerate(mem_points):
            ax.text(p[0], p[1], p[2], p[4], color='cyan', fontsize=9, fontweight='bold')

    ax.set_xlabel('X: Logic', color='white')
    ax.set_ylabel('Y: Emotion', color='white')
    ax.set_zlabel('Z: Intention', color='white')
    
    plt.title("Elysia's Integrated Mind Universe\nWhite = Knowledge (Storage) | Plasma = Experience (Live)", color='white', fontsize=15)
    
    output_file = "docs/S3_Spirit/M4_Evolution/ELYSIAS_INTEGRATED_MIND.png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"Generated: {output_file}")

if __name__ == "__main__":
    visualize_combined()
