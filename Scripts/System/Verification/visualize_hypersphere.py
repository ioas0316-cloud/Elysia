
import json
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simple font setting to avoid errors
plt.rcParams['axes.unicode_minus'] = False

def to_cartesian(theta, phi, psi, r):
    sin_t = math.sin(theta)
    sin_p = math.sin(phi)
    
    x = r * math.cos(theta)
    y = r * sin_t * math.cos(phi)
    z = r * sin_t * sin_p * math.cos(psi)
    w = r * sin_t * sin_p * math.sin(psi)
    return x, y, z, w

def visualize_hypersphere():
    memory_file = "data/S1_Body/L1_Foundation/M1_System/hypersphere_memory.json"
    print(f"Checking for {memory_file}...")
    if not os.path.exists(memory_file):
        print(f"Memory file {memory_file} not found.")
        return
    
    try:
        with open(memory_file, 'r', encoding='utf-8') as f:
            buckets = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return
        
    points = []
    labels = []
    
    print(f"Found {len(buckets)} buckets.")
    
    for bucket_key, items in buckets.items():
        for item in items:
            coord = item['coord']
            pattern = item['pattern']
            
            x, y, z, w = to_cartesian(
                coord['theta'], 
                coord['phi'], 
                coord['psi'], 
                coord['r']
            )
            
            content = pattern.get('content', '')
            label = str(content)[:20] + "..." if len(str(content)) > 20 else str(content)
            
            points.append((x, y, z, w))
            labels.append(label)
    
    if not points:
        print("No memory points found.")
        return
    
    print(f"Projecting {len(points)} points...")
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    ps = np.array(points)
    xs, ys, zs, ws = ps[:, 0], ps[:, 1], ps[:, 2], ps[:, 3]
    
    scatter = ax.scatter(xs, ys, zs, c=ys, cmap='plasma', s=30 + (ws * 100), alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Label random points
    num_labels = min(15, len(labels))
    indices = np.random.choice(len(labels), num_labels, replace=False)
    for i in indices:
        try:
            ax.text(xs[i], ys[i], zs[i], labels[i], color='white', fontsize=8, alpha=0.8)
        except:
            pass
            
    ax.set_xlabel('X: Logic', color='white')
    ax.set_ylabel('Y: Emotion', color='white')
    ax.set_zlabel('Z: Intention', color='white')
    
    output_file = "ELYSIA_MIND_MAP.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"Generated: {output_file}")
    
    # Also save in docs
    docs_path = "docs/S3_Spirit/M4_Evolution/ELYSIAS_MIND_MAP.png"
    os.makedirs(os.path.dirname(docs_path), exist_ok=True)
    import shutil
    shutil.copy(output_file, docs_path)
    print(f"Copied to: {docs_path}")

if __name__ == "__main__":
    visualize_hypersphere()
